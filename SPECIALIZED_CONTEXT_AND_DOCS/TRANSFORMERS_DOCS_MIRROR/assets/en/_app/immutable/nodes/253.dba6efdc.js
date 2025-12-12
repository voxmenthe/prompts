import{s as Qt,o as Kt,n as eo}from"../chunks/scheduler.18a86fab.js";import{S as to,i as oo,g as i,s as n,r as m,A as no,h as l,f as o,c as r,j as L,x as d,u as p,k as v,y as s,a,v as u,d as f,t as _,w as g}from"../chunks/index.98837b22.js";import{T as ro}from"../chunks/Tip.77304350.js";import{D as z}from"../chunks/Docstring.a1ef7999.js";import{C as Yt}from"../chunks/CodeBlock.8d0c2e8a.js";import{H as $e,E as so}from"../chunks/getInferenceSnippets.06c2775f.js";function ao(qe){let k,C='As LayoutXLM’s architecture is equivalent to that of LayoutLMv2, one can refer to <a href="layoutlmv2">LayoutLMv2’s documentation page</a> for all tips, code examples and notebooks.';return{c(){k=i("p"),k.innerHTML=C},l(x){k=l(x,"P",{"data-svelte-h":!0}),d(k)!=="svelte-t8okno"&&(k.innerHTML=C)},m(x,de){a(x,k,de)},p:eo,d(x){x&&o(k)}}}function io(qe){let k,C,x,de,D,Mt="<em>This model was released on 2021-04-18 and added to Hugging Face Transformers on 2021-11-03.</em>",Fe,W,Pe,$,zt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Ne,E,Ie,U,$t=`LayoutXLM was proposed in <a href="https://huggingface.co/papers/2104.08836" rel="nofollow">LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding</a> by Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha
Zhang, Furu Wei. It’s a multilingual extension of the <a href="https://huggingface.co/papers/2012.14740" rel="nofollow">LayoutLMv2 model</a> trained
on 53 languages.`,Ce,H,Xt="The abstract from the paper is the following:",De,S,qt=`<em>Multimodal pre-training with text, layout, and image has achieved SOTA performance for visually-rich document
understanding tasks recently, which demonstrates the great potential for joint learning across different modalities. In
this paper, we present LayoutXLM, a multimodal pre-trained model for multilingual document understanding, which aims to
bridge the language barriers for visually-rich document understanding. To accurately evaluate LayoutXLM, we also
introduce a multilingual form understanding benchmark dataset named XFUN, which includes form understanding samples in
7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese), and key-value pairs are manually labeled
for each language. Experiment results show that the LayoutXLM model has significantly outperformed the existing SOTA
cross-lingual pre-trained models on the XFUN dataset.</em>`,We,O,Ft='This model was contributed by <a href="https://huggingface.co/nielsr" rel="nofollow">nielsr</a>. The original code can be found <a href="https://github.com/microsoft/unilm" rel="nofollow">here</a>.',Ee,A,Ue,R,Pt="One can directly plug in the weights of LayoutXLM into a LayoutLMv2 model, like so:",He,V,Se,j,Nt=`Note that LayoutXLM has its own tokenizer, based on
<a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer">LayoutXLMTokenizer</a>/<a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast">LayoutXLMTokenizerFast</a>. You can initialize it as
follows:`,Oe,B,Ae,Z,It=`Similar to LayoutLMv2, you can use <a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMProcessor">LayoutXLMProcessor</a> (which internally applies
<a href="/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor">LayoutLMv2ImageProcessor</a> and
<a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer">LayoutXLMTokenizer</a>/<a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast">LayoutXLMTokenizerFast</a> in sequence) to prepare all
data for the model.`,Re,X,Ve,G,je,c,J,rt,ce,Ct=`Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.`,st,me,Dt=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,at,q,Y,it,pe,Wt=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.`,lt,T,Q,dt,ue,Et=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLM-RoBERTa sequence has the following format:`,ct,fe,Ut="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",mt,F,K,pt,_e,Ht=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,ut,P,ee,ft,ge,St=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
not make use of token type ids, therefore a list of zeros is returned.`,_t,he,te,Be,oe,Ze,y,ne,gt,ke,Ot=`Construct a “fast” LayoutXLM tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Adapted from
<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models" rel="nofollow">BPE</a>.`,ht,be,At=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,kt,N,re,bt,ye,Rt=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.`,Ge,se,Je,b,ae,yt,ve,Vt=`Constructs a LayoutXLM processor which combines a LayoutXLM image processor and a LayoutXLM tokenizer into a single
processor.`,vt,Le,jt='<a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMProcessor">LayoutXLMProcessor</a> offers all the functionalities you need to prepare data for the model.',Lt,xe,Bt=`It first uses <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor">LayoutLMv2ImageProcessor</a> to resize document images to a fixed size, and optionally applies OCR to
get words and normalized bounding boxes. These are then provided to <a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer">LayoutXLMTokenizer</a> or
<a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast">LayoutXLMTokenizerFast</a>, which turns the words and bounding boxes into token-level <code>input_ids</code>,
<code>attention_mask</code>, <code>token_type_ids</code>, <code>bbox</code>. Optionally, one can provide integer <code>word_labels</code>, which are turned
into token-level <code>labels</code> for token classification tasks (such as FUNSD, CORD).`,xt,w,ie,Tt,Te,Zt='This method first forwards the <code>images</code> argument to <code>~LayoutLMv2ImagePrpcessor.__call__</code>. In case\n<code>LayoutLMv2ImagePrpcessor</code> was initialized with <code>apply_ocr</code> set to <code>True</code>, it passes the obtained words and\nbounding boxes along with the additional arguments to <a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer.__call__"><strong>call</strong>()</a> and returns the output,\ntogether with resized <code>images</code>. In case <code>LayoutLMv2ImagePrpcessor</code> was initialized with <code>apply_ocr</code> set to\n<code>False</code>, it passes the words (<code>text</code>/<code>text_pair`) and `boxes` specified by the user along with the additional arguments to [__call__()](/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer.__call__) and returns the output, together with resized `images</code>.',wt,we,Gt="Please refer to the docstring of the above two methods for more information.",Ye,le,Qe,Xe,Ke;return W=new $e({props:{title:"LayoutXLM",local:"layoutxlm",headingTag:"h1"}}),E=new $e({props:{title:"Overview",local:"overview",headingTag:"h2"}}),A=new $e({props:{title:"Usage tips and examples",local:"usage-tips-and-examples",headingTag:"h2"}}),V=new Yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExheW91dExNdjJNb2RlbCUwQSUwQW1vZGVsJTIwJTNEJTIwTGF5b3V0TE12Mk1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZsYXlvdXR4bG0tYmFzZSUyMik=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMv2Model

model = LayoutLMv2Model.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutxlm-base&quot;</span>)`,wrap:!1}}),B=new Yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExheW91dFhMTVRva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMExheW91dFhMTVRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbGF5b3V0eGxtLWJhc2UlMjIp",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutXLMTokenizer

tokenizer = LayoutXLMTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutxlm-base&quot;</span>)`,wrap:!1}}),X=new ro({props:{$$slots:{default:[ao]},$$scope:{ctx:qe}}}),G=new $e({props:{title:"LayoutXLMTokenizer",local:"transformers.LayoutXLMTokenizer",headingTag:"h2"}}),J=new z({props:{name:"class transformers.LayoutXLMTokenizer",anchor:"transformers.LayoutXLMTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"cls_token_box",val:" = [0, 0, 0, 0]"},{name:"sep_token_box",val:" = [1000, 1000, 1000, 1000]"},{name:"pad_token_box",val:" = [0, 0, 0, 0]"},{name:"pad_token_label",val:" = -100"},{name:"only_label_first_subword",val:" = True"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutXLMTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.LayoutXLMTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.LayoutXLMTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.LayoutXLMTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.LayoutXLMTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.LayoutXLMTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LayoutXLMTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.LayoutXLMTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.LayoutXLMTokenizer.cls_token_box",description:`<strong>cls_token_box</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [CLS] token.`,name:"cls_token_box"},{anchor:"transformers.LayoutXLMTokenizer.sep_token_box",description:`<strong>sep_token_box</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[1000, 1000, 1000, 1000]</code>) &#x2014;
The bounding box to use for the special [SEP] token.`,name:"sep_token_box"},{anchor:"transformers.LayoutXLMTokenizer.pad_token_box",description:`<strong>pad_token_box</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [PAD] token.`,name:"pad_token_box"},{anchor:"transformers.LayoutXLMTokenizer.pad_token_label",description:`<strong>pad_token_label</strong> (<code>int</code>, <em>optional</em>, defaults to -100) &#x2014;
The label to use for padding tokens. Defaults to -100, which is the <code>ignore_index</code> of PyTorch&#x2019;s
CrossEntropyLoss.`,name:"pad_token_label"},{anchor:"transformers.LayoutXLMTokenizer.only_label_first_subword",description:`<strong>only_label_first_subword</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to only label the first subword, in case word labels are provided.`,name:"only_label_first_subword"},{anchor:"transformers.LayoutXLMTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.LayoutXLMTokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L148"}}),Y=new z({props:{name:"__call__",anchor:"transformers.LayoutXLMTokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]]]"},{name:"text_pair",val:": typing.Union[list[str], list[list[str]], NoneType] = None"},{name:"boxes",val:": typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None"},{name:"word_labels",val:": typing.Union[list[int], list[list[int]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutXLMTokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
(words of a single example or questions of a batch of examples) or a list of list of strings (batch of
words).`,name:"text"},{anchor:"transformers.LayoutXLMTokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
(pretokenized string).`,name:"text_pair"},{anchor:"transformers.LayoutXLMTokenizer.__call__.boxes",description:`<strong>boxes</strong> (<code>list[list[int]]</code>, <code>list[list[list[int]]]</code>) &#x2014;
Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.`,name:"boxes"},{anchor:"transformers.LayoutXLMTokenizer.__call__.word_labels",description:`<strong>word_labels</strong> (<code>list[int]</code>, <code>list[list[int]]</code>, <em>optional</em>) &#x2014;
Word-level integer labels (for token classification tasks such as FUNSD, CORD).`,name:"word_labels"},{anchor:"transformers.LayoutXLMTokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to encode the sequences with the special tokens relative to their model.`,name:"add_special_tokens"},{anchor:"transformers.LayoutXLMTokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LayoutXLMTokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.LayoutXLMTokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LayoutXLMTokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LayoutXLMTokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability <code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LayoutXLMTokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.LayoutXLMTokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.LayoutXLMTokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.LayoutXLMTokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.LayoutXLMTokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.LayoutXLMTokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.LayoutXLMTokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.LayoutXLMTokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.LayoutXLMTokenizer.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L439",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>bbox</strong> — List of bounding boxes to be fed to a model.</p>
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
<p><strong>labels</strong> — List of labels to be fed to a model. (when <code>word_labels</code> is specified).</p>
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
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>).</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),Q=new z({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.LayoutXLMTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.LayoutXLMTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.LayoutXLMTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L311",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),K=new z({props:{name:"get_special_tokens_mask",anchor:"transformers.LayoutXLMTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.LayoutXLMTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.LayoutXLMTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.LayoutXLMTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L337",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ee=new z({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.LayoutXLMTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.LayoutXLMTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.LayoutXLMTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L365",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),te=new z({props:{name:"save_vocabulary",anchor:"transformers.LayoutXLMTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm.py#L422"}}),oe=new $e({props:{title:"LayoutXLMTokenizerFast",local:"transformers.LayoutXLMTokenizerFast",headingTag:"h2"}}),ne=new z({props:{name:"class transformers.LayoutXLMTokenizerFast",anchor:"transformers.LayoutXLMTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"cls_token_box",val:" = [0, 0, 0, 0]"},{name:"sep_token_box",val:" = [1000, 1000, 1000, 1000]"},{name:"pad_token_box",val:" = [0, 0, 0, 0]"},{name:"pad_token_label",val:" = -100"},{name:"only_label_first_subword",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutXLMTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.LayoutXLMTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.LayoutXLMTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.LayoutXLMTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.LayoutXLMTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.LayoutXLMTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LayoutXLMTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.LayoutXLMTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.LayoutXLMTokenizerFast.cls_token_box",description:`<strong>cls_token_box</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [CLS] token.`,name:"cls_token_box"},{anchor:"transformers.LayoutXLMTokenizerFast.sep_token_box",description:`<strong>sep_token_box</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[1000, 1000, 1000, 1000]</code>) &#x2014;
The bounding box to use for the special [SEP] token.`,name:"sep_token_box"},{anchor:"transformers.LayoutXLMTokenizerFast.pad_token_box",description:`<strong>pad_token_box</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[0, 0, 0, 0]</code>) &#x2014;
The bounding box to use for the special [PAD] token.`,name:"pad_token_box"},{anchor:"transformers.LayoutXLMTokenizerFast.pad_token_label",description:`<strong>pad_token_label</strong> (<code>int</code>, <em>optional</em>, defaults to -100) &#x2014;
The label to use for padding tokens. Defaults to -100, which is the <code>ignore_index</code> of PyTorch&#x2019;s
CrossEntropyLoss.`,name:"pad_token_label"},{anchor:"transformers.LayoutXLMTokenizerFast.only_label_first_subword",description:`<strong>only_label_first_subword</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to only label the first subword, in case word labels are provided.`,name:"only_label_first_subword"},{anchor:"transformers.LayoutXLMTokenizerFast.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&quot;&lt;s&gt;NOTUSED&quot;, &quot;&lt;/s&gt;NOTUSED&quot;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm_fast.py#L149"}}),re=new z({props:{name:"__call__",anchor:"transformers.LayoutXLMTokenizerFast.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]]]"},{name:"text_pair",val:": typing.Union[list[str], list[list[str]], NoneType] = None"},{name:"boxes",val:": typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None"},{name:"word_labels",val:": typing.Union[list[int], list[list[int]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutXLMTokenizerFast.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
(words of a single example or questions of a batch of examples) or a list of list of strings (batch of
words).`,name:"text"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.text_pair",description:`<strong>text_pair</strong> (<code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
(pretokenized string).`,name:"text_pair"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.boxes",description:`<strong>boxes</strong> (<code>list[list[int]]</code>, <code>list[list[list[int]]]</code>) &#x2014;
Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.`,name:"boxes"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.word_labels",description:`<strong>word_labels</strong> (<code>list[int]</code>, <code>list[list[int]]</code>, <em>optional</em>) &#x2014;
Word-level integer labels (for token classification tasks such as FUNSD, CORD).`,name:"word_labels"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to encode the sequences with the special tokens relative to their model.`,name:"add_special_tokens"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability <code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.LayoutXLMTokenizerFast.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/tokenization_layoutxlm_fast.py#L263",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>bbox</strong> — List of bounding boxes to be fed to a model.</p>
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
<p><strong>labels</strong> — List of labels to be fed to a model. (when <code>word_labels</code> is specified).</p>
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
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>).</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),se=new $e({props:{title:"LayoutXLMProcessor",local:"transformers.LayoutXLMProcessor",headingTag:"h2"}}),ae=new z({props:{name:"class transformers.LayoutXLMProcessor",anchor:"transformers.LayoutXLMProcessor",parameters:[{name:"image_processor",val:" = None"},{name:"tokenizer",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutXLMProcessor.image_processor",description:`<strong>image_processor</strong> (<code>LayoutLMv2ImageProcessor</code>, <em>optional</em>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor">LayoutLMv2ImageProcessor</a>. The image processor is a required input.`,name:"image_processor"},{anchor:"transformers.LayoutXLMProcessor.tokenizer",description:`<strong>tokenizer</strong> (<code>LayoutXLMTokenizer</code> or <code>LayoutXLMTokenizerFast</code>, <em>optional</em>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizer">LayoutXLMTokenizer</a> or <a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast">LayoutXLMTokenizerFast</a>. The tokenizer is a required input.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/processing_layoutxlm.py#L27"}}),ie=new z({props:{name:"__call__",anchor:"transformers.LayoutXLMProcessor.__call__",parameters:[{name:"images",val:""},{name:"text",val:": typing.Union[str, list[str], list[list[str]]] = None"},{name:"text_pair",val:": typing.Union[list[str], list[list[str]], NoneType] = None"},{name:"boxes",val:": typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None"},{name:"word_labels",val:": typing.Union[list[int], list[list[int]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutxlm/processing_layoutxlm.py#L68"}}),le=new so({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/layoutxlm.md"}}),{c(){k=i("meta"),C=n(),x=i("p"),de=n(),D=i("p"),D.innerHTML=Mt,Fe=n(),m(W.$$.fragment),Pe=n(),$=i("div"),$.innerHTML=zt,Ne=n(),m(E.$$.fragment),Ie=n(),U=i("p"),U.innerHTML=$t,Ce=n(),H=i("p"),H.textContent=Xt,De=n(),S=i("p"),S.innerHTML=qt,We=n(),O=i("p"),O.innerHTML=Ft,Ee=n(),m(A.$$.fragment),Ue=n(),R=i("p"),R.textContent=Pt,He=n(),m(V.$$.fragment),Se=n(),j=i("p"),j.innerHTML=Nt,Oe=n(),m(B.$$.fragment),Ae=n(),Z=i("p"),Z.innerHTML=It,Re=n(),m(X.$$.fragment),Ve=n(),m(G.$$.fragment),je=n(),c=i("div"),m(J.$$.fragment),rt=n(),ce=i("p"),ce.innerHTML=Ct,st=n(),me=i("p"),me.innerHTML=Dt,at=n(),q=i("div"),m(Y.$$.fragment),it=n(),pe=i("p"),pe.textContent=Wt,lt=n(),T=i("div"),m(Q.$$.fragment),dt=n(),ue=i("p"),ue.textContent=Et,ct=n(),fe=i("ul"),fe.innerHTML=Ut,mt=n(),F=i("div"),m(K.$$.fragment),pt=n(),_e=i("p"),_e.innerHTML=Ht,ut=n(),P=i("div"),m(ee.$$.fragment),ft=n(),ge=i("p"),ge.textContent=St,_t=n(),he=i("div"),m(te.$$.fragment),Be=n(),m(oe.$$.fragment),Ze=n(),y=i("div"),m(ne.$$.fragment),gt=n(),ke=i("p"),ke.innerHTML=Ot,ht=n(),be=i("p"),be.innerHTML=At,kt=n(),N=i("div"),m(re.$$.fragment),bt=n(),ye=i("p"),ye.textContent=Rt,Ge=n(),m(se.$$.fragment),Je=n(),b=i("div"),m(ae.$$.fragment),yt=n(),ve=i("p"),ve.textContent=Vt,vt=n(),Le=i("p"),Le.innerHTML=jt,Lt=n(),xe=i("p"),xe.innerHTML=Bt,xt=n(),w=i("div"),m(ie.$$.fragment),Tt=n(),Te=i("p"),Te.innerHTML=Zt,wt=n(),we=i("p"),we.textContent=Gt,Ye=n(),m(le.$$.fragment),Qe=n(),Xe=i("p"),this.h()},l(e){const t=no("svelte-u9bgzb",document.head);k=l(t,"META",{name:!0,content:!0}),t.forEach(o),C=r(e),x=l(e,"P",{}),L(x).forEach(o),de=r(e),D=l(e,"P",{"data-svelte-h":!0}),d(D)!=="svelte-fm1mno"&&(D.innerHTML=Mt),Fe=r(e),p(W.$$.fragment,e),Pe=r(e),$=l(e,"DIV",{class:!0,"data-svelte-h":!0}),d($)!=="svelte-13t8s2t"&&($.innerHTML=zt),Ne=r(e),p(E.$$.fragment,e),Ie=r(e),U=l(e,"P",{"data-svelte-h":!0}),d(U)!=="svelte-1b3a5ve"&&(U.innerHTML=$t),Ce=r(e),H=l(e,"P",{"data-svelte-h":!0}),d(H)!=="svelte-vfdo9a"&&(H.textContent=Xt),De=r(e),S=l(e,"P",{"data-svelte-h":!0}),d(S)!=="svelte-zdtbbl"&&(S.innerHTML=qt),We=r(e),O=l(e,"P",{"data-svelte-h":!0}),d(O)!=="svelte-d5pc84"&&(O.innerHTML=Ft),Ee=r(e),p(A.$$.fragment,e),Ue=r(e),R=l(e,"P",{"data-svelte-h":!0}),d(R)!=="svelte-u36c33"&&(R.textContent=Pt),He=r(e),p(V.$$.fragment,e),Se=r(e),j=l(e,"P",{"data-svelte-h":!0}),d(j)!=="svelte-1usw2g4"&&(j.innerHTML=Nt),Oe=r(e),p(B.$$.fragment,e),Ae=r(e),Z=l(e,"P",{"data-svelte-h":!0}),d(Z)!=="svelte-1cze4i5"&&(Z.innerHTML=It),Re=r(e),p(X.$$.fragment,e),Ve=r(e),p(G.$$.fragment,e),je=r(e),c=l(e,"DIV",{class:!0});var h=L(c);p(J.$$.fragment,h),rt=r(h),ce=l(h,"P",{"data-svelte-h":!0}),d(ce)!=="svelte-19vr0qz"&&(ce.innerHTML=Ct),st=r(h),me=l(h,"P",{"data-svelte-h":!0}),d(me)!=="svelte-ntrhio"&&(me.innerHTML=Dt),at=r(h),q=l(h,"DIV",{class:!0});var et=L(q);p(Y.$$.fragment,et),it=r(et),pe=l(et,"P",{"data-svelte-h":!0}),d(pe)!=="svelte-1w6bb17"&&(pe.textContent=Wt),et.forEach(o),lt=r(h),T=l(h,"DIV",{class:!0});var Me=L(T);p(Q.$$.fragment,Me),dt=r(Me),ue=l(Me,"P",{"data-svelte-h":!0}),d(ue)!=="svelte-1ooxl9e"&&(ue.textContent=Et),ct=r(Me),fe=l(Me,"UL",{"data-svelte-h":!0}),d(fe)!=="svelte-rq8uot"&&(fe.innerHTML=Ut),Me.forEach(o),mt=r(h),F=l(h,"DIV",{class:!0});var tt=L(F);p(K.$$.fragment,tt),pt=r(tt),_e=l(tt,"P",{"data-svelte-h":!0}),d(_e)!=="svelte-1f4f5kp"&&(_e.innerHTML=Ht),tt.forEach(o),ut=r(h),P=l(h,"DIV",{class:!0});var ot=L(P);p(ee.$$.fragment,ot),ft=r(ot),ge=l(ot,"P",{"data-svelte-h":!0}),d(ge)!=="svelte-bub0ru"&&(ge.textContent=St),ot.forEach(o),_t=r(h),he=l(h,"DIV",{class:!0});var Jt=L(he);p(te.$$.fragment,Jt),Jt.forEach(o),h.forEach(o),Be=r(e),p(oe.$$.fragment,e),Ze=r(e),y=l(e,"DIV",{class:!0});var I=L(y);p(ne.$$.fragment,I),gt=r(I),ke=l(I,"P",{"data-svelte-h":!0}),d(ke)!=="svelte-dnanf2"&&(ke.innerHTML=Ot),ht=r(I),be=l(I,"P",{"data-svelte-h":!0}),d(be)!=="svelte-gxzj9w"&&(be.innerHTML=At),kt=r(I),N=l(I,"DIV",{class:!0});var nt=L(N);p(re.$$.fragment,nt),bt=r(nt),ye=l(nt,"P",{"data-svelte-h":!0}),d(ye)!=="svelte-1w6bb17"&&(ye.textContent=Rt),nt.forEach(o),I.forEach(o),Ge=r(e),p(se.$$.fragment,e),Je=r(e),b=l(e,"DIV",{class:!0});var M=L(b);p(ae.$$.fragment,M),yt=r(M),ve=l(M,"P",{"data-svelte-h":!0}),d(ve)!=="svelte-1fy1bx7"&&(ve.textContent=Vt),vt=r(M),Le=l(M,"P",{"data-svelte-h":!0}),d(Le)!=="svelte-1npliwu"&&(Le.innerHTML=jt),Lt=r(M),xe=l(M,"P",{"data-svelte-h":!0}),d(xe)!=="svelte-3h9cxf"&&(xe.innerHTML=Bt),xt=r(M),w=l(M,"DIV",{class:!0});var ze=L(w);p(ie.$$.fragment,ze),Tt=r(ze),Te=l(ze,"P",{"data-svelte-h":!0}),d(Te)!=="svelte-1c21tx4"&&(Te.innerHTML=Zt),wt=r(ze),we=l(ze,"P",{"data-svelte-h":!0}),d(we)!=="svelte-ws0hzs"&&(we.textContent=Gt),ze.forEach(o),M.forEach(o),Ye=r(e),p(le.$$.fragment,e),Qe=r(e),Xe=l(e,"P",{}),L(Xe).forEach(o),this.h()},h(){v(k,"name","hf:doc:metadata"),v(k,"content",lo),v($,"class","flex flex-wrap space-x-1"),v(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(c,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(b,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){s(document.head,k),a(e,C,t),a(e,x,t),a(e,de,t),a(e,D,t),a(e,Fe,t),u(W,e,t),a(e,Pe,t),a(e,$,t),a(e,Ne,t),u(E,e,t),a(e,Ie,t),a(e,U,t),a(e,Ce,t),a(e,H,t),a(e,De,t),a(e,S,t),a(e,We,t),a(e,O,t),a(e,Ee,t),u(A,e,t),a(e,Ue,t),a(e,R,t),a(e,He,t),u(V,e,t),a(e,Se,t),a(e,j,t),a(e,Oe,t),u(B,e,t),a(e,Ae,t),a(e,Z,t),a(e,Re,t),u(X,e,t),a(e,Ve,t),u(G,e,t),a(e,je,t),a(e,c,t),u(J,c,null),s(c,rt),s(c,ce),s(c,st),s(c,me),s(c,at),s(c,q),u(Y,q,null),s(q,it),s(q,pe),s(c,lt),s(c,T),u(Q,T,null),s(T,dt),s(T,ue),s(T,ct),s(T,fe),s(c,mt),s(c,F),u(K,F,null),s(F,pt),s(F,_e),s(c,ut),s(c,P),u(ee,P,null),s(P,ft),s(P,ge),s(c,_t),s(c,he),u(te,he,null),a(e,Be,t),u(oe,e,t),a(e,Ze,t),a(e,y,t),u(ne,y,null),s(y,gt),s(y,ke),s(y,ht),s(y,be),s(y,kt),s(y,N),u(re,N,null),s(N,bt),s(N,ye),a(e,Ge,t),u(se,e,t),a(e,Je,t),a(e,b,t),u(ae,b,null),s(b,yt),s(b,ve),s(b,vt),s(b,Le),s(b,Lt),s(b,xe),s(b,xt),s(b,w),u(ie,w,null),s(w,Tt),s(w,Te),s(w,wt),s(w,we),a(e,Ye,t),u(le,e,t),a(e,Qe,t),a(e,Xe,t),Ke=!0},p(e,[t]){const h={};t&2&&(h.$$scope={dirty:t,ctx:e}),X.$set(h)},i(e){Ke||(f(W.$$.fragment,e),f(E.$$.fragment,e),f(A.$$.fragment,e),f(V.$$.fragment,e),f(B.$$.fragment,e),f(X.$$.fragment,e),f(G.$$.fragment,e),f(J.$$.fragment,e),f(Y.$$.fragment,e),f(Q.$$.fragment,e),f(K.$$.fragment,e),f(ee.$$.fragment,e),f(te.$$.fragment,e),f(oe.$$.fragment,e),f(ne.$$.fragment,e),f(re.$$.fragment,e),f(se.$$.fragment,e),f(ae.$$.fragment,e),f(ie.$$.fragment,e),f(le.$$.fragment,e),Ke=!0)},o(e){_(W.$$.fragment,e),_(E.$$.fragment,e),_(A.$$.fragment,e),_(V.$$.fragment,e),_(B.$$.fragment,e),_(X.$$.fragment,e),_(G.$$.fragment,e),_(J.$$.fragment,e),_(Y.$$.fragment,e),_(Q.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(re.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),Ke=!1},d(e){e&&(o(C),o(x),o(de),o(D),o(Fe),o(Pe),o($),o(Ne),o(Ie),o(U),o(Ce),o(H),o(De),o(S),o(We),o(O),o(Ee),o(Ue),o(R),o(He),o(Se),o(j),o(Oe),o(Ae),o(Z),o(Re),o(Ve),o(je),o(c),o(Be),o(Ze),o(y),o(Ge),o(Je),o(b),o(Ye),o(Qe),o(Xe)),o(k),g(W,e),g(E,e),g(A,e),g(V,e),g(B,e),g(X,e),g(G,e),g(J),g(Y),g(Q),g(K),g(ee),g(te),g(oe,e),g(ne),g(re),g(se,e),g(ae),g(ie),g(le,e)}}}const lo='{"title":"LayoutXLM","local":"layoutxlm","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips and examples","local":"usage-tips-and-examples","sections":[],"depth":2},{"title":"LayoutXLMTokenizer","local":"transformers.LayoutXLMTokenizer","sections":[],"depth":2},{"title":"LayoutXLMTokenizerFast","local":"transformers.LayoutXLMTokenizerFast","sections":[],"depth":2},{"title":"LayoutXLMProcessor","local":"transformers.LayoutXLMProcessor","sections":[],"depth":2}],"depth":1}';function co(qe){return Kt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ho extends to{constructor(k){super(),oo(this,k,co,io,Qt,{})}}export{ho as component};
