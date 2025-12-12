import{s as He,o as Ae,n as De}from"../chunks/scheduler.18a86fab.js";import{S as Ve,i as Re,g as r,s,r as u,A as Fe,h as a,f as n,c as i,j,x as p,u as h,k as B,y as f,a as o,v as g,d as _,t as k,w as y}from"../chunks/index.98837b22.js";import{T as Ke}from"../chunks/Tip.77304350.js";import{D as we}from"../chunks/Docstring.a1ef7999.js";import{C as Se}from"../chunks/CodeBlock.8d0c2e8a.js";import{H as _e,E as Je}from"../chunks/getInferenceSnippets.06c2775f.js";function Oe(G){let l,w=`As mLUKE’s architecture is equivalent to that of LUKE, one can refer to <a href="luke">LUKE’s documentation page</a> for all
tips, code examples and notebooks.`;return{c(){l=r("p"),l.innerHTML=w},l(m){l=a(m,"P",{"data-svelte-h":!0}),p(l)!=="svelte-1h0p68f"&&(l.innerHTML=w)},m(m,R){o(m,l,R)},p:De,d(m){m&&n(l)}}}function Be(G){let l,w,m,R,x,xe="<em>This model was released on 2021-10-15 and added to Hugging Face Transformers on 2021-12-07.</em>",X,q,Y,b,qe='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Q,L,ee,z,Le=`The mLUKE model was proposed in <a href="https://huggingface.co/papers/2110.08151" rel="nofollow">mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models</a> by Ryokan Ri, Ikuya Yamada, and Yoshimasa Tsuruoka. It’s a multilingual extension
of the <a href="https://huggingface.co/papers/2010.01057" rel="nofollow">LUKE model</a> trained on the basis of XLM-RoBERTa.`,te,M,ze=`It is based on XLM-RoBERTa and adds entity embeddings, which helps improve performance on various downstream tasks
involving reasoning about entities such as named entity recognition, extractive question answering, relation
classification, cloze-style knowledge completion.`,ne,$,Me="The abstract from the paper is the following:",oe,P,$e=`<em>Recent studies have shown that multilingual pretrained language models can be effectively improved with cross-lingual
alignment information from Wikipedia entities. However, existing methods only exploit entity information in pretraining
and do not explicitly use entities in downstream tasks. In this study, we explore the effectiveness of leveraging
entity representations for downstream cross-lingual tasks. We train a multilingual language model with 24 languages
with entity representations and show the model consistently outperforms word-based pretrained models in various
cross-lingual transfer tasks. We also analyze the model and the key insight is that incorporating entity
representations into the input allows us to extract more language-agnostic features. We also evaluate the model with a
multilingual cloze prompt task with the mLAMA dataset. We show that entity-based prompt elicits correct factual
knowledge more likely than using only word representations.</em>`,se,E,Pe='This model was contributed by <a href="https://huggingface.co/ryo0634" rel="nofollow">ryo0634</a>. The original code can be found <a href="https://github.com/studio-ousia/luke" rel="nofollow">here</a>.',ie,I,re,N,Ee="One can directly plug in the weights of mLUKE into a LUKE model, like so:",ae,U,le,W,Ie='Note that mLUKE has its own tokenizer, <a href="/docs/transformers/v4.56.2/en/model_doc/mluke#transformers.MLukeTokenizer">MLukeTokenizer</a>. You can initialize it as follows:',de,C,ce,v,pe,S,me,d,H,ke,F,Ne=`Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer">XLMRobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeTokenizer">LukeTokenizer</a>. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.`,ye,K,Ue=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,be,T,A,ve,J,We=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences, depending on the task you want to prepare them for.`,Te,O,D,ue,V,he,Z,fe;return q=new _e({props:{title:"mLUKE",local:"mluke",headingTag:"h1"}}),L=new _e({props:{title:"Overview",local:"overview",headingTag:"h2"}}),I=new _e({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),U=new Se({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEx1a2VNb2RlbCUwQSUwQW1vZGVsJTIwJTNEJTIwTHVrZU1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJzdHVkaW8tb3VzaWElMkZtbHVrZS1iYXNlJTIyKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LukeModel

model = LukeModel.from_pretrained(<span class="hljs-string">&quot;studio-ousia/mluke-base&quot;</span>)`,wrap:!1}}),C=new Se({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1MdWtlVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwTUx1a2VUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRm1sdWtlLWJhc2UlMjIp",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MLukeTokenizer

tokenizer = MLukeTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/mluke-base&quot;</span>)`,wrap:!1}}),v=new Ke({props:{$$slots:{default:[Oe]},$$scope:{ctx:G}}}),S=new _e({props:{title:"MLukeTokenizer",local:"transformers.MLukeTokenizer",headingTag:"h2"}}),H=new we({props:{name:"class transformers.MLukeTokenizer",anchor:"transformers.MLukeTokenizer",parameters:[{name:"vocab_file",val:""},{name:"entity_vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"task",val:" = None"},{name:"max_entity_length",val:" = 32"},{name:"max_mention_length",val:" = 30"},{name:"entity_token_1",val:" = '<ent>'"},{name:"entity_token_2",val:" = '<ent2>'"},{name:"entity_unk_token",val:" = '[UNK]'"},{name:"entity_pad_token",val:" = '[PAD]'"},{name:"entity_mask_token",val:" = '[MASK]'"},{name:"entity_mask2_token",val:" = '[MASK2]'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MLukeTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.MLukeTokenizer.entity_vocab_file",description:`<strong>entity_vocab_file</strong> (<code>str</code>) &#x2014;
Path to the entity vocabulary file.`,name:"entity_vocab_file"},{anchor:"transformers.MLukeTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.MLukeTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.MLukeTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MLukeTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MLukeTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MLukeTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MLukeTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MLukeTokenizer.task",description:`<strong>task</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Task for which you want to prepare sequences. One of <code>&quot;entity_classification&quot;</code>,
<code>&quot;entity_pair_classification&quot;</code>, or <code>&quot;entity_span_classification&quot;</code>. If you specify this argument, the entity
sequence is automatically created based on the given entity span(s).`,name:"task"},{anchor:"transformers.MLukeTokenizer.max_entity_length",description:`<strong>max_entity_length</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The maximum length of <code>entity_ids</code>.`,name:"max_entity_length"},{anchor:"transformers.MLukeTokenizer.max_mention_length",description:`<strong>max_mention_length</strong> (<code>int</code>, <em>optional</em>, defaults to 30) &#x2014;
The maximum number of tokens inside an entity span.`,name:"max_mention_length"},{anchor:"transformers.MLukeTokenizer.entity_token_1",description:`<strong>entity_token_1</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;ent&gt;</code>) &#x2014;
The special token used to represent an entity span in a word token sequence. This token is only used when
<code>task</code> is set to <code>&quot;entity_classification&quot;</code> or <code>&quot;entity_pair_classification&quot;</code>.`,name:"entity_token_1"},{anchor:"transformers.MLukeTokenizer.entity_token_2",description:`<strong>entity_token_2</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;ent2&gt;</code>) &#x2014;
The special token used to represent an entity span in a word token sequence. This token is only used when
<code>task</code> is set to <code>&quot;entity_pair_classification&quot;</code>.`,name:"entity_token_2"},{anchor:"transformers.MLukeTokenizer.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&quot;&lt;s&gt;NOTUSED&quot;, &quot;&lt;/s&gt;NOTUSED&quot;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"},{anchor:"transformers.MLukeTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.MLukeTokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mluke/tokenization_mluke.py#L133"}}),A=new we({props:{name:"__call__",anchor:"transformers.MLukeTokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str]]"},{name:"text_pair",val:": typing.Union[str, list[str], NoneType] = None"},{name:"entity_spans",val:": typing.Union[list[tuple[int, int]], list[list[tuple[int, int]]], NoneType] = None"},{name:"entity_spans_pair",val:": typing.Union[list[tuple[int, int]], list[list[tuple[int, int]]], NoneType] = None"},{name:"entities",val:": typing.Union[list[str], list[list[str]], NoneType] = None"},{name:"entities_pair",val:": typing.Union[list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"max_entity_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": typing.Optional[bool] = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MLukeTokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
tokenizer does not support tokenization based on pretokenized strings.`,name:"text"},{anchor:"transformers.MLukeTokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
tokenizer does not support tokenization based on pretokenized strings.`,name:"text_pair"},{anchor:"transformers.MLukeTokenizer.__call__.entity_spans",description:`<strong>entity_spans</strong> (<code>list[tuple[int, int]]</code>, <code>list[list[tuple[int, int]]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
with two integers denoting character-based start and end positions of entities. If you specify
<code>&quot;entity_classification&quot;</code> or <code>&quot;entity_pair_classification&quot;</code> as the <code>task</code> argument in the constructor,
the length of each sequence must be 1 or 2, respectively. If you specify <code>entities</code>, the length of each
sequence must be equal to the length of each sequence of <code>entities</code>.`,name:"entity_spans"},{anchor:"transformers.MLukeTokenizer.__call__.entity_spans_pair",description:`<strong>entity_spans_pair</strong> (<code>list[tuple[int, int]]</code>, <code>list[list[tuple[int, int]]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
with two integers denoting character-based start and end positions of entities. If you specify the
<code>task</code> argument in the constructor, this argument is ignored. If you specify <code>entities_pair</code>, the
length of each sequence must be equal to the length of each sequence of <code>entities_pair</code>.`,name:"entity_spans_pair"},{anchor:"transformers.MLukeTokenizer.__call__.entities",description:`<strong>entities</strong> (<code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
Angeles). This argument is ignored if you specify the <code>task</code> argument in the constructor. The length of
each sequence must be equal to the length of each sequence of <code>entity_spans</code>. If you specify
<code>entity_spans</code> without specifying this argument, the entity sequence or the batch of entity sequences
is automatically constructed by filling it with the [MASK] entity.`,name:"entities"},{anchor:"transformers.MLukeTokenizer.__call__.entities_pair",description:`<strong>entities_pair</strong> (<code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
Angeles). This argument is ignored if you specify the <code>task</code> argument in the constructor. The length of
each sequence must be equal to the length of each sequence of <code>entity_spans_pair</code>. If you specify
<code>entity_spans_pair</code> without specifying this argument, the entity sequence or the batch of entity
sequences is automatically constructed by filling it with the [MASK] entity.`,name:"entities_pair"},{anchor:"transformers.MLukeTokenizer.__call__.max_entity_length",description:`<strong>max_entity_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The maximum length of <code>entity_ids</code>.`,name:"max_entity_length"},{anchor:"transformers.MLukeTokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.MLukeTokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MLukeTokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.MLukeTokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MLukeTokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.MLukeTokenizer.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.MLukeTokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MLukeTokenizer.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MLukeTokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MLukeTokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.MLukeTokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.MLukeTokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.MLukeTokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.MLukeTokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.MLukeTokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.MLukeTokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.MLukeTokenizer.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mluke/tokenization_mluke.py#L386",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
<p><strong>entity_ids</strong> — List of entity ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>entity_position_ids</strong> — List of entity positions in the input sequence to be fed to a model.</p>
</li>
<li>
<p><strong>entity_token_type_ids</strong> — List of entity token type ids to be fed to a model (when
<code>return_token_type_ids=True</code> or if <em>“entity_token_type_ids”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a></p>
</li>
<li>
<p><strong>entity_attention_mask</strong> — List of indices specifying which entities should be attended to by the model
(when <code>return_attention_mask=True</code> or if <em>“entity_attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>entity_start_positions</strong> — List of the start positions of entities in the word token sequence (when
<code>task="entity_span_classification"</code>).</p>
</li>
<li>
<p><strong>entity_end_positions</strong> — List of the end positions of entities in the word token sequence (when
<code>task="entity_span_classification"</code>).</p>
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
`}}),D=new we({props:{name:"save_vocabulary",anchor:"transformers.MLukeTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mluke/tokenization_mluke.py#L1533"}}),V=new Je({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mluke.md"}}),{c(){l=r("meta"),w=s(),m=r("p"),R=s(),x=r("p"),x.innerHTML=xe,X=s(),u(q.$$.fragment),Y=s(),b=r("div"),b.innerHTML=qe,Q=s(),u(L.$$.fragment),ee=s(),z=r("p"),z.innerHTML=Le,te=s(),M=r("p"),M.textContent=ze,ne=s(),$=r("p"),$.textContent=Me,oe=s(),P=r("p"),P.innerHTML=$e,se=s(),E=r("p"),E.innerHTML=Pe,ie=s(),u(I.$$.fragment),re=s(),N=r("p"),N.textContent=Ee,ae=s(),u(U.$$.fragment),le=s(),W=r("p"),W.innerHTML=Ie,de=s(),u(C.$$.fragment),ce=s(),u(v.$$.fragment),pe=s(),u(S.$$.fragment),me=s(),d=r("div"),u(H.$$.fragment),ke=s(),F=r("p"),F.innerHTML=Ne,ye=s(),K=r("p"),K.innerHTML=Ue,be=s(),T=r("div"),u(A.$$.fragment),ve=s(),J=r("p"),J.textContent=We,Te=s(),O=r("div"),u(D.$$.fragment),ue=s(),u(V.$$.fragment),he=s(),Z=r("p"),this.h()},l(e){const t=Fe("svelte-u9bgzb",document.head);l=a(t,"META",{name:!0,content:!0}),t.forEach(n),w=i(e),m=a(e,"P",{}),j(m).forEach(n),R=i(e),x=a(e,"P",{"data-svelte-h":!0}),p(x)!=="svelte-bibzv9"&&(x.innerHTML=xe),X=i(e),h(q.$$.fragment,e),Y=i(e),b=a(e,"DIV",{class:!0,"data-svelte-h":!0}),p(b)!=="svelte-13t8s2t"&&(b.innerHTML=qe),Q=i(e),h(L.$$.fragment,e),ee=i(e),z=a(e,"P",{"data-svelte-h":!0}),p(z)!=="svelte-ymom85"&&(z.innerHTML=Le),te=i(e),M=a(e,"P",{"data-svelte-h":!0}),p(M)!=="svelte-luogjx"&&(M.textContent=ze),ne=i(e),$=a(e,"P",{"data-svelte-h":!0}),p($)!=="svelte-vfdo9a"&&($.textContent=Me),oe=i(e),P=a(e,"P",{"data-svelte-h":!0}),p(P)!=="svelte-2cofqb"&&(P.innerHTML=$e),se=i(e),E=a(e,"P",{"data-svelte-h":!0}),p(E)!=="svelte-197cpmu"&&(E.innerHTML=Pe),ie=i(e),h(I.$$.fragment,e),re=i(e),N=a(e,"P",{"data-svelte-h":!0}),p(N)!=="svelte-3ftlrg"&&(N.textContent=Ee),ae=i(e),h(U.$$.fragment,e),le=i(e),W=a(e,"P",{"data-svelte-h":!0}),p(W)!=="svelte-1rnjlhx"&&(W.innerHTML=Ie),de=i(e),h(C.$$.fragment,e),ce=i(e),h(v.$$.fragment,e),pe=i(e),h(S.$$.fragment,e),me=i(e),d=a(e,"DIV",{class:!0});var c=j(d);h(H.$$.fragment,c),ke=i(c),F=a(c,"P",{"data-svelte-h":!0}),p(F)!=="svelte-e1n1g7"&&(F.innerHTML=Ne),ye=i(c),K=a(c,"P",{"data-svelte-h":!0}),p(K)!=="svelte-ntrhio"&&(K.innerHTML=Ue),be=i(c),T=a(c,"DIV",{class:!0});var ge=j(T);h(A.$$.fragment,ge),ve=i(ge),J=a(ge,"P",{"data-svelte-h":!0}),p(J)!=="svelte-16lcbtv"&&(J.textContent=We),ge.forEach(n),Te=i(c),O=a(c,"DIV",{class:!0});var Ce=j(O);h(D.$$.fragment,Ce),Ce.forEach(n),c.forEach(n),ue=i(e),h(V.$$.fragment,e),he=i(e),Z=a(e,"P",{}),j(Z).forEach(n),this.h()},h(){B(l,"name","hf:doc:metadata"),B(l,"content",Ze),B(b,"class","flex flex-wrap space-x-1"),B(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),B(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),B(d,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){f(document.head,l),o(e,w,t),o(e,m,t),o(e,R,t),o(e,x,t),o(e,X,t),g(q,e,t),o(e,Y,t),o(e,b,t),o(e,Q,t),g(L,e,t),o(e,ee,t),o(e,z,t),o(e,te,t),o(e,M,t),o(e,ne,t),o(e,$,t),o(e,oe,t),o(e,P,t),o(e,se,t),o(e,E,t),o(e,ie,t),g(I,e,t),o(e,re,t),o(e,N,t),o(e,ae,t),g(U,e,t),o(e,le,t),o(e,W,t),o(e,de,t),g(C,e,t),o(e,ce,t),g(v,e,t),o(e,pe,t),g(S,e,t),o(e,me,t),o(e,d,t),g(H,d,null),f(d,ke),f(d,F),f(d,ye),f(d,K),f(d,be),f(d,T),g(A,T,null),f(T,ve),f(T,J),f(d,Te),f(d,O),g(D,O,null),o(e,ue,t),g(V,e,t),o(e,he,t),o(e,Z,t),fe=!0},p(e,[t]){const c={};t&2&&(c.$$scope={dirty:t,ctx:e}),v.$set(c)},i(e){fe||(_(q.$$.fragment,e),_(L.$$.fragment,e),_(I.$$.fragment,e),_(U.$$.fragment,e),_(C.$$.fragment,e),_(v.$$.fragment,e),_(S.$$.fragment,e),_(H.$$.fragment,e),_(A.$$.fragment,e),_(D.$$.fragment,e),_(V.$$.fragment,e),fe=!0)},o(e){k(q.$$.fragment,e),k(L.$$.fragment,e),k(I.$$.fragment,e),k(U.$$.fragment,e),k(C.$$.fragment,e),k(v.$$.fragment,e),k(S.$$.fragment,e),k(H.$$.fragment,e),k(A.$$.fragment,e),k(D.$$.fragment,e),k(V.$$.fragment,e),fe=!1},d(e){e&&(n(w),n(m),n(R),n(x),n(X),n(Y),n(b),n(Q),n(ee),n(z),n(te),n(M),n(ne),n($),n(oe),n(P),n(se),n(E),n(ie),n(re),n(N),n(ae),n(le),n(W),n(de),n(ce),n(pe),n(me),n(d),n(ue),n(he),n(Z)),n(l),y(q,e),y(L,e),y(I,e),y(U,e),y(C,e),y(v,e),y(S,e),y(H),y(A),y(D),y(V,e)}}}const Ze='{"title":"mLUKE","local":"mluke","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"MLukeTokenizer","local":"transformers.MLukeTokenizer","sections":[],"depth":2}],"depth":1}';function je(G){return Ae(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class nt extends Ve{constructor(l){super(),Re(this,l,je,Be,He,{})}}export{nt as component};
