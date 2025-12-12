import{s as Ze,o as Je,n as Ke}from"../chunks/scheduler.18a86fab.js";import{S as Ye,i as et,g as s,s as r,r as u,A as tt,h as i,f as o,c as a,j,x as p,u as f,k as y,y as c,a as n,v as _,d as g,t as v,w as k}from"../chunks/index.98837b22.js";import{T as ot}from"../chunks/Tip.77304350.js";import{D as oe}from"../chunks/Docstring.a1ef7999.js";import{H as xe,E as nt}from"../chunks/getInferenceSnippets.06c2775f.js";function rt(ne){let l,z=`Wav2Vec2Phoneme’s architecture is based on the Wav2Vec2 model, for API reference, check out <a href="wav2vec2"><code>Wav2Vec2</code></a>’s documentation page
except for the tokenizer.`;return{c(){l=s("p"),l.innerHTML=z},l(h){l=i(h,"P",{"data-svelte-h":!0}),p(l)!=="svelte-odsari"&&(l.innerHTML=z)},m(h,B){n(h,l,B)},p:Ke,d(h){h&&o(l)}}}function at(ne){let l,z,h,B,P,Ie="<em>This model was released on 2021-09-23 and added to Hugging Face Transformers on 2021-12-17.</em>",re,$,ae,b,Ne='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',se,W,ie,V,Ee=`The Wav2Vec2Phoneme model was proposed in <a href="https://huggingface.co/papers/2109.11680" rel="nofollow">Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (Xu et al.,
2021)</a> by Qiantong Xu, Alexei Baevski, Michael Auli.`,ce,q,He="The abstract from the paper is the following:",de,L,Me=`<em>Recent progress in self-training, self-supervised pretraining and unsupervised learning enabled well performing speech
recognition systems without any labeled data. However, in many cases there is labeled data available for related
languages which is not utilized by these methods. This paper extends previous work on zero-shot cross-lingual transfer
learning by fine-tuning a multilingually pretrained wav2vec 2.0 model to transcribe unseen languages. This is done by
mapping phonemes of the training languages to the target language using articulatory features. Experiments show that
this simple method significantly outperforms prior work which introduced task-specific architectures and used only part
of a monolingually pretrained model.</em>`,le,F,De='Relevant checkpoints can be found under <a href="https://huggingface.co/models?other=phoneme-recognition" rel="nofollow">https://huggingface.co/models?other=phoneme-recognition</a>.',me,I,Oe='This model was contributed by <a href="https://huggingface.co/patrickvonplaten" rel="nofollow">patrickvonplaten</a>',pe,N,Ue='The original code can be found <a href="https://github.com/pytorch/fairseq/tree/master/fairseq/models/wav2vec" rel="nofollow">here</a>.',he,E,ue,H,Re=`<li>Wav2Vec2Phoneme uses the exact same architecture as Wav2Vec2</li> <li>Wav2Vec2Phoneme is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.</li> <li>Wav2Vec2Phoneme model was trained using connectionist temporal classification (CTC) so the model output has to be
decoded using <a href="/docs/transformers/v4.56.2/en/model_doc/wav2vec2_phoneme#transformers.Wav2Vec2PhonemeCTCTokenizer">Wav2Vec2PhonemeCTCTokenizer</a>.</li> <li>Wav2Vec2Phoneme can be fine-tuned on multiple language at once and decode unseen languages in a single forward pass
to a sequence of phonemes</li> <li>By default, the model outputs a sequence of phonemes. In order to transform the phonemes to a sequence of words one
should make use of a dictionary and language model.</li>`,fe,w,_e,M,ge,d,D,Ce,X,Ae="Constructs a Wav2Vec2PhonemeCTC tokenizer.",ye,G,Se=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.`,ze,x,O,Pe,Q,je=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`,$e,C,U,We,Z,Be="Convert a list of lists of token ids into a list of strings by calling decode.",Ve,T,R,qe,J,Xe=`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`,Le,K,Ge="Similar to doing <code>self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))</code>.",Fe,Y,A,ve,S,ke,te,Te;return $=new xe({props:{title:"Wav2Vec2Phoneme",local:"wav2vec2phoneme",headingTag:"h1"}}),W=new xe({props:{title:"Overview",local:"overview",headingTag:"h2"}}),E=new xe({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),w=new ot({props:{$$slots:{default:[rt]},$$scope:{ctx:ne}}}),M=new xe({props:{title:"Wav2Vec2PhonemeCTCTokenizer",local:"transformers.Wav2Vec2PhonemeCTCTokenizer",headingTag:"h2"}}),D=new oe({props:{name:"class transformers.Wav2Vec2PhonemeCTCTokenizer",anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"phone_delimiter_token",val:" = ' '"},{name:"word_delimiter_token",val:" = None"},{name:"do_phonemize",val:" = True"},{name:"phonemizer_lang",val:" = 'en-us'"},{name:"phonemizer_backend",val:" = 'espeak'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sentence token.`,name:"bos_token"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sentence token.`,name:"eos_token"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.do_phonemize",description:`<strong>do_phonemize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the tokenizer should phonetize the input or not. Only if a sequence of phonemes is passed to the
tokenizer, <code>do_phonemize</code> should be set to <code>False</code>.`,name:"do_phonemize"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.phonemizer_lang",description:`<strong>phonemizer_lang</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;en-us&quot;</code>) &#x2014;
The language of the phoneme set to which the tokenizer should phonetize the input text to.`,name:"phonemizer_lang"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.phonemizer_backend",description:`<strong>phonemizer_backend</strong> (<code>str</code>, <em>optional</em>. defaults to <code>&quot;espeak&quot;</code>) &#x2014;
The backend phonetization library that shall be used by the phonemizer library. Defaults to <code>espeak-ng</code>.
See the <a href="https://github.com/bootphon/phonemizer#readme" rel="nofollow">phonemizer package</a>. for more information.`,name:"phonemizer_backend"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.*kwargs",description:`*<strong>*kwargs</strong> &#x2014;
Additional keyword arguments passed along to <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L80"}}),O=new oe({props:{name:"__call__",anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
`}}),U=new oe({props:{name:"batch_decode",anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.batch_decode",parameters:[{name:"sequences",val:": typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"output_char_offsets",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.batch_decode.sequences",description:`<strong>sequences</strong> (<code>Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"sequences"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.batch_decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.batch_decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.batch_decode.output_char_offsets",description:`<strong>output_char_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to output character offsets. Character offsets can be used in combination with the
sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Please take a look at the Example of <code>~models.wav2vec2.tokenization_wav2vec2.decode</code> to better
understand how to make use of <code>output_word_offsets</code>.
<code>~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode</code> works analogous with phonemes
and batched output.</p>

					</div>`,name:"output_char_offsets"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.batch_decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L510",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The
decoded sentence. Will be a
<code>~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput</code> when
<code>output_char_offsets == True</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code> or <code>~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput</code></p>
`}}),R=new oe({props:{name:"decode",anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.decode",parameters:[{name:"token_ids",val:": typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"output_char_offsets",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.decode.token_ids",description:`<strong>token_ids</strong> (<code>Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"token_ids"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.decode.output_char_offsets",description:`<strong>output_char_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to output character offsets. Character offsets can be used in combination with the
sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>Please take a look at the Example of <code>~models.wav2vec2.tokenization_wav2vec2.decode</code> to better
understand how to make use of <code>output_word_offsets</code>.
<code>~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode</code> works the same way with
phonemes.</p>

					</div>`,name:"output_char_offsets"},{anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L454",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded
sentence. Will be a <code>~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput</code>
when <code>output_char_offsets == True</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code> or <code>~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput</code></p>
`}}),A=new oe({props:{name:"phonemize",anchor:"transformers.Wav2Vec2PhonemeCTCTokenizer.phonemize",parameters:[{name:"text",val:": str"},{name:"phonemizer_lang",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py#L252"}}),S=new nt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/wav2vec2_phoneme.md"}}),{c(){l=s("meta"),z=r(),h=s("p"),B=r(),P=s("p"),P.innerHTML=Ie,re=r(),u($.$$.fragment),ae=r(),b=s("div"),b.innerHTML=Ne,se=r(),u(W.$$.fragment),ie=r(),V=s("p"),V.innerHTML=Ee,ce=r(),q=s("p"),q.textContent=He,de=r(),L=s("p"),L.innerHTML=Me,le=r(),F=s("p"),F.innerHTML=De,me=r(),I=s("p"),I.innerHTML=Oe,pe=r(),N=s("p"),N.innerHTML=Ue,he=r(),u(E.$$.fragment),ue=r(),H=s("ul"),H.innerHTML=Re,fe=r(),u(w.$$.fragment),_e=r(),u(M.$$.fragment),ge=r(),d=s("div"),u(D.$$.fragment),Ce=r(),X=s("p"),X.textContent=Ae,ye=r(),G=s("p"),G.innerHTML=Se,ze=r(),x=s("div"),u(O.$$.fragment),Pe=r(),Q=s("p"),Q.textContent=je,$e=r(),C=s("div"),u(U.$$.fragment),We=r(),Z=s("p"),Z.textContent=Be,Ve=r(),T=s("div"),u(R.$$.fragment),qe=r(),J=s("p"),J.textContent=Xe,Le=r(),K=s("p"),K.innerHTML=Ge,Fe=r(),Y=s("div"),u(A.$$.fragment),ve=r(),u(S.$$.fragment),ke=r(),te=s("p"),this.h()},l(e){const t=tt("svelte-u9bgzb",document.head);l=i(t,"META",{name:!0,content:!0}),t.forEach(o),z=a(e),h=i(e,"P",{}),j(h).forEach(o),B=a(e),P=i(e,"P",{"data-svelte-h":!0}),p(P)!=="svelte-fv92zt"&&(P.innerHTML=Ie),re=a(e),f($.$$.fragment,e),ae=a(e),b=i(e,"DIV",{class:!0,"data-svelte-h":!0}),p(b)!=="svelte-13t8s2t"&&(b.innerHTML=Ne),se=a(e),f(W.$$.fragment,e),ie=a(e),V=i(e,"P",{"data-svelte-h":!0}),p(V)!=="svelte-10rppi5"&&(V.innerHTML=Ee),ce=a(e),q=i(e,"P",{"data-svelte-h":!0}),p(q)!=="svelte-vfdo9a"&&(q.textContent=He),de=a(e),L=i(e,"P",{"data-svelte-h":!0}),p(L)!=="svelte-148nuc2"&&(L.innerHTML=Me),le=a(e),F=i(e,"P",{"data-svelte-h":!0}),p(F)!=="svelte-5tdu8e"&&(F.innerHTML=De),me=a(e),I=i(e,"P",{"data-svelte-h":!0}),p(I)!=="svelte-13jbx2b"&&(I.innerHTML=Oe),pe=a(e),N=i(e,"P",{"data-svelte-h":!0}),p(N)!=="svelte-12gzw10"&&(N.innerHTML=Ue),he=a(e),f(E.$$.fragment,e),ue=a(e),H=i(e,"UL",{"data-svelte-h":!0}),p(H)!=="svelte-8bz8ui"&&(H.innerHTML=Re),fe=a(e),f(w.$$.fragment,e),_e=a(e),f(M.$$.fragment,e),ge=a(e),d=i(e,"DIV",{class:!0});var m=j(d);f(D.$$.fragment,m),Ce=a(m),X=i(m,"P",{"data-svelte-h":!0}),p(X)!=="svelte-1g1me1i"&&(X.textContent=Ae),ye=a(m),G=i(m,"P",{"data-svelte-h":!0}),p(G)!=="svelte-y4ylxw"&&(G.innerHTML=Se),ze=a(m),x=i(m,"DIV",{class:!0});var be=j(x);f(O.$$.fragment,be),Pe=a(be),Q=i(be,"P",{"data-svelte-h":!0}),p(Q)!=="svelte-kpxj0c"&&(Q.textContent=je),be.forEach(o),$e=a(m),C=i(m,"DIV",{class:!0});var we=j(C);f(U.$$.fragment,we),We=a(we),Z=i(we,"P",{"data-svelte-h":!0}),p(Z)!=="svelte-1deng2j"&&(Z.textContent=Be),we.forEach(o),Ve=a(m),T=i(m,"DIV",{class:!0});var ee=j(T);f(R.$$.fragment,ee),qe=a(ee),J=i(ee,"P",{"data-svelte-h":!0}),p(J)!=="svelte-vbfkpu"&&(J.textContent=Xe),Le=a(ee),K=i(ee,"P",{"data-svelte-h":!0}),p(K)!=="svelte-125uxon"&&(K.innerHTML=Ge),ee.forEach(o),Fe=a(m),Y=i(m,"DIV",{class:!0});var Qe=j(Y);f(A.$$.fragment,Qe),Qe.forEach(o),m.forEach(o),ve=a(e),f(S.$$.fragment,e),ke=a(e),te=i(e,"P",{}),j(te).forEach(o),this.h()},h(){y(l,"name","hf:doc:metadata"),y(l,"content",st),y(b,"class","flex flex-wrap space-x-1"),y(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(d,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){c(document.head,l),n(e,z,t),n(e,h,t),n(e,B,t),n(e,P,t),n(e,re,t),_($,e,t),n(e,ae,t),n(e,b,t),n(e,se,t),_(W,e,t),n(e,ie,t),n(e,V,t),n(e,ce,t),n(e,q,t),n(e,de,t),n(e,L,t),n(e,le,t),n(e,F,t),n(e,me,t),n(e,I,t),n(e,pe,t),n(e,N,t),n(e,he,t),_(E,e,t),n(e,ue,t),n(e,H,t),n(e,fe,t),_(w,e,t),n(e,_e,t),_(M,e,t),n(e,ge,t),n(e,d,t),_(D,d,null),c(d,Ce),c(d,X),c(d,ye),c(d,G),c(d,ze),c(d,x),_(O,x,null),c(x,Pe),c(x,Q),c(d,$e),c(d,C),_(U,C,null),c(C,We),c(C,Z),c(d,Ve),c(d,T),_(R,T,null),c(T,qe),c(T,J),c(T,Le),c(T,K),c(d,Fe),c(d,Y),_(A,Y,null),n(e,ve,t),_(S,e,t),n(e,ke,t),n(e,te,t),Te=!0},p(e,[t]){const m={};t&2&&(m.$$scope={dirty:t,ctx:e}),w.$set(m)},i(e){Te||(g($.$$.fragment,e),g(W.$$.fragment,e),g(E.$$.fragment,e),g(w.$$.fragment,e),g(M.$$.fragment,e),g(D.$$.fragment,e),g(O.$$.fragment,e),g(U.$$.fragment,e),g(R.$$.fragment,e),g(A.$$.fragment,e),g(S.$$.fragment,e),Te=!0)},o(e){v($.$$.fragment,e),v(W.$$.fragment,e),v(E.$$.fragment,e),v(w.$$.fragment,e),v(M.$$.fragment,e),v(D.$$.fragment,e),v(O.$$.fragment,e),v(U.$$.fragment,e),v(R.$$.fragment,e),v(A.$$.fragment,e),v(S.$$.fragment,e),Te=!1},d(e){e&&(o(z),o(h),o(B),o(P),o(re),o(ae),o(b),o(se),o(ie),o(V),o(ce),o(q),o(de),o(L),o(le),o(F),o(me),o(I),o(pe),o(N),o(he),o(ue),o(H),o(fe),o(_e),o(ge),o(d),o(ve),o(ke),o(te)),o(l),k($,e),k(W,e),k(E,e),k(w,e),k(M,e),k(D),k(O),k(U),k(R),k(A),k(S,e)}}}const st='{"title":"Wav2Vec2Phoneme","local":"wav2vec2phoneme","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Wav2Vec2PhonemeCTCTokenizer","local":"transformers.Wav2Vec2PhonemeCTCTokenizer","sections":[],"depth":2}],"depth":1}';function it(ne){return Je(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ht extends Ye{constructor(l){super(),et(this,l,it,at,Ze,{})}}export{ht as component};
