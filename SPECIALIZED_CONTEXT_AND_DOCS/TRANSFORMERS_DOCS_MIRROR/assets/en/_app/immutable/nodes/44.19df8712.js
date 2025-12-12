import{s as As,o as Hs,n as ze}from"../chunks/scheduler.18a86fab.js";import{S as Gs,i as Xs,g as r,s as n,r as m,A as Ys,h as s,f as a,c as o,j as v,u,x as c,k as y,y as e,a as w,v as h,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as Kn}from"../chunks/Tip.77304350.js";import{D as x}from"../chunks/Docstring.a1ef7999.js";import{C as no}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as eo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as to,E as Os}from"../chunks/getInferenceSnippets.06c2775f.js";function Qs(M){let i,z="This method is deprecated, <code>__call__</code> should be used instead.";return{c(){i=r("p"),i.innerHTML=z},l(k){i=s(k,"P",{"data-svelte-h":!0}),c(i)!=="svelte-1phrc72"&&(i.innerHTML=z)},m(k,T){w(k,i,T)},p:ze,d(k){k&&a(i)}}}function Ks(M){let i,z="This method is deprecated, <code>__call__</code> should be used instead.";return{c(){i=r("p"),i.innerHTML=z},l(k){i=s(k,"P",{"data-svelte-h":!0}),c(i)!=="svelte-1phrc72"&&(i.innerHTML=z)},m(k,T){w(k,i,T)},p:ze,d(k){k&&a(i)}}}function ea(M){let i,z="Passing <code>token=True</code> is required when you want to use a private model.";return{c(){i=r("p"),i.innerHTML=z},l(k){i=s(k,"P",{"data-svelte-h":!0}),c(i)!=="svelte-15auxyb"&&(i.innerHTML=z)},m(k,T){w(k,i,T)},p:ze,d(k){k&&a(i)}}}function ta(M){let i,z="Examples:",k,T,$;return T=new no({props:{code:"JTIzJTIwV2UlMjBjYW4ndCUyMGluc3RhbnRpYXRlJTIwZGlyZWN0bHklMjB0aGUlMjBiYXNlJTIwY2xhc3MlMjAqUHJlVHJhaW5lZFRva2VuaXplckJhc2UqJTIwc28lMjBsZXQncyUyMHNob3clMjBvdXIlMjBleGFtcGxlcyUyMG9uJTIwYSUyMGRlcml2ZWQlMjBjbGFzcyUzQSUyMEJlcnRUb2tlbml6ZXIlMEElMjMlMjBEb3dubG9hZCUyMHZvY2FidWxhcnklMjBmcm9tJTIwaHVnZ2luZ2ZhY2UuY28lMjBhbmQlMjBjYWNoZS4lMEF0b2tlbml6ZXIlMjAlM0QlMjBCZXJ0VG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQSUwQSUyMyUyMERvd25sb2FkJTIwdm9jYWJ1bGFyeSUyMGZyb20lMjBodWdnaW5nZmFjZS5jbyUyMCh1c2VyLXVwbG9hZGVkKSUyMGFuZCUyMGNhY2hlLiUwQXRva2VuaXplciUyMCUzRCUyMEJlcnRUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmRibWR6JTJGYmVydC1iYXNlLWdlcm1hbi1jYXNlZCUyMiklMEElMEElMjMlMjBJZiUyMHZvY2FidWxhcnklMjBmaWxlcyUyMGFyZSUyMGluJTIwYSUyMGRpcmVjdG9yeSUyMChlLmcuJTIwdG9rZW5pemVyJTIwd2FzJTIwc2F2ZWQlMjB1c2luZyUyMCpzYXZlX3ByZXRyYWluZWQoJy4lMkZ0ZXN0JTJGc2F2ZWRfbW9kZWwlMkYnKSopJTBBdG9rZW5pemVyJTIwJTNEJTIwQmVydFRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyLiUyRnRlc3QlMkZzYXZlZF9tb2RlbCUyRiUyMiklMEElMEElMjMlMjBJZiUyMHRoZSUyMHRva2VuaXplciUyMHVzZXMlMjBhJTIwc2luZ2xlJTIwdm9jYWJ1bGFyeSUyMGZpbGUlMkMlMjB5b3UlMjBjYW4lMjBwb2ludCUyMGRpcmVjdGx5JTIwdG8lMjB0aGlzJTIwZmlsZSUwQXRva2VuaXplciUyMCUzRCUyMEJlcnRUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMi4lMkZ0ZXN0JTJGc2F2ZWRfbW9kZWwlMkZteV92b2NhYi50eHQlMjIpJTBBJTBBJTIzJTIwWW91JTIwY2FuJTIwbGluayUyMHRva2VucyUyMHRvJTIwc3BlY2lhbCUyMHZvY2FidWxhcnklMjB3aGVuJTIwaW5zdGFudGlhdGluZyUwQXRva2VuaXplciUyMCUzRCUyMEJlcnRUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIlMkMlMjB1bmtfdG9rZW4lM0QlMjIlM0N1bmslM0UlMjIpJTBBJTIzJTIwWW91JTIwc2hvdWxkJTIwYmUlMjBzdXJlJTIwJyUzQ3VuayUzRSclMjBpcyUyMGluJTIwdGhlJTIwdm9jYWJ1bGFyeSUyMHdoZW4lMjBkb2luZyUyMHRoYXQuJTBBJTIzJTIwT3RoZXJ3aXNlJTIwdXNlJTIwdG9rZW5pemVyLmFkZF9zcGVjaWFsX3Rva2VucyglN0IndW5rX3Rva2VuJyUzQSUyMCclM0N1bmslM0UnJTdEKSUyMGluc3RlYWQpJTBBYXNzZXJ0JTIwdG9rZW5pemVyLnVua190b2tlbiUyMCUzRCUzRCUyMCUyMiUzQ3VuayUzRSUyMg==",highlighted:`<span class="hljs-comment"># We can&#x27;t instantiate directly the base class *PreTrainedTokenizerBase* so let&#x27;s show our examples on a derived class: BertTokenizer</span>
<span class="hljs-comment"># Download vocabulary from huggingface.co and cache.</span>
tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-comment"># Download vocabulary from huggingface.co (user-uploaded) and cache.</span>
tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;dbmdz/bert-base-german-cased&quot;</span>)

<span class="hljs-comment"># If vocabulary files are in a directory (e.g. tokenizer was saved using *save_pretrained(&#x27;./test/saved_model/&#x27;)*)</span>
tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;./test/saved_model/&quot;</span>)

<span class="hljs-comment"># If the tokenizer uses a single vocabulary file, you can point directly to this file</span>
tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;./test/saved_model/my_vocab.txt&quot;</span>)

<span class="hljs-comment"># You can link tokens to special vocabulary when instantiating</span>
tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>, unk_token=<span class="hljs-string">&quot;&lt;unk&gt;&quot;</span>)
<span class="hljs-comment"># You should be sure &#x27;&lt;unk&gt;&#x27; is in the vocabulary when doing that.</span>
<span class="hljs-comment"># Otherwise use tokenizer.add_special_tokens({&#x27;unk_token&#x27;: &#x27;&lt;unk&gt;&#x27;}) instead)</span>
<span class="hljs-keyword">assert</span> tokenizer.unk_token == <span class="hljs-string">&quot;&lt;unk&gt;&quot;</span>`,wrap:!1}}),{c(){i=r("p"),i.textContent=z,k=n(),m(T.$$.fragment)},l(p){i=s(p,"P",{"data-svelte-h":!0}),c(i)!=="svelte-kvfsh7"&&(i.textContent=z),k=o(p),u(T.$$.fragment,p)},m(p,P){w(p,i,P),w(p,k,P),h(T,p,P),$=!0},p:ze,i(p){$||(f(T.$$.fragment,p),$=!0)},o(p){g(T.$$.fragment,p),$=!1},d(p){p&&(a(i),a(k)),_(T,p)}}}function na(M){let i,z=`If the <code>encoded_inputs</code> passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
result will use the same type unless you provide a different tensor type with <code>return_tensors</code>. In the case of
PyTorch tensors, you will lose the specific device of your tensors however.`;return{c(){i=r("p"),i.innerHTML=z},l(k){i=s(k,"P",{"data-svelte-h":!0}),c(i)!=="svelte-ppz3re"&&(i.innerHTML=z)},m(k,T){w(k,i,T)},p:ze,d(k){k&&a(i)}}}function oa(M){let i,z="Examples:",k,T,$;return T=new no({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS1jYXNlZCUyMiklMEElMEElMjMlMjBQdXNoJTIwdGhlJTIwdG9rZW5pemVyJTIwdG8lMjB5b3VyJTIwbmFtZXNwYWNlJTIwd2l0aCUyMHRoZSUyMG5hbWUlMjAlMjJteS1maW5ldHVuZWQtYmVydCUyMi4lMEF0b2tlbml6ZXIucHVzaF90b19odWIoJTIybXktZmluZXR1bmVkLWJlcnQlMjIpJTBBJTBBJTIzJTIwUHVzaCUyMHRoZSUyMHRva2VuaXplciUyMHRvJTIwYW4lMjBvcmdhbml6YXRpb24lMjB3aXRoJTIwdGhlJTIwbmFtZSUyMCUyMm15LWZpbmV0dW5lZC1iZXJ0JTIyLiUwQXRva2VuaXplci5wdXNoX3RvX2h1YiglMjJodWdnaW5nZmFjZSUyRm15LWZpbmV0dW5lZC1iZXJ0JTIyKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-cased&quot;</span>)

<span class="hljs-comment"># Push the tokenizer to your namespace with the name &quot;my-finetuned-bert&quot;.</span>
tokenizer.push_to_hub(<span class="hljs-string">&quot;my-finetuned-bert&quot;</span>)

<span class="hljs-comment"># Push the tokenizer to an organization with the name &quot;my-finetuned-bert&quot;.</span>
tokenizer.push_to_hub(<span class="hljs-string">&quot;huggingface/my-finetuned-bert&quot;</span>)`,wrap:!1}}),{c(){i=r("p"),i.textContent=z,k=n(),m(T.$$.fragment)},l(p){i=s(p,"P",{"data-svelte-h":!0}),c(i)!=="svelte-kvfsh7"&&(i.textContent=z),k=o(p),u(T.$$.fragment,p)},m(p,P){w(p,i,P),w(p,k,P),h(T,p,P),$=!0},p:ze,i(p){$||(f(T.$$.fragment,p),$=!0)},o(p){g(T.$$.fragment,p),$=!1},d(p){p&&(a(i),a(k)),_(T,p)}}}function ra(M){let i,z="Examples:",k,T,$;return T=new no({props:{code:"JTIzJTIwTGV0J3MlMjBzZWUlMjBob3clMjB0byUyMGFkZCUyMGElMjBuZXclMjBjbGFzc2lmaWNhdGlvbiUyMHRva2VuJTIwdG8lMjBHUFQtMiUwQXRva2VuaXplciUyMCUzRCUyMEdQVDJUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haS1jb21tdW5pdHklMkZncHQyJTIyKSUwQW1vZGVsJTIwJTNEJTIwR1BUMk1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMiklMEElMEFzcGVjaWFsX3Rva2Vuc19kaWN0JTIwJTNEJTIwJTdCJTIyY2xzX3Rva2VuJTIyJTNBJTIwJTIyJTNDQ0xTJTNFJTIyJTdEJTBBJTBBbnVtX2FkZGVkX3Rva3MlMjAlM0QlMjB0b2tlbml6ZXIuYWRkX3NwZWNpYWxfdG9rZW5zKHNwZWNpYWxfdG9rZW5zX2RpY3QpJTBBcHJpbnQoJTIyV2UlMjBoYXZlJTIwYWRkZWQlMjIlMkMlMjBudW1fYWRkZWRfdG9rcyUyQyUyMCUyMnRva2VucyUyMiklMEElMjMlMjBOb3RpY2UlM0ElMjByZXNpemVfdG9rZW5fZW1iZWRkaW5ncyUyMGV4cGVjdCUyMHRvJTIwcmVjZWl2ZSUyMHRoZSUyMGZ1bGwlMjBzaXplJTIwb2YlMjB0aGUlMjBuZXclMjB2b2NhYnVsYXJ5JTJDJTIwaS5lLiUyQyUyMHRoZSUyMGxlbmd0aCUyMG9mJTIwdGhlJTIwdG9rZW5pemVyLiUwQW1vZGVsLnJlc2l6ZV90b2tlbl9lbWJlZGRpbmdzKGxlbih0b2tlbml6ZXIpKSUwQSUwQWFzc2VydCUyMHRva2VuaXplci5jbHNfdG9rZW4lMjAlM0QlM0QlMjAlMjIlM0NDTFMlM0UlMjI=",highlighted:`<span class="hljs-comment"># Let&#x27;s see how to add a new classification token to GPT-2</span>
tokenizer = GPT2Tokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
model = GPT2Model.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

special_tokens_dict = {<span class="hljs-string">&quot;cls_token&quot;</span>: <span class="hljs-string">&quot;&lt;CLS&gt;&quot;</span>}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;We have added&quot;</span>, num_added_toks, <span class="hljs-string">&quot;tokens&quot;</span>)
<span class="hljs-comment"># Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.</span>
model.resize_token_embeddings(<span class="hljs-built_in">len</span>(tokenizer))

<span class="hljs-keyword">assert</span> tokenizer.cls_token == <span class="hljs-string">&quot;&lt;CLS&gt;&quot;</span>`,wrap:!1}}),{c(){i=r("p"),i.textContent=z,k=n(),m(T.$$.fragment)},l(p){i=s(p,"P",{"data-svelte-h":!0}),c(i)!=="svelte-kvfsh7"&&(i.textContent=z),k=o(p),u(T.$$.fragment,p)},m(p,P){w(p,i,P),w(p,k,P),h(T,p,P),$=!0},p:ze,i(p){$||(f(T.$$.fragment,p),$=!0)},o(p){g(T.$$.fragment,p),$=!1},d(p){p&&(a(i),a(k)),_(T,p)}}}function sa(M){let i,z="Examples:",k,T,$;return T=new no({props:{code:"JTIzJTIwTGV0J3MlMjBzZWUlMjBob3clMjB0byUyMGluY3JlYXNlJTIwdGhlJTIwdm9jYWJ1bGFyeSUyMG9mJTIwQmVydCUyMG1vZGVsJTIwYW5kJTIwdG9rZW5pemVyJTBBdG9rZW5pemVyJTIwJTNEJTIwQmVydFRva2VuaXplckZhc3QuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0TW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBbnVtX2FkZGVkX3Rva3MlMjAlM0QlMjB0b2tlbml6ZXIuYWRkX3Rva2VucyglNUIlMjJuZXdfdG9rMSUyMiUyQyUyMCUyMm15X25ldy10b2syJTIyJTVEKSUwQXByaW50KCUyMldlJTIwaGF2ZSUyMGFkZGVkJTIyJTJDJTIwbnVtX2FkZGVkX3Rva3MlMkMlMjAlMjJ0b2tlbnMlMjIpJTBBJTIzJTIwTm90aWNlJTNBJTIwcmVzaXplX3Rva2VuX2VtYmVkZGluZ3MlMjBleHBlY3QlMjB0byUyMHJlY2VpdmUlMjB0aGUlMjBmdWxsJTIwc2l6ZSUyMG9mJTIwdGhlJTIwbmV3JTIwdm9jYWJ1bGFyeSUyQyUyMGkuZS4lMkMlMjB0aGUlMjBsZW5ndGglMjBvZiUyMHRoZSUyMHRva2VuaXplci4lMEFtb2RlbC5yZXNpemVfdG9rZW5fZW1iZWRkaW5ncyhsZW4odG9rZW5pemVyKSk=",highlighted:`<span class="hljs-comment"># Let&#x27;s see how to increase the vocabulary of Bert model and tokenizer</span>
tokenizer = BertTokenizerFast.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = BertModel.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

num_added_toks = tokenizer.add_tokens([<span class="hljs-string">&quot;new_tok1&quot;</span>, <span class="hljs-string">&quot;my_new-tok2&quot;</span>])
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;We have added&quot;</span>, num_added_toks, <span class="hljs-string">&quot;tokens&quot;</span>)
<span class="hljs-comment"># Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.</span>
model.resize_token_embeddings(<span class="hljs-built_in">len</span>(tokenizer))`,wrap:!1}}),{c(){i=r("p"),i.textContent=z,k=n(),m(T.$$.fragment)},l(p){i=s(p,"P",{"data-svelte-h":!0}),c(i)!=="svelte-kvfsh7"&&(i.textContent=z),k=o(p),u(T.$$.fragment,p)},m(p,P){w(p,i,P),w(p,k,P),h(T,p,P),$=!0},p:ze,i(p){$||(f(T.$$.fragment,p),$=!0)},o(p){g(T.$$.fragment,p),$=!1},d(p){p&&(a(i),a(k)),_(T,p)}}}function aa(M){let i,z,k,T,$,p,P,Dr=`This page lists all the utility functions used by the tokenizers, mainly the class
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase">PreTrainedTokenizerBase</a> that implements the common methods between
<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> and the mixin
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.SpecialTokensMixin">SpecialTokensMixin</a>.`,Pn,$e,Er="Most of those are only useful if you are studying the code of the tokenizers in the library.",Bn,Pe,Mn,d,Be,oo,_t,Rr='Base class for <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>.',ro,kt,Ar="Handles shared (mostly boiler plate) methods for those two classes.",so,bt,Hr="Class attributes (overridden by derived classes)",ao,Tt,Gr=`<li><strong>vocab_files_names</strong> (<code>dict[str, str]</code>) — A dictionary with, as keys, the <code>__init__</code> keyword name of each
vocabulary file required by the model, and as associated values, the filename for saving the associated file
(string).</li> <li><strong>pretrained_vocab_files_map</strong> (<code>dict[str, dict[str, str]]</code>) — A dictionary of dictionaries, with the
high-level keys being the <code>__init__</code> keyword name of each vocabulary file required by the model, the
low-level being the <code>short-cut-names</code> of the pretrained models with, as associated values, the <code>url</code> to the
associated pretrained vocabulary file.</li> <li><strong>model_input_names</strong> (<code>list[str]</code>) — A list of inputs expected in the forward pass of the model.</li> <li><strong>padding_side</strong> (<code>str</code>) — The default value for the side on which the model should have padding applied.
Should be <code>&#39;right&#39;</code> or <code>&#39;left&#39;</code>.</li> <li><strong>truncation_side</strong> (<code>str</code>) — The default value for the side on which the model should have truncation
applied. Should be <code>&#39;right&#39;</code> or <code>&#39;left&#39;</code>.</li>`,io,K,Me,lo,vt,Xr=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`,co,ee,qe,po,yt,Yr=`Converts a list of dictionaries with <code>&quot;role&quot;</code> and <code>&quot;content&quot;</code> keys to a list of token
ids. This method is intended for use with chat models, and will read the tokenizer’s chat_template attribute to
determine the format and control tokens to use when converting.`,mo,te,Ie,uo,xt,Or=`Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
sequence-to-sequence models that need a slightly different processing for the labels.`,ho,ne,Ce,fo,wt,Qr="Convert a list of lists of token ids into a list of strings by calling decode.",go,j,We,_o,zt,Kr="Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.",ko,oe,bo,F,Ne,To,$t,es=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens.`,vo,Pt,ts="This implementation does not add special tokens and this method should be overridden in a subclass.",yo,re,Ue,xo,Bt,ns="Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.",wo,se,je,zo,Mt,os=`Converts a sequence of tokens in a single string. The most simple way to do it is <code>&quot; &quot;.join(tokens)</code> but we
often want to remove sub-word tokenization artifacts at the same time.`,$o,J,Fe,Po,qt,rs=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Bo,It,ss="Should be overridden in a subclass if the model has a special way of building those.",Mo,L,Je,qo,Ct,as=`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`,Io,Wt,is="Similar to doing <code>self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))</code>.",Co,S,Le,Wo,Nt,ds="Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.",No,Ut,ls="Same as doing <code>self.convert_tokens_to_ids(self.tokenize(text))</code>.",Uo,ae,Se,jo,jt,cs=`Tokenize a single message. This method is a convenience wrapper around <code>apply_chat_template</code> that allows you
to tokenize messages one by one. This is useful for things like token-by-token streaming.
This method is not guaranteed to be perfect. For some models, it may be impossible to robustly tokenize
single messages. For example, if the chat template adds tokens after each message, but also has a prefix that
is added to the entire chat, it will be impossible to distinguish a chat-start-token from a message-start-token.
In these cases, this method will do its best to find the correct tokenization, but it may not be perfect.
<strong>Note:</strong> This method does not support <code>add_generation_prompt</code>. If you want to add a generation prompt,
you should do it separately after tokenizing the conversation.`,Fo,Z,Ze,Jo,Ft,ps="Tokenize and prepare for the model a sequence or a pair of sequences.",Lo,ie,So,N,Ve,Zo,Jt,ms=`Instantiate a <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase">PreTrainedTokenizerBase</a> (or a derived class) from a predefined
tokenizer.`,Vo,de,Do,le,Eo,ce,De,Ro,Lt,us=`Retrieve the chat template string used for tokenizing chat messages. This template is used
internally by the <code>apply_chat_template</code> method and can also be used externally to retrieve the model’s chat
template for better generation tracking.`,Ao,pe,Ee,Ho,St,hs=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> or <code>encode_plus</code> methods.`,Go,V,Re,Xo,Zt,fs="Returns the vocabulary as a dictionary of token to index.",Yo,Vt,gs=`<code>tokenizer.get_vocab()[token]</code> is equivalent to <code>tokenizer.convert_tokens_to_ids(token)</code> when <code>token</code> is in the
vocab.`,Oo,C,Ae,Qo,Dt,_s=`Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
in the batch.`,Ko,Et,ks=`Padding side (left/right) padding token ids are defined at the tokenizer level (with <code>self.padding_side</code>,
<code>self.pad_token_id</code> and <code>self.pad_token_type_id</code>).`,er,Rt,bs=`Please note that with a fast tokenizer, using the <code>__call__</code> method is faster than using a method to encode the
text followed by a call to the <code>pad</code> method to get a padded encoding.`,tr,me,nr,ue,He,or,At,Ts=`Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
manages a moving window (with user defined stride) for overflowing tokens. Please Note, for <em>pair_ids</em>
different than <code>None</code> and <em>truncation_strategy = longest_first</em> or <code>True</code>, it is not possible to return
overflowing tokens. Such a combination of arguments will raise an error.`,rr,he,Ge,sr,Ht,vs="Prepare model inputs for translation. For best performance, translate one sentence at a time.",ar,D,Xe,ir,Gt,ys="Upload the tokenizer files to the 🤗 Model Hub.",dr,fe,lr,ge,Ye,cr,Xt,xs=`Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the
library are already mapped with <code>AutoTokenizer</code>.`,pr,_e,Oe,mr,Yt,ws=`Writes chat templates out to the save directory if we’re using the new format, and removes them from
the tokenizer config if present. If we’re using the legacy format, it doesn’t write any files, and instead
writes the templates to the tokenizer config in the correct format.`,ur,U,Qe,hr,Ot,zs="Save the full tokenizer state.",fr,Qt,$s=`This method make sure the full tokenizer can then be re-loaded using the
<code>~tokenization_utils_base.PreTrainedTokenizer.from_pretrained</code> class method..`,gr,Kt,Ps=`Warning,None This won’t save modifications you may have applied to the tokenizer after the instantiation (for
instance, modifying <code>tokenizer.do_lower_case</code> after creation).`,_r,E,Ke,kr,en,Bs="Save only the vocabulary of the tokenizer (vocabulary + added tokens).",br,tn,Ms=`This method won’t save the configuration and special token mappings of the tokenizer. Use
<code>_save_pretrained()</code> to save the whole state of the tokenizer.`,Tr,ke,et,vr,nn,qs="Converts a string into a sequence of tokens, replacing unknown tokens with the <code>unk_token</code>.",yr,be,tt,xr,on,Is="Truncates a sequence pair in-place following the strategy.",qn,nt,In,I,ot,wr,rn,Cs=`A mixin derived by <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> to handle specific behaviors related to
special tokens. In particular, this class hold the attributes which can be used to directly access these special
tokens in a model-independent manner and allow to set and update the special tokens.`,zr,B,rt,$r,sn,Ws=`Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
current vocabulary).`,Pr,an,Ns=`When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
model so that its embedding matrix matches the tokenizer.`,Br,dn,Us='In order to do that, please use the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings">resize_token_embeddings()</a> method.',Mr,ln,js="Using <code>add_special_tokens</code> will ensure your special tokens can be used in several ways:",qr,cn,Fs=`<li>Special tokens can be skipped when decoding using <code>skip_special_tokens = True</code>.</li> <li>Special tokens are carefully handled by the tokenizer (they are never split), similar to <code>AddedTokens</code>.</li> <li>You can easily refer to special tokens using tokenizer class attributes like <code>tokenizer.cls_token</code>. This
makes it easy to develop model-agnostic training and fine-tuning scripts.</li>`,Ir,pn,Js=`When possible, special tokens are already registered for provided pretrained models (for instance
<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a> <code>cls_token</code> is already registered to be <code>&#39;[CLS]&#39;</code> and XLM’s one is also registered to be
<code>&#39;&lt;/s&gt;&#39;</code>).`,Cr,Te,Wr,W,st,Nr,mn,Ls=`Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
it with indices starting from length of the current vocabulary and will be isolated before the tokenization
algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
not treated in the same way.`,Ur,un,Ss=`Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
of the model so that its embedding matrix matches the tokenizer.`,jr,hn,Zs='In order to do that, please use the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings">resize_token_embeddings()</a> method.',Fr,ve,Jr,ye,at,Lr,fn,Vs=`The <code>sanitize_special_tokens</code> is now deprecated kept for backward compatibility and will be removed in
transformers v5.`,Cn,it,Wn,G,dt,Sr,gn,Ds=`Possible values for the <code>truncation</code> argument in <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizerBase.<strong>call</strong>()</a>. Useful for tab-completion in
an IDE.`,Nn,X,lt,Zr,_n,Es="Character span in the original string.",Un,Y,ct,Vr,kn,Rs="Token span in an encoded string (list of tokens).",jn,pt,Fn,$n,Jn;return $=new to({props:{title:"Utilities for Tokenizers",local:"utilities-for-tokenizers",headingTag:"h1"}}),Pe=new to({props:{title:"PreTrainedTokenizerBase",local:"transformers.PreTrainedTokenizerBase",headingTag:"h2"}}),Be=new x({props:{name:"class transformers.PreTrainedTokenizerBase",anchor:"transformers.PreTrainedTokenizerBase",parameters:[{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.model_max_length",description:`<strong>model_max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
loaded with <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained">from_pretrained()</a>, this will be set to the
value stored for the associated model in <code>max_model_input_sizes</code> (see above). If no value is provided, will
default to VERY_LARGE_INTEGER (<code>int(1e30)</code>).`,name:"model_max_length"},{anchor:"transformers.PreTrainedTokenizerBase.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerBase.truncation_side",description:`<strong>truncation_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have truncation applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"truncation_side"},{anchor:"transformers.PreTrainedTokenizerBase.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A Jinja template string that will be used to format lists of chat messages. See
<a href="https://huggingface.co/docs/transformers/chat_templating" rel="nofollow">https://huggingface.co/docs/transformers/chat_templating</a> for a full description.`,name:"chat_template"},{anchor:"transformers.PreTrainedTokenizerBase.model_input_names",description:`<strong>model_input_names</strong> (<code>list[string]</code>, <em>optional</em>) &#x2014;
The list of inputs accepted by the forward pass of the model (like <code>&quot;token_type_ids&quot;</code> or
<code>&quot;attention_mask&quot;</code>). Default value is picked from the class attribute of the same name.`,name:"model_input_names"},{anchor:"transformers.PreTrainedTokenizerBase.bos_token",description:`<strong>bos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the beginning of a sentence. Will be associated to <code>self.bos_token</code> and
<code>self.bos_token_id</code>.`,name:"bos_token"},{anchor:"transformers.PreTrainedTokenizerBase.eos_token",description:`<strong>eos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the end of a sentence. Will be associated to <code>self.eos_token</code> and
<code>self.eos_token_id</code>.`,name:"eos_token"},{anchor:"transformers.PreTrainedTokenizerBase.unk_token",description:`<strong>unk_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing an out-of-vocabulary token. Will be associated to <code>self.unk_token</code> and
<code>self.unk_token_id</code>.`,name:"unk_token"},{anchor:"transformers.PreTrainedTokenizerBase.sep_token",description:`<strong>sep_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token separating two different sentences in the same input (used by BERT for instance). Will be
associated to <code>self.sep_token</code> and <code>self.sep_token_id</code>.`,name:"sep_token"},{anchor:"transformers.PreTrainedTokenizerBase.pad_token",description:`<strong>pad_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation. Will be associated to <code>self.pad_token</code> and <code>self.pad_token_id</code>.`,name:"pad_token"},{anchor:"transformers.PreTrainedTokenizerBase.cls_token",description:`<strong>cls_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the class of the input (used by BERT for instance). Will be associated to
<code>self.cls_token</code> and <code>self.cls_token_id</code>.`,name:"cls_token"},{anchor:"transformers.PreTrainedTokenizerBase.mask_token",description:`<strong>mask_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT). Will be associated to <code>self.mask_token</code> and <code>self.mask_token_id</code>.`,name:"mask_token"},{anchor:"transformers.PreTrainedTokenizerBase.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (tuple or list of <code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A tuple or a list of additional special tokens. Add them here to ensure they are skipped when decoding with
<code>skip_special_tokens</code> is set to True. If they are not part of the vocabulary, they will be added at the end
of the vocabulary.`,name:"additional_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should cleanup the spaces that were added when splitting the input text during the
tokenization process.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizerBase.split_special_tokens",description:`<strong>split_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the special tokens should be split during the tokenization process. Passing will affect the
internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
<code>&lt;s&gt;</code> is the <code>bos_token</code>, then <code>tokenizer.tokenize(&quot;&lt;s&gt;&quot;) = [&apos;&lt;s&gt;</code>]. Otherwise, if
<code>split_special_tokens=True</code>, then <code>tokenizer.tokenize(&quot;&lt;s&gt;&quot;)</code> will be give <code>[&apos;&lt;&apos;,&apos;s&apos;, &apos;&gt;&apos;]</code>.`,name:"split_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1375"}}),Me=new x({props:{name:"__call__",anchor:"transformers.PreTrainedTokenizerBase.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.PreTrainedTokenizerBase.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
`}}),qe=new x({props:{name:"apply_chat_template",anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template",parameters:[{name:"conversation",val:": typing.Union[list[dict[str, str]], list[list[dict[str, str]]]]"},{name:"tools",val:": typing.Optional[list[typing.Union[dict, typing.Callable]]] = None"},{name:"documents",val:": typing.Optional[list[dict[str, str]]] = None"},{name:"chat_template",val:": typing.Optional[str] = None"},{name:"add_generation_prompt",val:": bool = False"},{name:"continue_final_message",val:": bool = False"},{name:"tokenize",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": bool = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_dict",val:": bool = False"},{name:"return_assistant_tokens_mask",val:": bool = False"},{name:"tokenizer_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.conversation",description:`<strong>conversation</strong> (Union[list[dict[str, str]], list[list[dict[str, str]]]]) &#x2014; A list of dicts
with &#x201C;role&#x201D; and &#x201C;content&#x201D; keys, representing the chat history so far.`,name:"conversation"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.tools",description:`<strong>tools</strong> (<code>list[Union[Dict, Callable]]</code>, <em>optional</em>) &#x2014;
A list of tools (callable functions) that will be accessible to the model. If the template does not
support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
giving the name, description and argument types for the tool. See our
<a href="https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use" rel="nofollow">chat templating guide</a>
for more information.`,name:"tools"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.documents",description:`<strong>documents</strong> (<code>list[dict[str, str]]</code>, <em>optional</em>) &#x2014;
A list of dicts representing documents that will be accessible to the model if it is performing RAG
(retrieval-augmented generation). If the template does not support RAG, this argument will have no
effect. We recommend that each document should be a dict containing &#x201C;title&#x201D; and &#x201C;text&#x201D; keys. Please
see the RAG section of the <a href="https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG" rel="nofollow">chat templating guide</a>
for examples of passing documents with chat templates.`,name:"documents"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
argument, as the model&#x2019;s template will be used by default.`,name:"chat_template"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt",description:`<strong>add_generation_prompt</strong> (bool, <em>optional</em>) &#x2014;
If this is set, a prompt with the token(s) that indicate
the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
Note that this argument will be passed to the chat template, and so it must be supported in the
template for this argument to have any effect.`,name:"add_generation_prompt"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message",description:`<strong>continue_final_message</strong> (bool, <em>optional</em>) &#x2014;
If this is set, the chat will be formatted so that the final
message in the chat is open-ended, without any EOS tokens. The model will continue this message
rather than starting a new one. This allows you to &#x201C;prefill&#x201D; part of
the model&#x2019;s response for it. Cannot be used at the same time as <code>add_generation_prompt</code>.`,name:"continue_final_message"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.tokenize",description:`<strong>tokenize</strong> (<code>bool</code>, defaults to <code>True</code>) &#x2014;
Whether to tokenize the output. If <code>False</code>, the output will be a string.`,name:"tokenize"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.truncation",description:`<strong>truncation</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to truncate sequences at the maximum length. Has no effect if tokenize is <code>False</code>.`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is <code>False</code>. If
not specified, the tokenizer&#x2019;s <code>max_length</code> attribute will be used as a default.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Has no effect if tokenize is <code>False</code>. Acceptable
values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.Tensor</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to return a dictionary with named outputs. Has no effect if tokenize is <code>False</code>.`,name:"return_dict"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.tokenizer_kwargs",description:"<strong>tokenizer_kwargs</strong> (<code>dict[str -- Any]</code>, <em>optional</em>): Additional kwargs to pass to the tokenizer.",name:"tokenizer_kwargs"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.return_assistant_tokens_mask",description:`<strong>return_assistant_tokens_mask</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
the mask will contain 1. For user and system tokens, the mask will contain 0.
This functionality is only available for chat templates that support it via the <code>{% generation %}</code> keyword.`,name:"return_assistant_tokens_mask"},{anchor:"transformers.PreTrainedTokenizerBase.apply_chat_template.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; Additional kwargs to pass to the template renderer. Will be accessible by the chat template.",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1518",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of token ids representing the tokenized chat so far, including control tokens. This
output is ready to pass to the model, either directly or via methods like <code>generate()</code>. If <code>return_dict</code> is
set, will return a dict of tokenizer outputs instead.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Union[list[int], Dict]</code></p>
`}}),Ie=new x({props:{name:"as_target_tokenizer",anchor:"transformers.PreTrainedTokenizerBase.as_target_tokenizer",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L4001"}}),Ce=new x({props:{name:"batch_decode",anchor:"transformers.PreTrainedTokenizerBase.batch_decode",parameters:[{name:"sequences",val:": typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.batch_decode.sequences",description:`<strong>sequences</strong> (<code>Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"sequences"},{anchor:"transformers.PreTrainedTokenizerBase.batch_decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.batch_decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizerBase.batch_decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3833",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of decoded sentences.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code></p>
`}}),We=new x({props:{name:"batch_encode_plus",anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus",parameters:[{name:"batch_text_or_text_pairs",val:": typing.Union[list[str], list[tuple[str, str]], list[list[str]], list[tuple[list[str], list[str]]], list[list[int]], list[tuple[list[int], list[int]]]]"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"split_special_tokens",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.batch_text_or_text_pairs",description:`<strong>batch_text_or_text_pairs</strong> (<code>list[str]</code>, <code>list[tuple[str, str]]</code>, <code>list[list[str]]</code>, <code>list[tuple[list[str], list[str]]]</code>, and for not-fast tokenizers, also <code>list[list[int]]</code>, <code>list[tuple[list[int], list[int]]]</code>) &#x2014;
Batch of sequences or pair of sequences to be encoded. This can be a list of
string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
details in <code>encode_plus</code>).`,name:"batch_text_or_text_pairs"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.PreTrainedTokenizerBase.batch_encode_plus.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3144",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
`}}),oe=new Kn({props:{warning:!0,$$slots:{default:[Qs]},$$scope:{ctx:M}}}),Ne=new x({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.PreTrainedTokenizerBase.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.build_inputs_with_special_tokens.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.PreTrainedTokenizerBase.build_inputs_with_special_tokens.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3456",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The model input with special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ue=new x({props:{name:"clean_up_tokenization",anchor:"transformers.PreTrainedTokenizerBase.clean_up_tokenization",parameters:[{name:"out_string",val:": str"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.clean_up_tokenization.out_string",description:"<strong>out_string</strong> (<code>str</code>) &#x2014; The text to clean up.",name:"out_string"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3944",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The cleaned-up string.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),je=new x({props:{name:"convert_tokens_to_string",anchor:"transformers.PreTrainedTokenizerBase.convert_tokens_to_string",parameters:[{name:"tokens",val:": list"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.convert_tokens_to_string.tokens",description:"<strong>tokens</strong> (<code>list[str]</code>) &#x2014; The token to join in a string.",name:"tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3820",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The joined tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),Fe=new x({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.PreTrainedTokenizerBase.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.PreTrainedTokenizerBase.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Je=new x({props:{name:"decode",anchor:"transformers.PreTrainedTokenizerBase.decode",parameters:[{name:"token_ids",val:": typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.decode.token_ids",description:`<strong>token_ids</strong> (<code>Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"token_ids"},{anchor:"transformers.PreTrainedTokenizerBase.decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizerBase.decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3867",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded sentence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),Le=new x({props:{name:"encode",anchor:"transformers.PreTrainedTokenizerBase.encode",parameters:[{name:"text",val:": typing.Union[str, list[str], list[int]]"},{name:"text_pair",val:": typing.Union[str, list[str], list[int], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.encode.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code> or <code>list[int]</code>) &#x2014;
The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
<code>tokenize</code> method) or a list of integers (tokenized string ids using the <code>convert_tokens_to_ids</code>
method).`,name:"text"},{anchor:"transformers.PreTrainedTokenizerBase.encode.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code> or <code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
the <code>tokenize</code> method) or a list of integers (tokenized string ids using the <code>convert_tokens_to_ids</code>
method).`,name:"text_pair"},{anchor:"transformers.PreTrainedTokenizerBase.encode.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.encode.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerBase.encode.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerBase.encode.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerBase.encode.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizerBase.encode.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizerBase.encode.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerBase.encode.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerBase.encode.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerBase.encode.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; Passed along to the <code>.tokenize()</code> method.",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2667",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The tokenized ids of the text.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code>, <code>torch.Tensor</code>, <code>tf.Tensor</code> or <code>np.ndarray</code></p>
`}}),Se=new x({props:{name:"encode_message_with_chat_template",anchor:"transformers.PreTrainedTokenizerBase.encode_message_with_chat_template",parameters:[{name:"message",val:": dict"},{name:"conversation_history",val:": typing.Optional[list[dict[str, str]]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.encode_message_with_chat_template.message",description:`<strong>message</strong> (<code>dict</code>) &#x2014;
A dictionary with &#x201C;role&#x201D; and &#x201C;content&#x201D; keys, representing the message to tokenize.`,name:"message"},{anchor:"transformers.PreTrainedTokenizerBase.encode_message_with_chat_template.conversation_history",description:`<strong>conversation_history</strong> (<code>list[dict]</code>, <em>optional</em>) &#x2014;
A list of dicts with &#x201C;role&#x201D; and &#x201C;content&#x201D; keys, representing the chat history so far. If you are
tokenizing messages one by one, you should pass the previous messages in the conversation here.`,name:"conversation_history"},{anchor:"transformers.PreTrainedTokenizerBase.encode_message_with_chat_template.*kwargs",description:`*<strong>*kwargs</strong> &#x2014;
Additional kwargs to pass to the <code>apply_chat_template</code> method.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1697",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of token ids representing the tokenized message.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ze=new x({props:{name:"encode_plus",anchor:"transformers.PreTrainedTokenizerBase.encode_plus",parameters:[{name:"text",val:": typing.Union[str, list[str], list[int]]"},{name:"text_pair",val:": typing.Union[str, list[str], list[int], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code> or (for non-fast tokenizers) <code>list[int]</code>) &#x2014;
The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
<code>tokenize</code> method) or a list of integers (tokenized string ids using the <code>convert_tokens_to_ids</code>
method).`,name:"text"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code> or <code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
the <code>tokenize</code> method) or a list of integers (tokenized string ids using the <code>convert_tokens_to_ids</code>
method).`,name:"text_pair"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.PreTrainedTokenizerBase.encode_plus.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3044",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
`}}),ie=new Kn({props:{warning:!0,$$slots:{default:[Ks]},$$scope:{ctx:M}}}),Ve=new x({props:{name:"from_pretrained",anchor:"transformers.PreTrainedTokenizerBase.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:": typing.Union[str, os.PathLike]"},{name:"*init_inputs",val:""},{name:"cache_dir",val:": typing.Union[str, os.PathLike, NoneType] = None"},{name:"force_download",val:": bool = False"},{name:"local_files_only",val:": bool = False"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"revision",val:": str = 'main'"},{name:"trust_remote_code",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.pretrained_model_name_or_path",description:`<strong>pretrained_model_name_or_path</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
Can be either:</p>
<ul>
<li>A string, the <em>model id</em> of a predefined tokenizer hosted inside a model repo on huggingface.co.</li>
<li>A path to a <em>directory</em> containing vocabulary files required by the tokenizer, for instance saved
using the <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained">save_pretrained()</a> method, e.g.,
<code>./my_model_directory/</code>.</li>
<li>(<strong>Deprecated</strong>, not applicable to all derived classes) A path or url to a single saved vocabulary
file (if and only if the tokenizer only requires a single vocabulary file like Bert or XLNet), e.g.,
<code>./my_model_directory/vocab.txt</code>.</li>
</ul>`,name:"pretrained_model_name_or_path"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.cache_dir",description:`<strong>cache_dir</strong> (<code>str</code> or <code>os.PathLike</code>, <em>optional</em>) &#x2014;
Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
standard cache should not be used.`,name:"cache_dir"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.force_download",description:`<strong>force_download</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
exist.`,name:"force_download"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.resume_download",description:`<strong>resume_download</strong> &#x2014;
Deprecated and ignored. All downloads are now resumed by default when possible.
Will be removed in v5 of Transformers.`,name:"resume_download"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.proxies",description:`<strong>proxies</strong> (<code>dict[str, str]</code>, <em>optional</em>) &#x2014;
A dictionary of proxy servers to use by protocol or endpoint, e.g., <code>{&apos;http&apos;: &apos;foo.bar:3128&apos;, &apos;http://hostname&apos;: &apos;foo.bar:4012&apos;}</code>. The proxies are used on each request.`,name:"proxies"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.token",description:`<strong>token</strong> (<code>str</code> or <em>bool</em>, <em>optional</em>) &#x2014;
The token to use as HTTP bearer authorization for remote files. If <code>True</code>, will use the token generated
when running <code>hf auth login</code> (stored in <code>~/.huggingface</code>).`,name:"token"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.local_files_only",description:`<strong>local_files_only</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to only rely on local files and not to attempt to download any files.`,name:"local_files_only"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.revision",description:`<strong>revision</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;main&quot;</code>) &#x2014;
The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
git-based system for storing models and other artifacts on huggingface.co, so <code>revision</code> can be any
identifier allowed by git.`,name:"revision"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.subfolder",description:`<strong>subfolder</strong> (<code>str</code>, <em>optional</em>) &#x2014;
In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
facebook/rag-token-base), specify it here.`,name:"subfolder"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.inputs",description:`<strong>inputs</strong> (additional positional arguments, <em>optional</em>) &#x2014;
Will be passed along to the Tokenizer <code>__init__</code> method.`,name:"inputs"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.trust_remote_code",description:`<strong>trust_remote_code</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
should only be set to <code>True</code> for repositories you trust and in which you have read the code, as it will
execute code present on the Hub on your local machine.`,name:"trust_remote_code"},{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the Tokenizer <code>__init__</code> method. Can be used to set special tokens like <code>bos_token</code>,
<code>eos_token</code>, <code>unk_token</code>, <code>sep_token</code>, <code>pad_token</code>, <code>cls_token</code>, <code>mask_token</code>,
<code>additional_special_tokens</code>. See parameters in the <code>__init__</code> for more details.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1807"}}),de=new Kn({props:{$$slots:{default:[ea]},$$scope:{ctx:M}}}),le=new eo({props:{anchor:"transformers.PreTrainedTokenizerBase.from_pretrained.example",$$slots:{default:[ta]},$$scope:{ctx:M}}}),De=new x({props:{name:"get_chat_template",anchor:"transformers.PreTrainedTokenizerBase.get_chat_template",parameters:[{name:"chat_template",val:": typing.Optional[str] = None"},{name:"tools",val:": typing.Optional[list[dict]] = None"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.get_chat_template.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A Jinja template or the name of a template to use for this conversion.
It is usually not necessary to pass anything to this argument,
as the model&#x2019;s template will be used by default.`,name:"chat_template"},{anchor:"transformers.PreTrainedTokenizerBase.get_chat_template.tools",description:`<strong>tools</strong> (<code>list[Dict]</code>, <em>optional</em>) &#x2014;
A list of tools (callable functions) that will be accessible to the model. If the template does not
support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
giving the name, description and argument types for the tool. See our
<a href="https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use" rel="nofollow">chat templating guide</a>
for more information.`,name:"tools"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1753",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The chat template string.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),Ee=new x({props:{name:"get_special_tokens_mask",anchor:"transformers.PreTrainedTokenizerBase.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids of the first sequence.`,name:"token_ids_0"},{anchor:"transformers.PreTrainedTokenizerBase.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
List of ids of the second sequence.`,name:"token_ids_1"},{anchor:"transformers.PreTrainedTokenizerBase.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3913",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]</p>
`}}),Re=new x({props:{name:"get_vocab",anchor:"transformers.PreTrainedTokenizerBase.get_vocab",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1506",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>dict[str, int]</code></p>
`}}),Ae=new x({props:{name:"pad",anchor:"transformers.PreTrainedTokenizerBase.pad",parameters:[{name:"encoded_inputs",val:": typing.Union[transformers.tokenization_utils_base.BatchEncoding, list[transformers.tokenization_utils_base.BatchEncoding], dict[str, list[int]], dict[str, list[list[int]]], list[dict[str, list[int]]]]"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"verbose",val:": bool = True"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.pad.encoded_inputs",description:`<strong>encoded_inputs</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, list of <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, <code>dict[str, list[int]]</code>, <code>dict[str, list[list[int]]</code> or <code>list[dict[str, list[int]]]</code>) &#x2014;
Tokenized inputs. Can represent one input (<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a> or <code>dict[str, list[int]]</code>) or a batch of
tokenized inputs (list of <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, <em>dict[str, list[list[int]]]</em> or <em>list[dict[str,
list[int]]]</em>) so you can use this method during preprocessing as well as in a PyTorch Dataloader
collate function.</p>
<p>Instead of <code>list[int]</code> you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
the note above for the return type.`,name:"encoded_inputs"},{anchor:"transformers.PreTrainedTokenizerBase.pad.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code> (default): Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code>: No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerBase.pad.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length of the returned list and optionally padding length (see above).`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerBase.pad.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value.</p>
<p>This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerBase.pad.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerBase.pad.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.PreTrainedTokenizerBase.pad.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerBase.pad.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3253"}}),me=new Kn({props:{$$slots:{default:[na]},$$scope:{ctx:M}}}),He=new x({props:{name:"prepare_for_model",anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model",parameters:[{name:"ids",val:": list"},{name:"pair_ids",val:": typing.Optional[list[int]] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"prepend_batch_axis",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.ids",description:`<strong>ids</strong> (<code>list[int]</code>) &#x2014;
Tokenized input ids of the first sequence. Can be obtained from a string by chaining the <code>tokenize</code> and
<code>convert_tokens_to_ids</code> methods.`,name:"ids"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.pair_ids",description:`<strong>pair_ids</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Tokenized input ids of the second sequence. Can be obtained from a string by chaining the <code>tokenize</code>
and <code>convert_tokens_to_ids</code> methods.`,name:"pair_ids"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_for_model.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3476",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
`}}),Ge=new x({props:{name:"prepare_seq2seq_batch",anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch",parameters:[{name:"src_texts",val:": list"},{name:"tgt_texts",val:": typing.Optional[list[str]] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"max_target_length",val:": typing.Optional[int] = None"},{name:"padding",val:": str = 'longest'"},{name:"return_tensors",val:": typing.Optional[str] = None"},{name:"truncation",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch.src_texts",description:`<strong>src_texts</strong> (<code>list[str]</code>) &#x2014;
List of documents to summarize or source language texts.`,name:"src_texts"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch.tgt_texts",description:`<strong>tgt_texts</strong> (<code>list</code>, <em>optional</em>) &#x2014;
List of summaries or target language texts.`,name:"tgt_texts"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length for encoder inputs (documents to summarize or source language texts) If
left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length is
required by one of the truncation/padding parameters. If the model has no specific maximum input length
(like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch.max_target_length",description:`<strong>max_target_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length of decoder inputs (target language texts or summaries) If left unset or set
to <code>None</code>, this will use the max_length value.`,name:"max_target_length"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerBase.prepare_seq2seq_batch.*kwargs",description:`*<strong>*kwargs</strong> &#x2014;
Additional keyword arguments passed along to <code>self.__call__</code>.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L4040",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li><strong>input_ids</strong> — List of token ids to be fed to the encoder.</li>
<li><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model.</li>
<li><strong>labels</strong> — List of token ids for tgt_texts.</li>
</ul>
<p>The full set of keys <code>[input_ids, attention_mask, labels]</code>, will only be returned if tgt_texts is passed.
Otherwise, input_ids, attention_mask will be the only keys.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),Xe=new x({props:{name:"push_to_hub",anchor:"transformers.PreTrainedTokenizerBase.push_to_hub",parameters:[{name:"repo_id",val:": str"},{name:"use_temp_dir",val:": typing.Optional[bool] = None"},{name:"commit_message",val:": typing.Optional[str] = None"},{name:"private",val:": typing.Optional[bool] = None"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"max_shard_size",val:": typing.Union[str, int, NoneType] = '5GB'"},{name:"create_pr",val:": bool = False"},{name:"safe_serialization",val:": bool = True"},{name:"revision",val:": typing.Optional[str] = None"},{name:"commit_description",val:": typing.Optional[str] = None"},{name:"tags",val:": typing.Optional[list[str]] = None"},{name:"**deprecated_kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.repo_id",description:`<strong>repo_id</strong> (<code>str</code>) &#x2014;
The name of the repository you want to push your tokenizer to. It should contain your organization name
when pushing to a given organization.`,name:"repo_id"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.use_temp_dir",description:`<strong>use_temp_dir</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
Will default to <code>True</code> if there is no directory named like <code>repo_id</code>, <code>False</code> otherwise.`,name:"use_temp_dir"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.commit_message",description:`<strong>commit_message</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Message to commit while pushing. Will default to <code>&quot;Upload tokenizer&quot;</code>.`,name:"commit_message"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.private",description:`<strong>private</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to make the repo private. If <code>None</code> (default), the repo will be public unless the organization&#x2019;s default is private. This value is ignored if the repo already exists.`,name:"private"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.token",description:`<strong>token</strong> (<code>bool</code> or <code>str</code>, <em>optional</em>) &#x2014;
The token to use as HTTP bearer authorization for remote files. If <code>True</code>, will use the token generated
when running <code>hf auth login</code> (stored in <code>~/.huggingface</code>). Will default to <code>True</code> if <code>repo_url</code>
is not specified.`,name:"token"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.max_shard_size",description:`<strong>max_shard_size</strong> (<code>int</code> or <code>str</code>, <em>optional</em>, defaults to <code>&quot;5GB&quot;</code>) &#x2014;
Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
will then be each of size lower than this size. If expressed as a string, needs to be digits followed
by a unit (like <code>&quot;5MB&quot;</code>). We default it to <code>&quot;5GB&quot;</code> so that users can easily load models on free-tier
Google Colab instances without any CPU OOM issues.`,name:"max_shard_size"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.create_pr",description:`<strong>create_pr</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to create a PR with the uploaded files or directly commit.`,name:"create_pr"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.safe_serialization",description:`<strong>safe_serialization</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to convert the model weights in safetensors format for safer serialization.`,name:"safe_serialization"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.revision",description:`<strong>revision</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Branch to push the uploaded files to.`,name:"revision"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.commit_description",description:`<strong>commit_description</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The description of the commit that will be created`,name:"commit_description"},{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.tags",description:`<strong>tags</strong> (<code>list[str]</code>, <em>optional</em>) &#x2014;
List of tags to push on the Hub.`,name:"tags"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L847"}}),fe=new eo({props:{anchor:"transformers.PreTrainedTokenizerBase.push_to_hub.example",$$slots:{default:[oa]},$$scope:{ctx:M}}}),Ye=new x({props:{name:"register_for_auto_class",anchor:"transformers.PreTrainedTokenizerBase.register_for_auto_class",parameters:[{name:"auto_class",val:" = 'AutoTokenizer'"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.register_for_auto_class.auto_class",description:`<strong>auto_class</strong> (<code>str</code> or <code>type</code>, <em>optional</em>, defaults to <code>&quot;AutoTokenizer&quot;</code>) &#x2014;
The auto class to register this new tokenizer with.`,name:"auto_class"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L4018"}}),Oe=new x({props:{name:"save_chat_templates",anchor:"transformers.PreTrainedTokenizerBase.save_chat_templates",parameters:[{name:"save_directory",val:": typing.Union[str, os.PathLike]"},{name:"tokenizer_config",val:": dict"},{name:"filename_prefix",val:": typing.Optional[str]"},{name:"save_jinja_files",val:": bool"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2370"}}),Qe=new x({props:{name:"save_pretrained",anchor:"transformers.PreTrainedTokenizerBase.save_pretrained",parameters:[{name:"save_directory",val:": typing.Union[str, os.PathLike]"},{name:"legacy_format",val:": typing.Optional[bool] = None"},{name:"filename_prefix",val:": typing.Optional[str] = None"},{name:"push_to_hub",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.save_pretrained.save_directory",description:"<strong>save_directory</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014; The path to a directory where the tokenizer will be saved.",name:"save_directory"},{anchor:"transformers.PreTrainedTokenizerBase.save_pretrained.legacy_format",description:`<strong>legacy_format</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Only applicable for a fast tokenizer. If unset (default), will save the tokenizer in the unified JSON
format as well as in legacy format if it exists, i.e. with tokenizer specific vocabulary and a separate
added_tokens files.</p>
<p>If <code>False</code>, will only save the tokenizer in the unified JSON format. This format is incompatible with
&#x201C;slow&#x201D; tokenizers (not powered by the <em>tokenizers</em> library), so the tokenizer will not be able to be
loaded in the corresponding &#x201C;slow&#x201D; tokenizer.</p>
<p>If <code>True</code>, will save the tokenizer in legacy format. If the &#x201C;slow&#x201D; tokenizer doesn&#x2019;t exits, a value
error is raised.`,name:"legacy_format"},{anchor:"transformers.PreTrainedTokenizerBase.save_pretrained.filename_prefix",description:`<strong>filename_prefix</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A prefix to add to the names of the files saved by the tokenizer.`,name:"filename_prefix"},{anchor:"transformers.PreTrainedTokenizerBase.save_pretrained.push_to_hub",description:`<strong>push_to_hub</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with <code>repo_id</code> (will default to the name of <code>save_directory</code> in your
namespace).`,name:"push_to_hub"},{anchor:"transformers.PreTrainedTokenizerBase.save_pretrained.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Additional key word arguments passed along to the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub">push_to_hub()</a> method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2425",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The files saved.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A tuple of <code>str</code></p>
`}}),Ke=new x({props:{name:"save_vocabulary",anchor:"transformers.PreTrainedTokenizerBase.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.save_vocabulary.save_directory",description:`<strong>save_directory</strong> (<code>str</code>) &#x2014;
The directory in which to save the vocabulary.`,name:"save_directory"},{anchor:"transformers.PreTrainedTokenizerBase.save_vocabulary.filename_prefix",description:`<strong>filename_prefix</strong> (<code>str</code>, <em>optional</em>) &#x2014;
An optional prefix to add to the named of the saved files.`,name:"filename_prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2629",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Paths to the files saved.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Tuple(str)</code></p>
`}}),et=new x({props:{name:"tokenize",anchor:"transformers.PreTrainedTokenizerBase.tokenize",parameters:[{name:"text",val:": str"},{name:"pair",val:": typing.Optional[str] = None"},{name:"add_special_tokens",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.tokenize.text",description:`<strong>text</strong> (<code>str</code>) &#x2014;
The sequence to be encoded.`,name:"text"},{anchor:"transformers.PreTrainedTokenizerBase.tokenize.pair",description:`<strong>pair</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A second sequence to be encoded with the first.`,name:"pair"},{anchor:"transformers.PreTrainedTokenizerBase.tokenize.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add the special tokens associated with the corresponding model.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizerBase.tokenize.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific encode method. See details in
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__"><strong>call</strong>()</a>`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2647",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code></p>
`}}),tt=new x({props:{name:"truncate_sequences",anchor:"transformers.PreTrainedTokenizerBase.truncate_sequences",parameters:[{name:"ids",val:": list"},{name:"pair_ids",val:": typing.Optional[list[int]] = None"},{name:"num_tokens_to_remove",val:": int = 0"},{name:"truncation_strategy",val:": typing.Union[str, transformers.tokenization_utils_base.TruncationStrategy] = 'longest_first'"},{name:"stride",val:": int = 0"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerBase.truncate_sequences.ids",description:`<strong>ids</strong> (<code>list[int]</code>) &#x2014;
Tokenized input ids of the first sequence. Can be obtained from a string by chaining the <code>tokenize</code> and
<code>convert_tokens_to_ids</code> methods.`,name:"ids"},{anchor:"transformers.PreTrainedTokenizerBase.truncate_sequences.pair_ids",description:`<strong>pair_ids</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Tokenized input ids of the second sequence. Can be obtained from a string by chaining the <code>tokenize</code>
and <code>convert_tokens_to_ids</code> methods.`,name:"pair_ids"},{anchor:"transformers.PreTrainedTokenizerBase.truncate_sequences.num_tokens_to_remove",description:`<strong>num_tokens_to_remove</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Number of tokens to remove using the truncation strategy.`,name:"num_tokens_to_remove"},{anchor:"transformers.PreTrainedTokenizerBase.truncate_sequences.truncation_strategy",description:`<strong>truncation_strategy</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>&apos;longest_first&apos;</code>) &#x2014;
The strategy to follow for truncation. Can be:</p>
<ul>
<li><code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will truncate
token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a
batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths greater
than the model maximum admissible input size).</li>
</ul>`,name:"truncation_strategy"},{anchor:"transformers.PreTrainedTokenizerBase.truncate_sequences.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a positive number, the overflowing tokens returned will contain some tokens from the main
sequence returned. The value of this argument defines the number of additional tokens.`,name:"stride"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3614",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The truncated <code>ids</code>, the truncated <code>pair_ids</code> and the list of
overflowing tokens. Note: The <em>longest_first</em> strategy returns empty list of overflowing tokens if a pair
of sequences (or a batch of pairs) is provided.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>tuple[list[int], list[int], list[int]]</code></p>
`}}),nt=new to({props:{title:"SpecialTokensMixin",local:"transformers.SpecialTokensMixin",headingTag:"h2"}}),ot=new x({props:{name:"class transformers.SpecialTokensMixin",anchor:"transformers.SpecialTokensMixin",parameters:[{name:"verbose",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpecialTokensMixin.bos_token",description:`<strong>bos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the beginning of a sentence.`,name:"bos_token"},{anchor:"transformers.SpecialTokensMixin.eos_token",description:`<strong>eos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the end of a sentence.`,name:"eos_token"},{anchor:"transformers.SpecialTokensMixin.unk_token",description:`<strong>unk_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing an out-of-vocabulary token.`,name:"unk_token"},{anchor:"transformers.SpecialTokensMixin.sep_token",description:`<strong>sep_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token separating two different sentences in the same input (used by BERT for instance).`,name:"sep_token"},{anchor:"transformers.SpecialTokensMixin.pad_token",description:`<strong>pad_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation.`,name:"pad_token"},{anchor:"transformers.SpecialTokensMixin.cls_token",description:`<strong>cls_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the class of the input (used by BERT for instance).`,name:"cls_token"},{anchor:"transformers.SpecialTokensMixin.mask_token",description:`<strong>mask_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT).`,name:"mask_token"},{anchor:"transformers.SpecialTokensMixin.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (tuple or list of <code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A tuple or a list of additional tokens, which will be marked as <code>special</code>, meaning that they will be
skipped when decoding if <code>skip_special_tokens</code> is set to <code>True</code>.`,name:"additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L818"}}),rt=new x({props:{name:"add_special_tokens",anchor:"transformers.SpecialTokensMixin.add_special_tokens",parameters:[{name:"special_tokens_dict",val:": dict"},{name:"replace_additional_special_tokens",val:" = True"}],parametersDescription:[{anchor:"transformers.SpecialTokensMixin.add_special_tokens.special_tokens_dict",description:`<strong>special_tokens_dict</strong> (dictionary <em>str</em> to <em>str</em>, <code>tokenizers.AddedToken</code>, or <code>Sequence[Union[str, AddedToken]]</code>) &#x2014;
Keys should be in the list of predefined special attributes: [<code>bos_token</code>, <code>eos_token</code>, <code>unk_token</code>,
<code>sep_token</code>, <code>pad_token</code>, <code>cls_token</code>, <code>mask_token</code>, <code>additional_special_tokens</code>].</p>
<p>Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
assign the index of the <code>unk_token</code> to them).`,name:"special_tokens_dict"},{anchor:"transformers.SpecialTokensMixin.add_special_tokens.replace_additional_special_tokens",description:`<strong>replace_additional_special_tokens</strong> (<code>bool</code>, <em>optional</em>,, defaults to <code>True</code>) &#x2014;
If <code>True</code>, the existing list of additional special tokens will be replaced by the list provided in
<code>special_tokens_dict</code>. Otherwise, <code>self._special_tokens_map[&quot;additional_special_tokens&quot;]</code> is just extended. In the former
case, the tokens will NOT be removed from the tokenizer&#x2019;s full vocabulary - they are only being flagged
as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
<code>added_tokens_encoder</code> and <code>added_tokens_decoder</code>. This means that the previous
<code>additional_special_tokens</code> are still added tokens, and will not be split by the model.`,name:"replace_additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L890",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of tokens added to the vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),Te=new eo({props:{anchor:"transformers.SpecialTokensMixin.add_special_tokens.example",$$slots:{default:[ra]},$$scope:{ctx:M}}}),st=new x({props:{name:"add_tokens",anchor:"transformers.SpecialTokensMixin.add_tokens",parameters:[{name:"new_tokens",val:": typing.Union[str, tokenizers.AddedToken, collections.abc.Sequence[typing.Union[str, tokenizers.AddedToken]]]"},{name:"special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.SpecialTokensMixin.add_tokens.new_tokens",description:`<strong>new_tokens</strong> (<code>str</code>, <code>tokenizers.AddedToken</code> or a sequence of <em>str</em> or <code>tokenizers.AddedToken</code>) &#x2014;
Tokens are only added if they are not already in the vocabulary. <code>tokenizers.AddedToken</code> wraps a string
token to let you personalize its behavior: whether this token should only match against a single word,
whether this token should strip all potential whitespaces on the left side, whether this token should
strip all potential whitespaces on the right side, etc.`,name:"new_tokens"},{anchor:"transformers.SpecialTokensMixin.add_tokens.special_tokens",description:`<strong>special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Can be used to specify if the token is a special token. This mostly change the normalization behavior
(special tokens like CLS or [MASK] are usually not lower-cased for instance).</p>
<p>See details for <code>tokenizers.AddedToken</code> in HuggingFace tokenizers library.`,name:"special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L994",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of tokens added to the vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),ve=new eo({props:{anchor:"transformers.SpecialTokensMixin.add_tokens.example",$$slots:{default:[sa]},$$scope:{ctx:M}}}),at=new x({props:{name:"sanitize_special_tokens",anchor:"transformers.SpecialTokensMixin.sanitize_special_tokens",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L882"}}),it=new to({props:{title:"Enums and namedtuples",local:"transformers.tokenization_utils_base.TruncationStrategy",headingTag:"h2"}}),dt=new x({props:{name:"class transformers.tokenization_utils_base.TruncationStrategy",anchor:"transformers.tokenization_utils_base.TruncationStrategy",parameters:[{name:"value",val:""},{name:"names",val:" = None"},{name:"module",val:" = None"},{name:"qualname",val:" = None"},{name:"type",val:" = None"},{name:"start",val:" = 1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L154"}}),lt=new x({props:{name:"class transformers.CharSpan",anchor:"transformers.CharSpan",parameters:[{name:"start",val:": int"},{name:"end",val:": int"}],parametersDescription:[{anchor:"transformers.CharSpan.start",description:"<strong>start</strong> (<code>int</code>) &#x2014; Index of the first character in the original string.",name:"start"},{anchor:"transformers.CharSpan.end",description:"<strong>end</strong> (<code>int</code>) &#x2014; Index of the character following the last character in the original string.",name:"end"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L166"}}),ct=new x({props:{name:"class transformers.TokenSpan",anchor:"transformers.TokenSpan",parameters:[{name:"start",val:": int"},{name:"end",val:": int"}],parametersDescription:[{anchor:"transformers.TokenSpan.start",description:"<strong>start</strong> (<code>int</code>) &#x2014; Index of the first token in the span.",name:"start"},{anchor:"transformers.TokenSpan.end",description:"<strong>end</strong> (<code>int</code>) &#x2014; Index of the token following the last token in the span.",name:"end"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L179"}}),pt=new Os({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/tokenization_utils.md"}}),{c(){i=r("meta"),z=n(),k=r("p"),T=n(),m($.$$.fragment),p=n(),P=r("p"),P.innerHTML=Dr,Pn=n(),$e=r("p"),$e.textContent=Er,Bn=n(),m(Pe.$$.fragment),Mn=n(),d=r("div"),m(Be.$$.fragment),oo=n(),_t=r("p"),_t.innerHTML=Rr,ro=n(),kt=r("p"),kt.textContent=Ar,so=n(),bt=r("p"),bt.textContent=Hr,ao=n(),Tt=r("ul"),Tt.innerHTML=Gr,io=n(),K=r("div"),m(Me.$$.fragment),lo=n(),vt=r("p"),vt.textContent=Xr,co=n(),ee=r("div"),m(qe.$$.fragment),po=n(),yt=r("p"),yt.innerHTML=Yr,mo=n(),te=r("div"),m(Ie.$$.fragment),uo=n(),xt=r("p"),xt.textContent=Or,ho=n(),ne=r("div"),m(Ce.$$.fragment),fo=n(),wt=r("p"),wt.textContent=Qr,go=n(),j=r("div"),m(We.$$.fragment),_o=n(),zt=r("p"),zt.textContent=Kr,ko=n(),m(oe.$$.fragment),bo=n(),F=r("div"),m(Ne.$$.fragment),To=n(),$t=r("p"),$t.textContent=es,vo=n(),Pt=r("p"),Pt.textContent=ts,yo=n(),re=r("div"),m(Ue.$$.fragment),xo=n(),Bt=r("p"),Bt.textContent=ns,wo=n(),se=r("div"),m(je.$$.fragment),zo=n(),Mt=r("p"),Mt.innerHTML=os,$o=n(),J=r("div"),m(Fe.$$.fragment),Po=n(),qt=r("p"),qt.innerHTML=rs,Bo=n(),It=r("p"),It.textContent=ss,Mo=n(),L=r("div"),m(Je.$$.fragment),qo=n(),Ct=r("p"),Ct.textContent=as,Io=n(),Wt=r("p"),Wt.innerHTML=is,Co=n(),S=r("div"),m(Le.$$.fragment),Wo=n(),Nt=r("p"),Nt.textContent=ds,No=n(),Ut=r("p"),Ut.innerHTML=ls,Uo=n(),ae=r("div"),m(Se.$$.fragment),jo=n(),jt=r("p"),jt.innerHTML=cs,Fo=n(),Z=r("div"),m(Ze.$$.fragment),Jo=n(),Ft=r("p"),Ft.textContent=ps,Lo=n(),m(ie.$$.fragment),So=n(),N=r("div"),m(Ve.$$.fragment),Zo=n(),Jt=r("p"),Jt.innerHTML=ms,Vo=n(),m(de.$$.fragment),Do=n(),m(le.$$.fragment),Eo=n(),ce=r("div"),m(De.$$.fragment),Ro=n(),Lt=r("p"),Lt.innerHTML=us,Ao=n(),pe=r("div"),m(Ee.$$.fragment),Ho=n(),St=r("p"),St.innerHTML=hs,Go=n(),V=r("div"),m(Re.$$.fragment),Xo=n(),Zt=r("p"),Zt.textContent=fs,Yo=n(),Vt=r("p"),Vt.innerHTML=gs,Oo=n(),C=r("div"),m(Ae.$$.fragment),Qo=n(),Dt=r("p"),Dt.textContent=_s,Ko=n(),Et=r("p"),Et.innerHTML=ks,er=n(),Rt=r("p"),Rt.innerHTML=bs,tr=n(),m(me.$$.fragment),nr=n(),ue=r("div"),m(He.$$.fragment),or=n(),At=r("p"),At.innerHTML=Ts,rr=n(),he=r("div"),m(Ge.$$.fragment),sr=n(),Ht=r("p"),Ht.textContent=vs,ar=n(),D=r("div"),m(Xe.$$.fragment),ir=n(),Gt=r("p"),Gt.textContent=ys,dr=n(),m(fe.$$.fragment),lr=n(),ge=r("div"),m(Ye.$$.fragment),cr=n(),Xt=r("p"),Xt.innerHTML=xs,pr=n(),_e=r("div"),m(Oe.$$.fragment),mr=n(),Yt=r("p"),Yt.textContent=ws,ur=n(),U=r("div"),m(Qe.$$.fragment),hr=n(),Ot=r("p"),Ot.textContent=zs,fr=n(),Qt=r("p"),Qt.innerHTML=$s,gr=n(),Kt=r("p"),Kt.innerHTML=Ps,_r=n(),E=r("div"),m(Ke.$$.fragment),kr=n(),en=r("p"),en.textContent=Bs,br=n(),tn=r("p"),tn.innerHTML=Ms,Tr=n(),ke=r("div"),m(et.$$.fragment),vr=n(),nn=r("p"),nn.innerHTML=qs,yr=n(),be=r("div"),m(tt.$$.fragment),xr=n(),on=r("p"),on.textContent=Is,qn=n(),m(nt.$$.fragment),In=n(),I=r("div"),m(ot.$$.fragment),wr=n(),rn=r("p"),rn.innerHTML=Cs,zr=n(),B=r("div"),m(rt.$$.fragment),$r=n(),sn=r("p"),sn.textContent=Ws,Pr=n(),an=r("p"),an.textContent=Ns,Br=n(),dn=r("p"),dn.innerHTML=Us,Mr=n(),ln=r("p"),ln.innerHTML=js,qr=n(),cn=r("ul"),cn.innerHTML=Fs,Ir=n(),pn=r("p"),pn.innerHTML=Js,Cr=n(),m(Te.$$.fragment),Wr=n(),W=r("div"),m(st.$$.fragment),Nr=n(),mn=r("p"),mn.textContent=Ls,Ur=n(),un=r("p"),un.textContent=Ss,jr=n(),hn=r("p"),hn.innerHTML=Zs,Fr=n(),m(ve.$$.fragment),Jr=n(),ye=r("div"),m(at.$$.fragment),Lr=n(),fn=r("p"),fn.innerHTML=Vs,Cn=n(),m(it.$$.fragment),Wn=n(),G=r("div"),m(dt.$$.fragment),Sr=n(),gn=r("p"),gn.innerHTML=Ds,Nn=n(),X=r("div"),m(lt.$$.fragment),Zr=n(),_n=r("p"),_n.textContent=Es,Un=n(),Y=r("div"),m(ct.$$.fragment),Vr=n(),kn=r("p"),kn.textContent=Rs,jn=n(),m(pt.$$.fragment),Fn=n(),$n=r("p"),this.h()},l(t){const b=Ys("svelte-u9bgzb",document.head);i=s(b,"META",{name:!0,content:!0}),b.forEach(a),z=o(t),k=s(t,"P",{}),v(k).forEach(a),T=o(t),u($.$$.fragment,t),p=o(t),P=s(t,"P",{"data-svelte-h":!0}),c(P)!=="svelte-u8b37w"&&(P.innerHTML=Dr),Pn=o(t),$e=s(t,"P",{"data-svelte-h":!0}),c($e)!=="svelte-1vo0znz"&&($e.textContent=Er),Bn=o(t),u(Pe.$$.fragment,t),Mn=o(t),d=s(t,"DIV",{class:!0});var l=v(d);u(Be.$$.fragment,l),oo=o(l),_t=s(l,"P",{"data-svelte-h":!0}),c(_t)!=="svelte-1edkrtl"&&(_t.innerHTML=Rr),ro=o(l),kt=s(l,"P",{"data-svelte-h":!0}),c(kt)!=="svelte-2oj7z8"&&(kt.textContent=Ar),so=o(l),bt=s(l,"P",{"data-svelte-h":!0}),c(bt)!=="svelte-1ixo79u"&&(bt.textContent=Hr),ao=o(l),Tt=s(l,"UL",{"data-svelte-h":!0}),c(Tt)!=="svelte-1gddudt"&&(Tt.innerHTML=Gr),io=o(l),K=s(l,"DIV",{class:!0});var mt=v(K);u(Me.$$.fragment,mt),lo=o(mt),vt=s(mt,"P",{"data-svelte-h":!0}),c(vt)!=="svelte-kpxj0c"&&(vt.textContent=Xr),mt.forEach(a),co=o(l),ee=s(l,"DIV",{class:!0});var ut=v(ee);u(qe.$$.fragment,ut),po=o(ut),yt=s(ut,"P",{"data-svelte-h":!0}),c(yt)!=="svelte-j87b6t"&&(yt.innerHTML=Yr),ut.forEach(a),mo=o(l),te=s(l,"DIV",{class:!0});var ht=v(te);u(Ie.$$.fragment,ht),uo=o(ht),xt=s(ht,"P",{"data-svelte-h":!0}),c(xt)!=="svelte-hxrl9"&&(xt.textContent=Or),ht.forEach(a),ho=o(l),ne=s(l,"DIV",{class:!0});var ft=v(ne);u(Ce.$$.fragment,ft),fo=o(ft),wt=s(ft,"P",{"data-svelte-h":!0}),c(wt)!=="svelte-1deng2j"&&(wt.textContent=Qr),ft.forEach(a),go=o(l),j=s(l,"DIV",{class:!0});var O=v(j);u(We.$$.fragment,O),_o=o(O),zt=s(O,"P",{"data-svelte-h":!0}),c(zt)!=="svelte-6p21pf"&&(zt.textContent=Kr),ko=o(O),u(oe.$$.fragment,O),O.forEach(a),bo=o(l),F=s(l,"DIV",{class:!0});var Q=v(F);u(Ne.$$.fragment,Q),To=o(Q),$t=s(Q,"P",{"data-svelte-h":!0}),c($t)!=="svelte-xip562"&&($t.textContent=es),vo=o(Q),Pt=s(Q,"P",{"data-svelte-h":!0}),c(Pt)!=="svelte-1yvfiyo"&&(Pt.textContent=ts),Q.forEach(a),yo=o(l),re=s(l,"DIV",{class:!0});var gt=v(re);u(Ue.$$.fragment,gt),xo=o(gt),Bt=s(gt,"P",{"data-svelte-h":!0}),c(Bt)!=="svelte-6a62nd"&&(Bt.textContent=ns),gt.forEach(a),wo=o(l),se=s(l,"DIV",{class:!0});var Ln=v(se);u(je.$$.fragment,Ln),zo=o(Ln),Mt=s(Ln,"P",{"data-svelte-h":!0}),c(Mt)!=="svelte-sfkaj8"&&(Mt.innerHTML=os),Ln.forEach(a),$o=o(l),J=s(l,"DIV",{class:!0});var bn=v(J);u(Fe.$$.fragment,bn),Po=o(bn),qt=s(bn,"P",{"data-svelte-h":!0}),c(qt)!=="svelte-zj1vf1"&&(qt.innerHTML=rs),Bo=o(bn),It=s(bn,"P",{"data-svelte-h":!0}),c(It)!=="svelte-9vptpw"&&(It.textContent=ss),bn.forEach(a),Mo=o(l),L=s(l,"DIV",{class:!0});var Tn=v(L);u(Je.$$.fragment,Tn),qo=o(Tn),Ct=s(Tn,"P",{"data-svelte-h":!0}),c(Ct)!=="svelte-vbfkpu"&&(Ct.textContent=as),Io=o(Tn),Wt=s(Tn,"P",{"data-svelte-h":!0}),c(Wt)!=="svelte-125uxon"&&(Wt.innerHTML=is),Tn.forEach(a),Co=o(l),S=s(l,"DIV",{class:!0});var vn=v(S);u(Le.$$.fragment,vn),Wo=o(vn),Nt=s(vn,"P",{"data-svelte-h":!0}),c(Nt)!=="svelte-12b8hzo"&&(Nt.textContent=ds),No=o(vn),Ut=s(vn,"P",{"data-svelte-h":!0}),c(Ut)!=="svelte-1kyhveh"&&(Ut.innerHTML=ls),vn.forEach(a),Uo=o(l),ae=s(l,"DIV",{class:!0});var Sn=v(ae);u(Se.$$.fragment,Sn),jo=o(Sn),jt=s(Sn,"P",{"data-svelte-h":!0}),c(jt)!=="svelte-1y16cvj"&&(jt.innerHTML=cs),Sn.forEach(a),Fo=o(l),Z=s(l,"DIV",{class:!0});var yn=v(Z);u(Ze.$$.fragment,yn),Jo=o(yn),Ft=s(yn,"P",{"data-svelte-h":!0}),c(Ft)!=="svelte-ma945j"&&(Ft.textContent=ps),Lo=o(yn),u(ie.$$.fragment,yn),yn.forEach(a),So=o(l),N=s(l,"DIV",{class:!0});var xe=v(N);u(Ve.$$.fragment,xe),Zo=o(xe),Jt=s(xe,"P",{"data-svelte-h":!0}),c(Jt)!=="svelte-17o3jn8"&&(Jt.innerHTML=ms),Vo=o(xe),u(de.$$.fragment,xe),Do=o(xe),u(le.$$.fragment,xe),xe.forEach(a),Eo=o(l),ce=s(l,"DIV",{class:!0});var Zn=v(ce);u(De.$$.fragment,Zn),Ro=o(Zn),Lt=s(Zn,"P",{"data-svelte-h":!0}),c(Lt)!=="svelte-1hrpjri"&&(Lt.innerHTML=us),Zn.forEach(a),Ao=o(l),pe=s(l,"DIV",{class:!0});var Vn=v(pe);u(Ee.$$.fragment,Vn),Ho=o(Vn),St=s(Vn,"P",{"data-svelte-h":!0}),c(St)!=="svelte-1wmjg8a"&&(St.innerHTML=hs),Vn.forEach(a),Go=o(l),V=s(l,"DIV",{class:!0});var xn=v(V);u(Re.$$.fragment,xn),Xo=o(xn),Zt=s(xn,"P",{"data-svelte-h":!0}),c(Zt)!=="svelte-1gbatu6"&&(Zt.textContent=fs),Yo=o(xn),Vt=s(xn,"P",{"data-svelte-h":!0}),c(Vt)!=="svelte-907bv"&&(Vt.innerHTML=gs),xn.forEach(a),Oo=o(l),C=s(l,"DIV",{class:!0});var R=v(C);u(Ae.$$.fragment,R),Qo=o(R),Dt=s(R,"P",{"data-svelte-h":!0}),c(Dt)!=="svelte-1n892mi"&&(Dt.textContent=_s),Ko=o(R),Et=s(R,"P",{"data-svelte-h":!0}),c(Et)!=="svelte-1xir9yc"&&(Et.innerHTML=ks),er=o(R),Rt=s(R,"P",{"data-svelte-h":!0}),c(Rt)!=="svelte-28v9x1"&&(Rt.innerHTML=bs),tr=o(R),u(me.$$.fragment,R),R.forEach(a),nr=o(l),ue=s(l,"DIV",{class:!0});var Dn=v(ue);u(He.$$.fragment,Dn),or=o(Dn),At=s(Dn,"P",{"data-svelte-h":!0}),c(At)!=="svelte-cwdwvn"&&(At.innerHTML=Ts),Dn.forEach(a),rr=o(l),he=s(l,"DIV",{class:!0});var En=v(he);u(Ge.$$.fragment,En),sr=o(En),Ht=s(En,"P",{"data-svelte-h":!0}),c(Ht)!=="svelte-dtuae6"&&(Ht.textContent=vs),En.forEach(a),ar=o(l),D=s(l,"DIV",{class:!0});var wn=v(D);u(Xe.$$.fragment,wn),ir=o(wn),Gt=s(wn,"P",{"data-svelte-h":!0}),c(Gt)!=="svelte-tpmkl3"&&(Gt.textContent=ys),dr=o(wn),u(fe.$$.fragment,wn),wn.forEach(a),lr=o(l),ge=s(l,"DIV",{class:!0});var Rn=v(ge);u(Ye.$$.fragment,Rn),cr=o(Rn),Xt=s(Rn,"P",{"data-svelte-h":!0}),c(Xt)!=="svelte-189h5u2"&&(Xt.innerHTML=xs),Rn.forEach(a),pr=o(l),_e=s(l,"DIV",{class:!0});var An=v(_e);u(Oe.$$.fragment,An),mr=o(An),Yt=s(An,"P",{"data-svelte-h":!0}),c(Yt)!=="svelte-zk9lwo"&&(Yt.textContent=ws),An.forEach(a),ur=o(l),U=s(l,"DIV",{class:!0});var we=v(U);u(Qe.$$.fragment,we),hr=o(we),Ot=s(we,"P",{"data-svelte-h":!0}),c(Ot)!=="svelte-u73u19"&&(Ot.textContent=zs),fr=o(we),Qt=s(we,"P",{"data-svelte-h":!0}),c(Qt)!=="svelte-p0dt88"&&(Qt.innerHTML=$s),gr=o(we),Kt=s(we,"P",{"data-svelte-h":!0}),c(Kt)!=="svelte-1qee1z"&&(Kt.innerHTML=Ps),we.forEach(a),_r=o(l),E=s(l,"DIV",{class:!0});var zn=v(E);u(Ke.$$.fragment,zn),kr=o(zn),en=s(zn,"P",{"data-svelte-h":!0}),c(en)!=="svelte-rx0wq1"&&(en.textContent=Bs),br=o(zn),tn=s(zn,"P",{"data-svelte-h":!0}),c(tn)!=="svelte-c9295b"&&(tn.innerHTML=Ms),zn.forEach(a),Tr=o(l),ke=s(l,"DIV",{class:!0});var Hn=v(ke);u(et.$$.fragment,Hn),vr=o(Hn),nn=s(Hn,"P",{"data-svelte-h":!0}),c(nn)!=="svelte-ikoqgw"&&(nn.innerHTML=qs),Hn.forEach(a),yr=o(l),be=s(l,"DIV",{class:!0});var Gn=v(be);u(tt.$$.fragment,Gn),xr=o(Gn),on=s(Gn,"P",{"data-svelte-h":!0}),c(on)!=="svelte-fkofn"&&(on.textContent=Is),Gn.forEach(a),l.forEach(a),qn=o(t),u(nt.$$.fragment,t),In=o(t),I=s(t,"DIV",{class:!0});var A=v(I);u(ot.$$.fragment,A),wr=o(A),rn=s(A,"P",{"data-svelte-h":!0}),c(rn)!=="svelte-1rb6jf9"&&(rn.innerHTML=Cs),zr=o(A),B=s(A,"DIV",{class:!0});var q=v(B);u(rt.$$.fragment,q),$r=o(q),sn=s(q,"P",{"data-svelte-h":!0}),c(sn)!=="svelte-1j8s0i5"&&(sn.textContent=Ws),Pr=o(q),an=s(q,"P",{"data-svelte-h":!0}),c(an)!=="svelte-1w3ayx9"&&(an.textContent=Ns),Br=o(q),dn=s(q,"P",{"data-svelte-h":!0}),c(dn)!=="svelte-mkudpf"&&(dn.innerHTML=Us),Mr=o(q),ln=s(q,"P",{"data-svelte-h":!0}),c(ln)!=="svelte-5hxtpc"&&(ln.innerHTML=js),qr=o(q),cn=s(q,"UL",{"data-svelte-h":!0}),c(cn)!=="svelte-1pes0uj"&&(cn.innerHTML=Fs),Ir=o(q),pn=s(q,"P",{"data-svelte-h":!0}),c(pn)!=="svelte-hs52sw"&&(pn.innerHTML=Js),Cr=o(q),u(Te.$$.fragment,q),q.forEach(a),Wr=o(A),W=s(A,"DIV",{class:!0});var H=v(W);u(st.$$.fragment,H),Nr=o(H),mn=s(H,"P",{"data-svelte-h":!0}),c(mn)!=="svelte-c27xjk"&&(mn.textContent=Ls),Ur=o(H),un=s(H,"P",{"data-svelte-h":!0}),c(un)!=="svelte-j0w5r1"&&(un.textContent=Ss),jr=o(H),hn=s(H,"P",{"data-svelte-h":!0}),c(hn)!=="svelte-mkudpf"&&(hn.innerHTML=Zs),Fr=o(H),u(ve.$$.fragment,H),H.forEach(a),Jr=o(A),ye=s(A,"DIV",{class:!0});var Xn=v(ye);u(at.$$.fragment,Xn),Lr=o(Xn),fn=s(Xn,"P",{"data-svelte-h":!0}),c(fn)!=="svelte-1em0285"&&(fn.innerHTML=Vs),Xn.forEach(a),A.forEach(a),Cn=o(t),u(it.$$.fragment,t),Wn=o(t),G=s(t,"DIV",{class:!0});var Yn=v(G);u(dt.$$.fragment,Yn),Sr=o(Yn),gn=s(Yn,"P",{"data-svelte-h":!0}),c(gn)!=="svelte-6nesdf"&&(gn.innerHTML=Ds),Yn.forEach(a),Nn=o(t),X=s(t,"DIV",{class:!0});var On=v(X);u(lt.$$.fragment,On),Zr=o(On),_n=s(On,"P",{"data-svelte-h":!0}),c(_n)!=="svelte-136gduh"&&(_n.textContent=Es),On.forEach(a),Un=o(t),Y=s(t,"DIV",{class:!0});var Qn=v(Y);u(ct.$$.fragment,Qn),Vr=o(Qn),kn=s(Qn,"P",{"data-svelte-h":!0}),c(kn)!=="svelte-18ocfae"&&(kn.textContent=Rs),Qn.forEach(a),jn=o(t),u(pt.$$.fragment,t),Fn=o(t),$n=s(t,"P",{}),v($n).forEach(a),this.h()},h(){y(i,"name","hf:doc:metadata"),y(i,"content",ia),y(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(_e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(d,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(t,b){e(document.head,i),w(t,z,b),w(t,k,b),w(t,T,b),h($,t,b),w(t,p,b),w(t,P,b),w(t,Pn,b),w(t,$e,b),w(t,Bn,b),h(Pe,t,b),w(t,Mn,b),w(t,d,b),h(Be,d,null),e(d,oo),e(d,_t),e(d,ro),e(d,kt),e(d,so),e(d,bt),e(d,ao),e(d,Tt),e(d,io),e(d,K),h(Me,K,null),e(K,lo),e(K,vt),e(d,co),e(d,ee),h(qe,ee,null),e(ee,po),e(ee,yt),e(d,mo),e(d,te),h(Ie,te,null),e(te,uo),e(te,xt),e(d,ho),e(d,ne),h(Ce,ne,null),e(ne,fo),e(ne,wt),e(d,go),e(d,j),h(We,j,null),e(j,_o),e(j,zt),e(j,ko),h(oe,j,null),e(d,bo),e(d,F),h(Ne,F,null),e(F,To),e(F,$t),e(F,vo),e(F,Pt),e(d,yo),e(d,re),h(Ue,re,null),e(re,xo),e(re,Bt),e(d,wo),e(d,se),h(je,se,null),e(se,zo),e(se,Mt),e(d,$o),e(d,J),h(Fe,J,null),e(J,Po),e(J,qt),e(J,Bo),e(J,It),e(d,Mo),e(d,L),h(Je,L,null),e(L,qo),e(L,Ct),e(L,Io),e(L,Wt),e(d,Co),e(d,S),h(Le,S,null),e(S,Wo),e(S,Nt),e(S,No),e(S,Ut),e(d,Uo),e(d,ae),h(Se,ae,null),e(ae,jo),e(ae,jt),e(d,Fo),e(d,Z),h(Ze,Z,null),e(Z,Jo),e(Z,Ft),e(Z,Lo),h(ie,Z,null),e(d,So),e(d,N),h(Ve,N,null),e(N,Zo),e(N,Jt),e(N,Vo),h(de,N,null),e(N,Do),h(le,N,null),e(d,Eo),e(d,ce),h(De,ce,null),e(ce,Ro),e(ce,Lt),e(d,Ao),e(d,pe),h(Ee,pe,null),e(pe,Ho),e(pe,St),e(d,Go),e(d,V),h(Re,V,null),e(V,Xo),e(V,Zt),e(V,Yo),e(V,Vt),e(d,Oo),e(d,C),h(Ae,C,null),e(C,Qo),e(C,Dt),e(C,Ko),e(C,Et),e(C,er),e(C,Rt),e(C,tr),h(me,C,null),e(d,nr),e(d,ue),h(He,ue,null),e(ue,or),e(ue,At),e(d,rr),e(d,he),h(Ge,he,null),e(he,sr),e(he,Ht),e(d,ar),e(d,D),h(Xe,D,null),e(D,ir),e(D,Gt),e(D,dr),h(fe,D,null),e(d,lr),e(d,ge),h(Ye,ge,null),e(ge,cr),e(ge,Xt),e(d,pr),e(d,_e),h(Oe,_e,null),e(_e,mr),e(_e,Yt),e(d,ur),e(d,U),h(Qe,U,null),e(U,hr),e(U,Ot),e(U,fr),e(U,Qt),e(U,gr),e(U,Kt),e(d,_r),e(d,E),h(Ke,E,null),e(E,kr),e(E,en),e(E,br),e(E,tn),e(d,Tr),e(d,ke),h(et,ke,null),e(ke,vr),e(ke,nn),e(d,yr),e(d,be),h(tt,be,null),e(be,xr),e(be,on),w(t,qn,b),h(nt,t,b),w(t,In,b),w(t,I,b),h(ot,I,null),e(I,wr),e(I,rn),e(I,zr),e(I,B),h(rt,B,null),e(B,$r),e(B,sn),e(B,Pr),e(B,an),e(B,Br),e(B,dn),e(B,Mr),e(B,ln),e(B,qr),e(B,cn),e(B,Ir),e(B,pn),e(B,Cr),h(Te,B,null),e(I,Wr),e(I,W),h(st,W,null),e(W,Nr),e(W,mn),e(W,Ur),e(W,un),e(W,jr),e(W,hn),e(W,Fr),h(ve,W,null),e(I,Jr),e(I,ye),h(at,ye,null),e(ye,Lr),e(ye,fn),w(t,Cn,b),h(it,t,b),w(t,Wn,b),w(t,G,b),h(dt,G,null),e(G,Sr),e(G,gn),w(t,Nn,b),w(t,X,b),h(lt,X,null),e(X,Zr),e(X,_n),w(t,Un,b),w(t,Y,b),h(ct,Y,null),e(Y,Vr),e(Y,kn),w(t,jn,b),h(pt,t,b),w(t,Fn,b),w(t,$n,b),Jn=!0},p(t,[b]){const l={};b&2&&(l.$$scope={dirty:b,ctx:t}),oe.$set(l);const mt={};b&2&&(mt.$$scope={dirty:b,ctx:t}),ie.$set(mt);const ut={};b&2&&(ut.$$scope={dirty:b,ctx:t}),de.$set(ut);const ht={};b&2&&(ht.$$scope={dirty:b,ctx:t}),le.$set(ht);const ft={};b&2&&(ft.$$scope={dirty:b,ctx:t}),me.$set(ft);const O={};b&2&&(O.$$scope={dirty:b,ctx:t}),fe.$set(O);const Q={};b&2&&(Q.$$scope={dirty:b,ctx:t}),Te.$set(Q);const gt={};b&2&&(gt.$$scope={dirty:b,ctx:t}),ve.$set(gt)},i(t){Jn||(f($.$$.fragment,t),f(Pe.$$.fragment,t),f(Be.$$.fragment,t),f(Me.$$.fragment,t),f(qe.$$.fragment,t),f(Ie.$$.fragment,t),f(Ce.$$.fragment,t),f(We.$$.fragment,t),f(oe.$$.fragment,t),f(Ne.$$.fragment,t),f(Ue.$$.fragment,t),f(je.$$.fragment,t),f(Fe.$$.fragment,t),f(Je.$$.fragment,t),f(Le.$$.fragment,t),f(Se.$$.fragment,t),f(Ze.$$.fragment,t),f(ie.$$.fragment,t),f(Ve.$$.fragment,t),f(de.$$.fragment,t),f(le.$$.fragment,t),f(De.$$.fragment,t),f(Ee.$$.fragment,t),f(Re.$$.fragment,t),f(Ae.$$.fragment,t),f(me.$$.fragment,t),f(He.$$.fragment,t),f(Ge.$$.fragment,t),f(Xe.$$.fragment,t),f(fe.$$.fragment,t),f(Ye.$$.fragment,t),f(Oe.$$.fragment,t),f(Qe.$$.fragment,t),f(Ke.$$.fragment,t),f(et.$$.fragment,t),f(tt.$$.fragment,t),f(nt.$$.fragment,t),f(ot.$$.fragment,t),f(rt.$$.fragment,t),f(Te.$$.fragment,t),f(st.$$.fragment,t),f(ve.$$.fragment,t),f(at.$$.fragment,t),f(it.$$.fragment,t),f(dt.$$.fragment,t),f(lt.$$.fragment,t),f(ct.$$.fragment,t),f(pt.$$.fragment,t),Jn=!0)},o(t){g($.$$.fragment,t),g(Pe.$$.fragment,t),g(Be.$$.fragment,t),g(Me.$$.fragment,t),g(qe.$$.fragment,t),g(Ie.$$.fragment,t),g(Ce.$$.fragment,t),g(We.$$.fragment,t),g(oe.$$.fragment,t),g(Ne.$$.fragment,t),g(Ue.$$.fragment,t),g(je.$$.fragment,t),g(Fe.$$.fragment,t),g(Je.$$.fragment,t),g(Le.$$.fragment,t),g(Se.$$.fragment,t),g(Ze.$$.fragment,t),g(ie.$$.fragment,t),g(Ve.$$.fragment,t),g(de.$$.fragment,t),g(le.$$.fragment,t),g(De.$$.fragment,t),g(Ee.$$.fragment,t),g(Re.$$.fragment,t),g(Ae.$$.fragment,t),g(me.$$.fragment,t),g(He.$$.fragment,t),g(Ge.$$.fragment,t),g(Xe.$$.fragment,t),g(fe.$$.fragment,t),g(Ye.$$.fragment,t),g(Oe.$$.fragment,t),g(Qe.$$.fragment,t),g(Ke.$$.fragment,t),g(et.$$.fragment,t),g(tt.$$.fragment,t),g(nt.$$.fragment,t),g(ot.$$.fragment,t),g(rt.$$.fragment,t),g(Te.$$.fragment,t),g(st.$$.fragment,t),g(ve.$$.fragment,t),g(at.$$.fragment,t),g(it.$$.fragment,t),g(dt.$$.fragment,t),g(lt.$$.fragment,t),g(ct.$$.fragment,t),g(pt.$$.fragment,t),Jn=!1},d(t){t&&(a(z),a(k),a(T),a(p),a(P),a(Pn),a($e),a(Bn),a(Mn),a(d),a(qn),a(In),a(I),a(Cn),a(Wn),a(G),a(Nn),a(X),a(Un),a(Y),a(jn),a(Fn),a($n)),a(i),_($,t),_(Pe,t),_(Be),_(Me),_(qe),_(Ie),_(Ce),_(We),_(oe),_(Ne),_(Ue),_(je),_(Fe),_(Je),_(Le),_(Se),_(Ze),_(ie),_(Ve),_(de),_(le),_(De),_(Ee),_(Re),_(Ae),_(me),_(He),_(Ge),_(Xe),_(fe),_(Ye),_(Oe),_(Qe),_(Ke),_(et),_(tt),_(nt,t),_(ot),_(rt),_(Te),_(st),_(ve),_(at),_(it,t),_(dt),_(lt),_(ct),_(pt,t)}}}const ia='{"title":"Utilities for Tokenizers","local":"utilities-for-tokenizers","sections":[{"title":"PreTrainedTokenizerBase","local":"transformers.PreTrainedTokenizerBase","sections":[],"depth":2},{"title":"SpecialTokensMixin","local":"transformers.SpecialTokensMixin","sections":[],"depth":2},{"title":"Enums and namedtuples","local":"transformers.tokenization_utils_base.TruncationStrategy","sections":[],"depth":2}],"depth":1}';function da(M){return Hs(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ga extends Gs{constructor(i){super(),Xs(this,i,da,aa,As,{})}}export{ga as component};
