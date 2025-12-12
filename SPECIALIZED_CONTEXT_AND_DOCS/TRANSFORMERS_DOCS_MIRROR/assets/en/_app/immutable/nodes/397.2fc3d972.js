import{s as ea,o as ta,n as Y}from"../chunks/scheduler.18a86fab.js";import{S as oa,i as na,g as r,s as o,r as m,A as sa,h as a,f as c,c as n,j as x,x as p,u,k,y as t,a as h,v as f,d as g,t as _,w as T}from"../chunks/index.98837b22.js";import{T as qo}from"../chunks/Tip.77304350.js";import{D as w}from"../chunks/Docstring.a1ef7999.js";import{C as No}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as jo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as X,E as ra}from"../chunks/getInferenceSnippets.06c2775f.js";function aa(S){let s,b="Example:",l,v,y;return v=new No({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFNwZWVjaFQ1TW9kZWwlMkMlMjBTcGVlY2hUNUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjAlMjJtaWNyb3NvZnQlMkZzcGVlY2h0NV9hc3IlMjIlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwU3BlZWNoVDVDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwJTIybWljcm9zb2Z0JTJGc3BlZWNodDVfYXNyJTIyJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBTcGVlY2hUNU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> SpeechT5Model, SpeechT5Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a &quot;microsoft/speecht5_asr&quot; style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = SpeechT5Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the &quot;microsoft/speecht5_asr&quot; style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SpeechT5Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){s=r("p"),s.textContent=b,l=o(),m(v.$$.fragment)},l(d){s=a(d,"P",{"data-svelte-h":!0}),p(s)!=="svelte-11lpom8"&&(s.textContent=b),l=n(d),u(v.$$.fragment,d)},m(d,$){h(d,s,$),h(d,l,$),f(v,d,$),y=!0},p:Y,i(d){y||(g(v.$$.fragment,d),y=!0)},o(d){_(v.$$.fragment,d),y=!1},d(d){d&&(c(s),c(l)),T(v,d)}}}function ca(S){let s,b="Example:",l,v,y;return v=new No({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFNwZWVjaFQ1SGlmaUdhbiUyQyUyMFNwZWVjaFQ1SGlmaUdhbkNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjAlMjJtaWNyb3NvZnQlMkZzcGVlY2h0NV9oaWZpZ2FuJTIyJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFNwZWVjaFQ1SGlmaUdhbkNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjAlMjJtaWNyb3NvZnQlMkZzcGVlY2h0NV9oaWZpZ2FuJTIyJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBTcGVlY2hUNUhpZmlHYW4oY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> SpeechT5HifiGan, SpeechT5HifiGanConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a &quot;microsoft/speecht5_hifigan&quot; style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = SpeechT5HifiGanConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the &quot;microsoft/speecht5_hifigan&quot; style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SpeechT5HifiGan(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){s=r("p"),s.textContent=b,l=o(),m(v.$$.fragment)},l(d){s=a(d,"P",{"data-svelte-h":!0}),p(s)!=="svelte-11lpom8"&&(s.textContent=b),l=n(d),u(v.$$.fragment,d)},m(d,$){h(d,s,$),h(d,l,$),f(v,d,$),y=!0},p:Y,i(d){y||(g(v.$$.fragment,d),y=!0)},o(d){_(v.$$.fragment,d),y=!1},d(d){d&&(c(s),c(l)),T(v,d)}}}function ia(S){let s,b=`This class method is simply calling the feature extractor
<a href="/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained">from_pretrained()</a>, image processor
<a href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin">ImageProcessingMixin</a> and the tokenizer
<code>~tokenization_utils_base.PreTrainedTokenizer.from_pretrained</code> methods. Please refer to the docstrings of the
methods above for more information.`;return{c(){s=r("p"),s.innerHTML=b},l(l){s=a(l,"P",{"data-svelte-h":!0}),p(s)!=="svelte-vj9ud3"&&(s.innerHTML=b)},m(l,v){h(l,s,v)},p:Y,d(l){l&&c(s)}}}function da(S){let s,b=`This class method is simply calling <a href="/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained">save_pretrained()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained">save_pretrained()</a>. Please refer to the docstrings of the
methods above for more information.`;return{c(){s=r("p"),s.innerHTML=b},l(l){s=a(l,"P",{"data-svelte-h":!0}),p(s)!=="svelte-1euzcqa"&&(s.innerHTML=b)},m(l,v){h(l,s,v)},p:Y,d(l){l&&c(s)}}}function la(S){let s,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=r("p"),s.innerHTML=b},l(l){s=a(l,"P",{"data-svelte-h":!0}),p(s)!=="svelte-fincs2"&&(s.innerHTML=b)},m(l,v){h(l,s,v)},p:Y,d(l){l&&c(s)}}}function pa(S){let s,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=r("p"),s.innerHTML=b},l(l){s=a(l,"P",{"data-svelte-h":!0}),p(s)!=="svelte-fincs2"&&(s.innerHTML=b)},m(l,v){h(l,s,v)},p:Y,d(l){l&&c(s)}}}function ha(S){let s,b="Example:",l,v,y;return v=new No({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFNwZWVjaFQ1UHJvY2Vzc29yJTJDJTIwU3BlZWNoVDVGb3JTcGVlY2hUb1RleHQlMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMEElMEFkYXRhc2V0JTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUwQSUyMCUyMCUyMCUyMCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZsaWJyaXNwZWVjaF9hc3JfZGVtbyUyMiUyQyUyMCUyMmNsZWFuJTIyJTJDJTIwc3BsaXQlM0QlMjJ2YWxpZGF0aW9uJTIyJTBBKSUyMCUyMCUyMyUyMGRvY3Rlc3QlM0ElMjAlMkJJR05PUkVfUkVTVUxUJTBBZGF0YXNldCUyMCUzRCUyMGRhdGFzZXQuc29ydCglMjJpZCUyMiklMEFzYW1wbGluZ19yYXRlJTIwJTNEJTIwZGF0YXNldC5mZWF0dXJlcyU1QiUyMmF1ZGlvJTIyJTVELnNhbXBsaW5nX3JhdGUlMEElMEFwcm9jZXNzb3IlMjAlM0QlMjBTcGVlY2hUNVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGc3BlZWNodDVfYXNyJTIyKSUwQW1vZGVsJTIwJTNEJTIwU3BlZWNoVDVGb3JTcGVlY2hUb1RleHQuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRnNwZWVjaHQ1X2FzciUyMiklMEElMEElMjMlMjBhdWRpbyUyMGZpbGUlMjBpcyUyMGRlY29kZWQlMjBvbiUyMHRoZSUyMGZseSUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3NvcihhdWRpbyUzRGRhdGFzZXQlNUIwJTVEJTVCJTIyYXVkaW8lMjIlNUQlNUIlMjJhcnJheSUyMiU1RCUyQyUyMHNhbXBsaW5nX3JhdGUlM0RzYW1wbGluZ19yYXRlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFwcmVkaWN0ZWRfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbGVuZ3RoJTNEMTAwKSUwQSUwQSUyMyUyMHRyYW5zY3JpYmUlMjBzcGVlY2glMEF0cmFuc2NyaXB0aW9uJTIwJTNEJTIwcHJvY2Vzc29yLmJhdGNoX2RlY29kZShwcmVkaWN0ZWRfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBdHJhbnNjcmlwdGlvbiU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> SpeechT5Processor, SpeechT5ForSpeechToText
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># doctest: +IGNORE_RESULT</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = SpeechT5Processor.from_pretrained(<span class="hljs-string">&quot;microsoft/speecht5_asr&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SpeechT5ForSpeechToText.from_pretrained(<span class="hljs-string">&quot;microsoft/speecht5_asr&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(audio=dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=sampling_rate, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_ids = model.generate(**inputs, max_length=<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># transcribe speech</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.batch_decode(predicted_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription[<span class="hljs-number">0</span>]
<span class="hljs-string">&#x27;mister quilter is the apostle of the middle classes and we are glad to welcome his gospel&#x27;</span>`,wrap:!1}}),{c(){s=r("p"),s.textContent=b,l=o(),m(v.$$.fragment)},l(d){s=a(d,"P",{"data-svelte-h":!0}),p(s)!=="svelte-11lpom8"&&(s.textContent=b),l=n(d),u(v.$$.fragment,d)},m(d,$){h(d,s,$),h(d,l,$),f(v,d,$),y=!0},p:Y,i(d){y||(g(v.$$.fragment,d),y=!0)},o(d){_(v.$$.fragment,d),y=!1},d(d){d&&(c(s),c(l)),T(v,d)}}}function ma(S){let s,b;return s=new No({props:{code:"aW5wdXRzJTVCJTIybGFiZWxzJTIyJTVEJTIwJTNEJTIwcHJvY2Vzc29yKHRleHRfdGFyZ2V0JTNEZGF0YXNldCU1QjAlNUQlNUIlMjJ0ZXh0JTIyJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikuaW5wdXRfaWRzJTBBJTBBJTIzJTIwY29tcHV0ZSUyMGxvc3MlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>inputs[<span class="hljs-string">&quot;labels&quot;</span>] = processor(text_target=dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;text&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute loss</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">19.68</span>`,wrap:!1}}),{c(){m(s.$$.fragment)},l(l){u(s.$$.fragment,l)},m(l,v){f(s,l,v),b=!0},p:Y,i(l){b||(g(s.$$.fragment,l),b=!0)},o(l){_(s.$$.fragment,l),b=!1},d(l){T(s,l)}}}function ua(S){let s,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=r("p"),s.innerHTML=b},l(l){s=a(l,"P",{"data-svelte-h":!0}),p(s)!=="svelte-fincs2"&&(s.innerHTML=b)},m(l,v){h(l,s,v)},p:Y,d(l){l&&c(s)}}}function fa(S){let s,b="Example:",l,v,y;return v=new No({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFNwZWVjaFQ1UHJvY2Vzc29yJTJDJTIwU3BlZWNoVDVGb3JUZXh0VG9TcGVlY2glMkMlMjBTcGVlY2hUNUhpZmlHYW4lMkMlMjBzZXRfc2VlZCUwQWltcG9ydCUyMHRvcmNoJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwU3BlZWNoVDVQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRnNwZWVjaHQ1X3R0cyUyMiklMEFtb2RlbCUyMCUzRCUyMFNwZWVjaFQ1Rm9yVGV4dFRvU3BlZWNoLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZzcGVlY2h0NV90dHMlMjIpJTBBdm9jb2RlciUyMCUzRCUyMFNwZWVjaFQ1SGlmaUdhbi5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGc3BlZWNodDVfaGlmaWdhbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IodGV4dCUzRCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXNwZWFrZXJfZW1iZWRkaW5ncyUyMCUzRCUyMHRvcmNoLnplcm9zKCgxJTJDJTIwNTEyKSklMjAlMjAlMjMlMjBvciUyMGxvYWQlMjB4dmVjdG9ycyUyMGZyb20lMjBhJTIwZmlsZSUwQSUwQXNldF9zZWVkKDU1NSklMjAlMjAlMjMlMjBtYWtlJTIwZGV0ZXJtaW5pc3RpYyUwQSUwQSUyMyUyMGdlbmVyYXRlJTIwc3BlZWNoJTBBc3BlZWNoJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTJDJTIwc3BlYWtlcl9lbWJlZGRpbmdzJTNEc3BlYWtlcl9lbWJlZGRpbmdzJTJDJTIwdm9jb2RlciUzRHZvY29kZXIpJTBBc3BlZWNoLnNoYXBl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = SpeechT5Processor.from_pretrained(<span class="hljs-string">&quot;microsoft/speecht5_tts&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SpeechT5ForTextToSpeech.from_pretrained(<span class="hljs-string">&quot;microsoft/speecht5_tts&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>vocoder = SpeechT5HifiGan.from_pretrained(<span class="hljs-string">&quot;microsoft/speecht5_hifigan&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(text=<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>speaker_embeddings = torch.zeros((<span class="hljs-number">1</span>, <span class="hljs-number">512</span>))  <span class="hljs-comment"># or load xvectors from a file</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>set_seed(<span class="hljs-number">555</span>)  <span class="hljs-comment"># make deterministic</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># generate speech</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>speech = model.generate(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>], speaker_embeddings=speaker_embeddings, vocoder=vocoder)
<span class="hljs-meta">&gt;&gt;&gt; </span>speech.shape
torch.Size([<span class="hljs-number">15872</span>])`,wrap:!1}}),{c(){s=r("p"),s.textContent=b,l=o(),m(v.$$.fragment)},l(d){s=a(d,"P",{"data-svelte-h":!0}),p(s)!=="svelte-11lpom8"&&(s.textContent=b),l=n(d),u(v.$$.fragment,d)},m(d,$){h(d,s,$),h(d,l,$),f(v,d,$),y=!0},p:Y,i(d){y||(g(v.$$.fragment,d),y=!0)},o(d){_(v.$$.fragment,d),y=!1},d(d){d&&(c(s),c(l)),T(v,d)}}}function ga(S){let s,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=r("p"),s.innerHTML=b},l(l){s=a(l,"P",{"data-svelte-h":!0}),p(s)!=="svelte-fincs2"&&(s.innerHTML=b)},m(l,v){h(l,s,v)},p:Y,d(l){l&&c(s)}}}function _a(S){let s,b="Example:",l,v,y;return v=new No({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFNwZWVjaFQ1UHJvY2Vzc29yJTJDJTIwU3BlZWNoVDVGb3JTcGVlY2hUb1NwZWVjaCUyQyUyMFNwZWVjaFQ1SGlmaUdhbiUyQyUyMHNldF9zZWVkJTBBZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTBBaW1wb3J0JTIwdG9yY2glMEElMEFkYXRhc2V0JTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUwQSUyMCUyMCUyMCUyMCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZsaWJyaXNwZWVjaF9hc3JfZGVtbyUyMiUyQyUyMCUyMmNsZWFuJTIyJTJDJTIwc3BsaXQlM0QlMjJ2YWxpZGF0aW9uJTIyJTBBKSUyMCUyMCUyMyUyMGRvY3Rlc3QlM0ElMjAlMkJJR05PUkVfUkVTVUxUJTBBZGF0YXNldCUyMCUzRCUyMGRhdGFzZXQuc29ydCglMjJpZCUyMiklMEFzYW1wbGluZ19yYXRlJTIwJTNEJTIwZGF0YXNldC5mZWF0dXJlcyU1QiUyMmF1ZGlvJTIyJTVELnNhbXBsaW5nX3JhdGUlMEElMEFwcm9jZXNzb3IlMjAlM0QlMjBTcGVlY2hUNVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGc3BlZWNodDVfdmMlMjIpJTBBbW9kZWwlMjAlM0QlMjBTcGVlY2hUNUZvclNwZWVjaFRvU3BlZWNoLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZzcGVlY2h0NV92YyUyMiklMEF2b2NvZGVyJTIwJTNEJTIwU3BlZWNoVDVIaWZpR2FuLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZzcGVlY2h0NV9oaWZpZ2FuJTIyKSUwQSUwQSUyMyUyMGF1ZGlvJTIwZmlsZSUyMGlzJTIwZGVjb2RlZCUyMG9uJTIwdGhlJTIwZmx5JTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKGF1ZGlvJTNEZGF0YXNldCU1QjAlNUQlNUIlMjJhdWRpbyUyMiU1RCU1QiUyMmFycmF5JTIyJTVEJTJDJTIwc2FtcGxpbmdfcmF0ZSUzRHNhbXBsaW5nX3JhdGUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXNwZWFrZXJfZW1iZWRkaW5ncyUyMCUzRCUyMHRvcmNoLnplcm9zKCgxJTJDJTIwNTEyKSklMjAlMjAlMjMlMjBvciUyMGxvYWQlMjB4dmVjdG9ycyUyMGZyb20lMjBhJTIwZmlsZSUwQSUwQXNldF9zZWVkKDU1NSklMjAlMjAlMjMlMjBtYWtlJTIwZGV0ZXJtaW5pc3RpYyUwQSUwQSUyMyUyMGdlbmVyYXRlJTIwc3BlZWNoJTBBc3BlZWNoJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGVfc3BlZWNoKGlucHV0cyU1QiUyMmlucHV0X3ZhbHVlcyUyMiU1RCUyQyUyMHNwZWFrZXJfZW1iZWRkaW5ncyUyQyUyMHZvY29kZXIlM0R2b2NvZGVyKSUwQXNwZWVjaC5zaGFwZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># doctest: +IGNORE_RESULT</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = SpeechT5Processor.from_pretrained(<span class="hljs-string">&quot;microsoft/speecht5_vc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SpeechT5ForSpeechToSpeech.from_pretrained(<span class="hljs-string">&quot;microsoft/speecht5_vc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>vocoder = SpeechT5HifiGan.from_pretrained(<span class="hljs-string">&quot;microsoft/speecht5_hifigan&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(audio=dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=sampling_rate, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>speaker_embeddings = torch.zeros((<span class="hljs-number">1</span>, <span class="hljs-number">512</span>))  <span class="hljs-comment"># or load xvectors from a file</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>set_seed(<span class="hljs-number">555</span>)  <span class="hljs-comment"># make deterministic</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># generate speech</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>speech = model.generate_speech(inputs[<span class="hljs-string">&quot;input_values&quot;</span>], speaker_embeddings, vocoder=vocoder)
<span class="hljs-meta">&gt;&gt;&gt; </span>speech.shape
torch.Size([<span class="hljs-number">77824</span>])`,wrap:!1}}),{c(){s=r("p"),s.textContent=b,l=o(),m(v.$$.fragment)},l(d){s=a(d,"P",{"data-svelte-h":!0}),p(s)!=="svelte-11lpom8"&&(s.textContent=b),l=n(d),u(v.$$.fragment,d)},m(d,$){h(d,s,$),h(d,l,$),f(v,d,$),y=!0},p:Y,i(d){y||(g(v.$$.fragment,d),y=!0)},o(d){_(v.$$.fragment,d),y=!1},d(d){d&&(c(s),c(l)),T(v,d)}}}function Ta(S){let s,b,l,v,y,d="<em>This model was released on 2021-10-14 and added to Hugging Face Transformers on 2023-02-03.</em>",$,je,Io,le,Os='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Po,Ne,Wo,Ue,Xs='The SpeechT5 model was proposed in <a href="https://huggingface.co/papers/2110.07205" rel="nofollow">SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing</a> by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.',Zo,Je,Ys="The abstract from the paper is the following:",Go,Ie,Qs="<em>Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder. Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder. Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.</em>",Ho,Pe,As='This model was contributed by <a href="https://huggingface.co/Matthijs" rel="nofollow">Matthijs</a>. The original code can be found <a href="https://github.com/microsoft/SpeechT5" rel="nofollow">here</a>.',Lo,We,Eo,E,Ze,_n,St,Ks=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Model">SpeechT5Model</a>. It is used to instantiate a
SpeechT5 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the SpeechT5
<a href="https://huggingface.co/microsoft/speecht5_asr" rel="nofollow">microsoft/speecht5_asr</a> architecture.`,Tn,$t,er=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,vn,pe,Vo,Ge,Do,V,He,bn,Mt,tr=`This is the configuration class to store the configuration of a <code>SpeechT5HifiGanModel</code>. It is used to instantiate
a SpeechT5 HiFi-GAN vocoder model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the SpeechT5
<a href="https://huggingface.co/microsoft/speecht5_hifigan" rel="nofollow">microsoft/speecht5_hifigan</a> architecture.`,yn,Ft,or=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,kn,he,Bo,Le,Ro,z,Ee,xn,zt,nr='Construct a SpeechT5 tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',wn,Ct,sr=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,Sn,me,Ve,$n,qt,rr=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`,Mn,jt,De,Fn,K,Be,zn,Nt,ar=`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`,Cn,Ut,cr="Similar to doing <code>self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))</code>.",qn,ue,Re,jn,Jt,ir="Convert a list of lists of token ids into a list of strings by calling decode.",Oo,Oe,Xo,q,Xe,Nn,It,dr="Constructs a SpeechT5 feature extractor.",Un,Pt,lr=`This class can pre-process a raw speech signal by (optionally) normalizing to zero-mean unit-variance, for use by
the SpeechT5 speech encoder prenet.`,Jn,Wt,pr=`This class can also extract log-mel filter bank features from raw speech, for use by the SpeechT5 speech decoder
prenet.`,In,Zt,hr=`This feature extractor inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor">SequenceFeatureExtractor</a> which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.`,Pn,ee,Ye,Wn,Gt,mr="Main method to featurize and prepare for the model one or several sequence(s).",Zn,Ht,ur=`Pass in a value for <code>audio</code> to extract waveform features. Pass in a value for <code>audio_target</code> to extract log-mel
spectrogram features.`,Yo,Qe,Qo,M,Ae,Gn,Lt,fr="Constructs a SpeechT5 processor which wraps a feature extractor and a tokenizer into a single processor.",Hn,Et,gr=`<a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor">SpeechT5Processor</a> offers all the functionalities of <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor">SpeechT5FeatureExtractor</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer">SpeechT5Tokenizer</a>. See
the docstring of <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__"><strong>call</strong>()</a> and <a href="/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode">decode()</a> for more information.`,Ln,C,Ke,En,Vt,_r="Processes audio and text input, as well as audio and text targets.",Vn,Dt,Tr=`You can process audio by using the argument <code>audio</code>, or process audio targets by using the argument
<code>audio_target</code>. This forwards the arguments to SpeechT5FeatureExtractor’s
<a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor.__call__"><strong>call</strong>()</a>.`,Dn,Bt,vr=`You can process text by using the argument <code>text</code>, or process text labels by using the argument <code>text_target</code>.
This forwards the arguments to SpeechT5Tokenizer’s <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__"><strong>call</strong>()</a>.`,Bn,Rt,br="Valid input combinations are:",Rn,Ot,yr="<li><code>text</code> only</li> <li><code>audio</code> only</li> <li><code>text_target</code> only</li> <li><code>audio_target</code> only</li> <li><code>text</code> and <code>audio_target</code></li> <li><code>audio</code> and <code>audio_target</code></li> <li><code>text</code> and <code>text_target</code></li> <li><code>audio</code> and <code>text_target</code></li>",On,Xt,kr="Please refer to the docstring of the above two methods for more information.",Xn,I,et,Yn,Yt,xr="Collates the audio and text inputs, as well as their targets, into a padded batch.",Qn,Qt,wr=`Audio inputs are padded by SpeechT5FeatureExtractor’s <a href="/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad">pad()</a>. Text inputs are padded
by SpeechT5Tokenizer’s <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad">pad()</a>.`,An,At,Sr="Valid input combinations are:",Kn,Kt,$r="<li><code>input_ids</code> only</li> <li><code>input_values</code> only</li> <li><code>labels</code> only, either log-mel spectrograms or text tokens</li> <li><code>input_ids</code> and log-mel spectrogram <code>labels</code></li> <li><code>input_values</code> and text <code>labels</code></li>",es,eo,Mr="Please refer to the docstring of the above two methods for more information.",ts,te,tt,os,to,Fr="Instantiate a processor associated with a pretrained model.",ns,fe,ss,oe,ot,rs,oo,zr=`Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it
can be reloaded using the <a href="/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained">from_pretrained()</a> method.`,as,ge,cs,_e,nt,is,no,Cr=`This method forwards all its arguments to PreTrainedTokenizer’s <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode">batch_decode()</a>. Please
refer to the docstring of this method for more information.`,ds,Te,st,ls,so,qr=`This method forwards all its arguments to PreTrainedTokenizer’s <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode">decode()</a>. Please refer to
the docstring of this method for more information.`,Ao,rt,Ko,P,at,ps,ro,jr="The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.",hs,ao,Nr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ms,co,Ur=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,us,ne,ct,fs,io,Jr='The <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Model">SpeechT5Model</a> forward method, overrides the <code>__call__</code> special method.',gs,ve,en,it,tn,W,dt,_s,lo,Ir="SpeechT5 Model with a speech encoder and a text decoder.",Ts,po,Pr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vs,ho,Wr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,bs,L,lt,ys,mo,Zr='The <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToText">SpeechT5ForSpeechToText</a> forward method, overrides the <code>__call__</code> special method.',ks,be,xs,ye,ws,ke,on,pt,nn,j,ht,Ss,uo,Gr="SpeechT5 Model with a text encoder and a speech decoder.",$s,fo,Hr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ms,go,Lr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fs,D,mt,zs,_o,Er='The <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5ForTextToSpeech">SpeechT5ForTextToSpeech</a> forward method, overrides the <code>__call__</code> special method.',Cs,xe,qs,we,js,Se,ut,Ns,To,Vr=`Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a
speech waveform using a vocoder.`,sn,ft,rn,N,gt,Us,vo,Dr="SpeechT5 Model with a speech encoder and a speech decoder.",Js,bo,Br=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Is,yo,Rr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ps,B,_t,Ws,ko,Or='The <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToSpeech">SpeechT5ForSpeechToSpeech</a> forward method, overrides the <code>__call__</code> special method.',Zs,$e,Gs,Me,Hs,Fe,Tt,Ls,xo,Xr=`Converts a raw speech waveform into a sequence of mel spectrograms, which are subsequently turned back into a
speech waveform using a vocoder.`,an,vt,cn,Z,bt,Es,wo,Yr="HiFi-GAN vocoder.",Vs,So,Qr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ds,$o,Ar=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bs,ze,yt,Rs,Mo,Kr=`Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
waveform.`,dn,kt,ln,Uo,pn;return je=new X({props:{title:"SpeechT5",local:"speecht5",headingTag:"h1"}}),Ne=new X({props:{title:"Overview",local:"overview",headingTag:"h2"}}),We=new X({props:{title:"SpeechT5Config",local:"transformers.SpeechT5Config",headingTag:"h2"}}),Ze=new w({props:{name:"class transformers.SpeechT5Config",anchor:"transformers.SpeechT5Config",parameters:[{name:"vocab_size",val:" = 81"},{name:"hidden_size",val:" = 768"},{name:"encoder_layers",val:" = 12"},{name:"encoder_attention_heads",val:" = 12"},{name:"encoder_ffn_dim",val:" = 3072"},{name:"encoder_layerdrop",val:" = 0.1"},{name:"decoder_layers",val:" = 6"},{name:"decoder_ffn_dim",val:" = 3072"},{name:"decoder_attention_heads",val:" = 12"},{name:"decoder_layerdrop",val:" = 0.1"},{name:"hidden_act",val:" = 'gelu'"},{name:"positional_dropout",val:" = 0.1"},{name:"hidden_dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.1"},{name:"activation_dropout",val:" = 0.1"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"scale_embedding",val:" = False"},{name:"feat_extract_norm",val:" = 'group'"},{name:"feat_proj_dropout",val:" = 0.0"},{name:"feat_extract_activation",val:" = 'gelu'"},{name:"conv_dim",val:" = (512, 512, 512, 512, 512, 512, 512)"},{name:"conv_stride",val:" = (5, 2, 2, 2, 2, 2, 2)"},{name:"conv_kernel",val:" = (10, 3, 3, 3, 3, 2, 2)"},{name:"conv_bias",val:" = False"},{name:"num_conv_pos_embeddings",val:" = 128"},{name:"num_conv_pos_embedding_groups",val:" = 16"},{name:"apply_spec_augment",val:" = True"},{name:"mask_time_prob",val:" = 0.05"},{name:"mask_time_length",val:" = 10"},{name:"mask_time_min_masks",val:" = 2"},{name:"mask_feature_prob",val:" = 0.0"},{name:"mask_feature_length",val:" = 10"},{name:"mask_feature_min_masks",val:" = 0"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"decoder_start_token_id",val:" = 2"},{name:"num_mel_bins",val:" = 80"},{name:"speech_decoder_prenet_layers",val:" = 2"},{name:"speech_decoder_prenet_units",val:" = 256"},{name:"speech_decoder_prenet_dropout",val:" = 0.5"},{name:"speaker_embedding_dim",val:" = 512"},{name:"speech_decoder_postnet_layers",val:" = 5"},{name:"speech_decoder_postnet_units",val:" = 256"},{name:"speech_decoder_postnet_kernel",val:" = 5"},{name:"speech_decoder_postnet_dropout",val:" = 0.5"},{name:"reduction_factor",val:" = 2"},{name:"max_speech_positions",val:" = 4000"},{name:"max_text_positions",val:" = 450"},{name:"encoder_max_relative_position",val:" = 160"},{name:"use_guided_attention_loss",val:" = True"},{name:"guided_attention_loss_num_heads",val:" = 2"},{name:"guided_attention_loss_sigma",val:" = 0.4"},{name:"guided_attention_loss_scale",val:" = 10.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 81) &#x2014;
Vocabulary size of the SpeechT5 model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed to the forward method of <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Model">SpeechT5Model</a>.`,name:"vocab_size"},{anchor:"transformers.SpeechT5Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.SpeechT5Config.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"encoder_layers"},{anchor:"transformers.SpeechT5Config.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.SpeechT5Config.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.SpeechT5Config.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.SpeechT5Config.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"decoder_layers"},{anchor:"transformers.SpeechT5Config.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.SpeechT5Config.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.SpeechT5Config.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.SpeechT5Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.SpeechT5Config.positional_dropout",description:`<strong>positional_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the text position encoding layers.`,name:"positional_dropout"},{anchor:"transformers.SpeechT5Config.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout"},{anchor:"transformers.SpeechT5Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.SpeechT5Config.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.SpeechT5Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.SpeechT5Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.SpeechT5Config.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.SpeechT5Config.feat_extract_norm",description:`<strong>feat_extract_norm</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;group&quot;</code>) &#x2014;
The norm to be applied to 1D convolutional layers in the speech encoder pre-net. One of <code>&quot;group&quot;</code> for group
normalization of only the first 1D convolutional layer or <code>&quot;layer&quot;</code> for layer normalization of all 1D
convolutional layers.`,name:"feat_extract_norm"},{anchor:"transformers.SpeechT5Config.feat_proj_dropout",description:`<strong>feat_proj_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for output of the speech encoder pre-net.`,name:"feat_proj_dropout"},{anchor:"transformers.SpeechT5Config.feat_extract_activation",description:"<strong>feat_extract_activation</strong> (<code>str, </code>optional<code>, defaults to </code>&#x201C;gelu&#x201D;<code>) -- The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string, </code>&#x201C;gelu&#x201D;<code>, </code>&#x201C;relu&#x201D;<code>, </code>&#x201C;selu&#x201D;<code>and</code>&#x201C;gelu_new&#x201D;` are supported.",name:"feat_extract_activation"},{anchor:"transformers.SpeechT5Config.conv_dim",description:`<strong>conv_dim</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(512, 512, 512, 512, 512, 512, 512)</code>) &#x2014;
A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
speech encoder pre-net. The length of <em>conv_dim</em> defines the number of 1D convolutional layers.`,name:"conv_dim"},{anchor:"transformers.SpeechT5Config.conv_stride",description:`<strong>conv_stride</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(5, 2, 2, 2, 2, 2, 2)</code>) &#x2014;
A tuple of integers defining the stride of each 1D convolutional layer in the speech encoder pre-net. The
length of <em>conv_stride</em> defines the number of convolutional layers and has to match the length of
<em>conv_dim</em>.`,name:"conv_stride"},{anchor:"transformers.SpeechT5Config.conv_kernel",description:`<strong>conv_kernel</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(10, 3, 3, 3, 3, 3, 3)</code>) &#x2014;
A tuple of integers defining the kernel size of each 1D convolutional layer in the speech encoder pre-net.
The length of <em>conv_kernel</em> defines the number of convolutional layers and has to match the length of
<em>conv_dim</em>.`,name:"conv_kernel"},{anchor:"transformers.SpeechT5Config.conv_bias",description:`<strong>conv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the 1D convolutional layers have a bias.`,name:"conv_bias"},{anchor:"transformers.SpeechT5Config.num_conv_pos_embeddings",description:`<strong>num_conv_pos_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
embeddings layer.`,name:"num_conv_pos_embeddings"},{anchor:"transformers.SpeechT5Config.num_conv_pos_embedding_groups",description:`<strong>num_conv_pos_embedding_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of groups of 1D convolutional positional embeddings layer.`,name:"num_conv_pos_embedding_groups"},{anchor:"transformers.SpeechT5Config.apply_spec_augment",description:`<strong>apply_spec_augment</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to apply <em>SpecAugment</em> data augmentation to the outputs of the speech encoder pre-net. For
reference see <a href="https://huggingface.co/papers/1904.08779" rel="nofollow">SpecAugment: A Simple Data Augmentation Method for Automatic Speech
Recognition</a>.`,name:"apply_spec_augment"},{anchor:"transformers.SpeechT5Config.mask_time_prob",description:`<strong>mask_time_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.05) &#x2014;
Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
procedure generates &#x201D;mask_time_prob<em>len(time_axis)/mask_time_length&#x201D; independent masks over the axis. If
reasoning from the probability of each feature vector to be chosen as the start of the vector span to be
masked, </em>mask_time_prob<em> should be \`prob_vector_start</em>mask_time_length<code>. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if </code>apply_spec_augment is True\`.`,name:"mask_time_prob"},{anchor:"transformers.SpeechT5Config.mask_time_length",description:`<strong>mask_time_length</strong> (<code>int</code>, <em>optional</em>, defaults to 10) &#x2014;
Length of vector span along the time axis.`,name:"mask_time_length"},{anchor:"transformers.SpeechT5Config.mask_time_min_masks",description:`<strong>mask_time_min_masks</strong> (<code>int</code>, <em>optional</em>, defaults to 2), &#x2014;
The minimum number of masks of length <code>mask_feature_length</code> generated along the time axis, each time step,
irrespectively of <code>mask_feature_prob</code>. Only relevant if &#x201D;mask_time_prob*len(time_axis)/mask_time_length &lt;
mask_time_min_masks&#x201D;`,name:"mask_time_min_masks"},{anchor:"transformers.SpeechT5Config.mask_feature_prob",description:`<strong>mask_feature_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
masking procedure generates &#x201D;mask_feature_prob<em>len(feature_axis)/mask_time_length&#x201D; independent masks over
the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
span to be masked, </em>mask_feature_prob<em> should be \`prob_vector_start</em>mask_feature_length<code>. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if </code>apply_spec_augment is
True\`.`,name:"mask_feature_prob"},{anchor:"transformers.SpeechT5Config.mask_feature_length",description:`<strong>mask_feature_length</strong> (<code>int</code>, <em>optional</em>, defaults to 10) &#x2014;
Length of vector span along the feature axis.`,name:"mask_feature_length"},{anchor:"transformers.SpeechT5Config.mask_feature_min_masks",description:`<strong>mask_feature_min_masks</strong> (<code>int</code>, <em>optional</em>, defaults to 0), &#x2014;
The minimum number of masks of length <code>mask_feature_length</code> generated along the feature axis, each time
step, irrespectively of <code>mask_feature_prob</code>. Only relevant if
&#x201D;mask_feature_prob*len(feature_axis)/mask_feature_length &lt; mask_feature_min_masks&#x201D;`,name:"mask_feature_min_masks"},{anchor:"transformers.SpeechT5Config.num_mel_bins",description:`<strong>num_mel_bins</strong> (<code>int</code>, <em>optional</em>, defaults to 80) &#x2014;
Number of mel features used per input features. Used by the speech decoder pre-net. Should correspond to
the value used in the <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor">SpeechT5Processor</a> class.`,name:"num_mel_bins"},{anchor:"transformers.SpeechT5Config.speech_decoder_prenet_layers",description:`<strong>speech_decoder_prenet_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of layers in the speech decoder pre-net.`,name:"speech_decoder_prenet_layers"},{anchor:"transformers.SpeechT5Config.speech_decoder_prenet_units",description:`<strong>speech_decoder_prenet_units</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the layers in the speech decoder pre-net.`,name:"speech_decoder_prenet_units"},{anchor:"transformers.SpeechT5Config.speech_decoder_prenet_dropout",description:`<strong>speech_decoder_prenet_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The dropout probability for the speech decoder pre-net layers.`,name:"speech_decoder_prenet_dropout"},{anchor:"transformers.SpeechT5Config.speaker_embedding_dim",description:`<strong>speaker_embedding_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the <em>XVector</em> embedding vectors.`,name:"speaker_embedding_dim"},{anchor:"transformers.SpeechT5Config.speech_decoder_postnet_layers",description:`<strong>speech_decoder_postnet_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 5) &#x2014;
Number of layers in the speech decoder post-net.`,name:"speech_decoder_postnet_layers"},{anchor:"transformers.SpeechT5Config.speech_decoder_postnet_units",description:`<strong>speech_decoder_postnet_units</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the layers in the speech decoder post-net.`,name:"speech_decoder_postnet_units"},{anchor:"transformers.SpeechT5Config.speech_decoder_postnet_kernel",description:`<strong>speech_decoder_postnet_kernel</strong> (<code>int</code>, <em>optional</em>, defaults to 5) &#x2014;
Number of convolutional filter channels in the speech decoder post-net.`,name:"speech_decoder_postnet_kernel"},{anchor:"transformers.SpeechT5Config.speech_decoder_postnet_dropout",description:`<strong>speech_decoder_postnet_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The dropout probability for the speech decoder post-net layers.`,name:"speech_decoder_postnet_dropout"},{anchor:"transformers.SpeechT5Config.reduction_factor",description:`<strong>reduction_factor</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Spectrogram length reduction factor for the speech decoder inputs.`,name:"reduction_factor"},{anchor:"transformers.SpeechT5Config.max_speech_positions",description:`<strong>max_speech_positions</strong> (<code>int</code>, <em>optional</em>, defaults to 4000) &#x2014;
The maximum sequence length of speech features that this model might ever be used with.`,name:"max_speech_positions"},{anchor:"transformers.SpeechT5Config.max_text_positions",description:`<strong>max_text_positions</strong> (<code>int</code>, <em>optional</em>, defaults to 450) &#x2014;
The maximum sequence length of text features that this model might ever be used with.`,name:"max_text_positions"},{anchor:"transformers.SpeechT5Config.encoder_max_relative_position",description:`<strong>encoder_max_relative_position</strong> (<code>int</code>, <em>optional</em>, defaults to 160) &#x2014;
Maximum distance for relative position embedding in the encoder.`,name:"encoder_max_relative_position"},{anchor:"transformers.SpeechT5Config.use_guided_attention_loss",description:`<strong>use_guided_attention_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to apply guided attention loss while training the TTS model.`,name:"use_guided_attention_loss"},{anchor:"transformers.SpeechT5Config.guided_attention_loss_num_heads",description:`<strong>guided_attention_loss_num_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of attention heads the guided attention loss will be applied to. Use -1 to apply this loss to all
attention heads.`,name:"guided_attention_loss_num_heads"},{anchor:"transformers.SpeechT5Config.guided_attention_loss_sigma",description:`<strong>guided_attention_loss_sigma</strong> (<code>float</code>, <em>optional</em>, defaults to 0.4) &#x2014;
Standard deviation for guided attention loss.`,name:"guided_attention_loss_sigma"},{anchor:"transformers.SpeechT5Config.guided_attention_loss_scale",description:`<strong>guided_attention_loss_scale</strong> (<code>float</code>, <em>optional</em>, defaults to 10.0) &#x2014;
Scaling coefficient for guided attention loss (also known as lambda).`,name:"guided_attention_loss_scale"},{anchor:"transformers.SpeechT5Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/configuration_speecht5.py#L27"}}),pe=new jo({props:{anchor:"transformers.SpeechT5Config.example",$$slots:{default:[aa]},$$scope:{ctx:S}}}),Ge=new X({props:{title:"SpeechT5HifiGanConfig",local:"transformers.SpeechT5HifiGanConfig",headingTag:"h2"}}),He=new w({props:{name:"class transformers.SpeechT5HifiGanConfig",anchor:"transformers.SpeechT5HifiGanConfig",parameters:[{name:"model_in_dim",val:" = 80"},{name:"sampling_rate",val:" = 16000"},{name:"upsample_initial_channel",val:" = 512"},{name:"upsample_rates",val:" = [4, 4, 4, 4]"},{name:"upsample_kernel_sizes",val:" = [8, 8, 8, 8]"},{name:"resblock_kernel_sizes",val:" = [3, 7, 11]"},{name:"resblock_dilation_sizes",val:" = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]"},{name:"initializer_range",val:" = 0.01"},{name:"leaky_relu_slope",val:" = 0.1"},{name:"normalize_before",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5HifiGanConfig.model_in_dim",description:`<strong>model_in_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 80) &#x2014;
The number of frequency bins in the input log-mel spectrogram.`,name:"model_in_dim"},{anchor:"transformers.SpeechT5HifiGanConfig.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, <em>optional</em>, defaults to 16000) &#x2014;
The sampling rate at which the output audio will be generated, expressed in hertz (Hz).`,name:"sampling_rate"},{anchor:"transformers.SpeechT5HifiGanConfig.upsample_initial_channel",description:`<strong>upsample_initial_channel</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The number of input channels into the upsampling network.`,name:"upsample_initial_channel"},{anchor:"transformers.SpeechT5HifiGanConfig.upsample_rates",description:`<strong>upsample_rates</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>[4, 4, 4, 4]</code>) &#x2014;
A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
length of <em>upsample_rates</em> defines the number of convolutional layers and has to match the length of
<em>upsample_kernel_sizes</em>.`,name:"upsample_rates"},{anchor:"transformers.SpeechT5HifiGanConfig.upsample_kernel_sizes",description:`<strong>upsample_kernel_sizes</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>[8, 8, 8, 8]</code>) &#x2014;
A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
length of <em>upsample_kernel_sizes</em> defines the number of convolutional layers and has to match the length of
<em>upsample_rates</em>.`,name:"upsample_kernel_sizes"},{anchor:"transformers.SpeechT5HifiGanConfig.resblock_kernel_sizes",description:`<strong>resblock_kernel_sizes</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>[3, 7, 11]</code>) &#x2014;
A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
fusion (MRF) module.`,name:"resblock_kernel_sizes"},{anchor:"transformers.SpeechT5HifiGanConfig.resblock_dilation_sizes",description:`<strong>resblock_dilation_sizes</strong> (<code>tuple[tuple[int]]</code> or <code>list[list[int]]</code>, <em>optional</em>, defaults to <code>[[1, 3, 5], [1, 3, 5], [1, 3, 5]]</code>) &#x2014;
A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
multi-receptive field fusion (MRF) module.`,name:"resblock_dilation_sizes"},{anchor:"transformers.SpeechT5HifiGanConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.01) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.SpeechT5HifiGanConfig.leaky_relu_slope",description:`<strong>leaky_relu_slope</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The angle of the negative slope used by the leaky ReLU activation.`,name:"leaky_relu_slope"},{anchor:"transformers.SpeechT5HifiGanConfig.normalize_before",description:`<strong>normalize_before</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to normalize the spectrogram before vocoding using the vocoder&#x2019;s learned mean and variance.`,name:"normalize_before"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/configuration_speecht5.py#L340"}}),he=new jo({props:{anchor:"transformers.SpeechT5HifiGanConfig.example",$$slots:{default:[ca]},$$scope:{ctx:S}}}),Le=new X({props:{title:"SpeechT5Tokenizer",local:"transformers.SpeechT5Tokenizer",headingTag:"h2"}}),Ee=new w({props:{name:"class transformers.SpeechT5Tokenizer",anchor:"transformers.SpeechT5Tokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"normalize",val:" = False"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5Tokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.SpeechT5Tokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The begin of sequence token.`,name:"bos_token"},{anchor:"transformers.SpeechT5Tokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.SpeechT5Tokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.SpeechT5Tokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.SpeechT5Tokenizer.normalize",description:`<strong>normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to convert numeric quantities in the text to their spelt-out english counterparts.`,name:"normalize"},{anchor:"transformers.SpeechT5Tokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.SpeechT5Tokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/tokenization_speecht5.py#L35"}}),Ve=new w({props:{name:"__call__",anchor:"transformers.SpeechT5Tokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5Tokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.SpeechT5Tokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.SpeechT5Tokenizer.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.SpeechT5Tokenizer.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.SpeechT5Tokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.SpeechT5Tokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.SpeechT5Tokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.SpeechT5Tokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.SpeechT5Tokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.SpeechT5Tokenizer.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.SpeechT5Tokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.SpeechT5Tokenizer.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.SpeechT5Tokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.SpeechT5Tokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.SpeechT5Tokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.SpeechT5Tokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.SpeechT5Tokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.SpeechT5Tokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.SpeechT5Tokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.SpeechT5Tokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.SpeechT5Tokenizer.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
`}}),De=new w({props:{name:"save_vocabulary",anchor:"transformers.SpeechT5Tokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/tokenization_speecht5.py#L205"}}),Be=new w({props:{name:"decode",anchor:"transformers.SpeechT5Tokenizer.decode",parameters:[{name:"token_ids",val:": typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5Tokenizer.decode.token_ids",description:`<strong>token_ids</strong> (<code>Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"token_ids"},{anchor:"transformers.SpeechT5Tokenizer.decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.SpeechT5Tokenizer.decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.SpeechT5Tokenizer.decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3867",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded sentence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),Re=new w({props:{name:"batch_decode",anchor:"transformers.SpeechT5Tokenizer.batch_decode",parameters:[{name:"sequences",val:": typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5Tokenizer.batch_decode.sequences",description:`<strong>sequences</strong> (<code>Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"sequences"},{anchor:"transformers.SpeechT5Tokenizer.batch_decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.SpeechT5Tokenizer.batch_decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.SpeechT5Tokenizer.batch_decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3833",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of decoded sentences.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code></p>
`}}),Oe=new X({props:{title:"SpeechT5FeatureExtractor",local:"transformers.SpeechT5FeatureExtractor",headingTag:"h2"}}),Xe=new w({props:{name:"class transformers.SpeechT5FeatureExtractor",anchor:"transformers.SpeechT5FeatureExtractor",parameters:[{name:"feature_size",val:": int = 1"},{name:"sampling_rate",val:": int = 16000"},{name:"padding_value",val:": float = 0.0"},{name:"do_normalize",val:": bool = False"},{name:"num_mel_bins",val:": int = 80"},{name:"hop_length",val:": int = 16"},{name:"win_length",val:": int = 64"},{name:"win_function",val:": str = 'hann_window'"},{name:"frame_signal_scale",val:": float = 1.0"},{name:"fmin",val:": float = 80"},{name:"fmax",val:": float = 7600"},{name:"mel_floor",val:": float = 1e-10"},{name:"reduction_factor",val:": int = 2"},{name:"return_attention_mask",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5FeatureExtractor.feature_size",description:`<strong>feature_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The feature dimension of the extracted features.`,name:"feature_size"},{anchor:"transformers.SpeechT5FeatureExtractor.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, <em>optional</em>, defaults to 16000) &#x2014;
The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).`,name:"sampling_rate"},{anchor:"transformers.SpeechT5FeatureExtractor.padding_value",description:`<strong>padding_value</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The value that is used to fill the padding values.`,name:"padding_value"},{anchor:"transformers.SpeechT5FeatureExtractor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
improve the performance for some models.`,name:"do_normalize"},{anchor:"transformers.SpeechT5FeatureExtractor.num_mel_bins",description:`<strong>num_mel_bins</strong> (<code>int</code>, <em>optional</em>, defaults to 80) &#x2014;
The number of mel-frequency bins in the extracted spectrogram features.`,name:"num_mel_bins"},{anchor:"transformers.SpeechT5FeatureExtractor.hop_length",description:`<strong>hop_length</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of ms between windows. Otherwise referred to as &#x201C;shift&#x201D; in many papers.`,name:"hop_length"},{anchor:"transformers.SpeechT5FeatureExtractor.win_length",description:`<strong>win_length</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of ms per window.`,name:"win_length"},{anchor:"transformers.SpeechT5FeatureExtractor.win_function",description:`<strong>win_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;hann_window&quot;</code>) &#x2014;
Name for the window function used for windowing, must be accessible via <code>torch.{win_function}</code>`,name:"win_function"},{anchor:"transformers.SpeechT5FeatureExtractor.frame_signal_scale",description:`<strong>frame_signal_scale</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Constant multiplied in creating the frames before applying DFT. This argument is deprecated.`,name:"frame_signal_scale"},{anchor:"transformers.SpeechT5FeatureExtractor.fmin",description:`<strong>fmin</strong> (<code>float</code>, <em>optional</em>, defaults to 80) &#x2014;
Minimum mel frequency in Hz.`,name:"fmin"},{anchor:"transformers.SpeechT5FeatureExtractor.fmax",description:`<strong>fmax</strong> (<code>float</code>, <em>optional</em>, defaults to 7600) &#x2014;
Maximum mel frequency in Hz.`,name:"fmax"},{anchor:"transformers.SpeechT5FeatureExtractor.mel_floor",description:`<strong>mel_floor</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-10) &#x2014;
Minimum value of mel frequency banks.`,name:"mel_floor"},{anchor:"transformers.SpeechT5FeatureExtractor.reduction_factor",description:`<strong>reduction_factor</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Spectrogram length reduction factor. This argument is deprecated.`,name:"reduction_factor"},{anchor:"transformers.SpeechT5FeatureExtractor.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor.__call__"><strong>call</strong>()</a> should return <code>attention_mask</code>.`,name:"return_attention_mask"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/feature_extraction_speecht5.py#L31"}}),Ye=new w({props:{name:"__call__",anchor:"transformers.SpeechT5FeatureExtractor.__call__",parameters:[{name:"audio",val:": typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]], NoneType] = None"},{name:"audio_target",val:": typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]], NoneType] = None"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"truncation",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"sampling_rate",val:": typing.Optional[int] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5FeatureExtractor.__call__.audio",description:`<strong>audio</strong> (<code>np.ndarray</code>, <code>list[float]</code>, <code>list[np.ndarray]</code>, <code>list[list[float]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
values, a list of numpy arrays or a list of list of float values. This outputs waveform features. Must
be mono channel audio, not stereo, i.e. single float per timestep.`,name:"audio"},{anchor:"transformers.SpeechT5FeatureExtractor.__call__.audio_target",description:`<strong>audio_target</strong> (<code>np.ndarray</code>, <code>list[float]</code>, <code>list[np.ndarray]</code>, <code>list[list[float]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be processed as targets. Each sequence can be a numpy array, a
list of float values, a list of numpy arrays or a list of list of float values. This outputs log-mel
spectrogram features.`,name:"audio_target"},{anchor:"transformers.SpeechT5FeatureExtractor.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.SpeechT5FeatureExtractor.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length of the returned list and optionally padding length (see above).`,name:"max_length"},{anchor:"transformers.SpeechT5FeatureExtractor.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>) &#x2014;
Activates truncation to cut input sequences longer than <em>max_length</em> to <em>max_length</em>.`,name:"truncation"},{anchor:"transformers.SpeechT5FeatureExtractor.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value.</p>
<p>This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.`,name:"pad_to_multiple_of"},{anchor:"transformers.SpeechT5FeatureExtractor.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific feature_extractor&#x2019;s default.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.SpeechT5FeatureExtractor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.SpeechT5FeatureExtractor.__call__.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The sampling rate at which the <code>audio</code> or <code>audio_target</code> input was sampled. It is strongly recommended
to pass <code>sampling_rate</code> at the forward call to prevent silent errors.`,name:"sampling_rate"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/feature_extraction_speecht5.py#L180"}}),Qe=new X({props:{title:"SpeechT5Processor",local:"transformers.SpeechT5Processor",headingTag:"h2"}}),Ae=new w({props:{name:"class transformers.SpeechT5Processor",anchor:"transformers.SpeechT5Processor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5Processor.feature_extractor",description:`<strong>feature_extractor</strong> (<code>SpeechT5FeatureExtractor</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor">SpeechT5FeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.SpeechT5Processor.tokenizer",description:`<strong>tokenizer</strong> (<code>SpeechT5Tokenizer</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer">SpeechT5Tokenizer</a>. The tokenizer is a required input.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/processing_speecht5.py#L20"}}),Ke=new w({props:{name:"__call__",anchor:"transformers.SpeechT5Processor.__call__",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/processing_speecht5.py#L40"}}),et=new w({props:{name:"pad",anchor:"transformers.SpeechT5Processor.pad",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/processing_speecht5.py#L111"}}),tt=new w({props:{name:"from_pretrained",anchor:"transformers.SpeechT5Processor.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:": typing.Union[str, os.PathLike]"},{name:"cache_dir",val:": typing.Union[str, os.PathLike, NoneType] = None"},{name:"force_download",val:": bool = False"},{name:"local_files_only",val:": bool = False"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"revision",val:": str = 'main'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5Processor.from_pretrained.pretrained_model_name_or_path",description:`<strong>pretrained_model_name_or_path</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
This can be either:</p>
<ul>
<li>a string, the <em>model id</em> of a pretrained feature_extractor hosted inside a model repo on
huggingface.co.</li>
<li>a path to a <em>directory</em> containing a feature extractor file saved using the
<a href="/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained">save_pretrained()</a> method, e.g., <code>./my_model_directory/</code>.</li>
<li>a path or url to a saved feature extractor JSON <em>file</em>, e.g.,
<code>./my_model_directory/preprocessor_config.json</code>.</li>
</ul>`,name:"pretrained_model_name_or_path"},{anchor:"transformers.SpeechT5Processor.from_pretrained.*kwargs",description:`*<strong>*kwargs</strong> &#x2014;
Additional keyword arguments passed along to both
<a href="/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained">from_pretrained()</a> and
<code>~tokenization_utils_base.PreTrainedTokenizer.from_pretrained</code>.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1272"}}),fe=new qo({props:{$$slots:{default:[ia]},$$scope:{ctx:S}}}),ot=new w({props:{name:"save_pretrained",anchor:"transformers.SpeechT5Processor.save_pretrained",parameters:[{name:"save_directory",val:""},{name:"push_to_hub",val:": bool = False"},{name:"legacy_serialization",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5Processor.save_pretrained.save_directory",description:`<strong>save_directory</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
be created if it does not exist).`,name:"save_directory"},{anchor:"transformers.SpeechT5Processor.save_pretrained.push_to_hub",description:`<strong>push_to_hub</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with <code>repo_id</code> (will default to the name of <code>save_directory</code> in your
namespace).`,name:"push_to_hub"},{anchor:"transformers.SpeechT5Processor.save_pretrained.legacy_serialization",description:`<strong>legacy_serialization</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to save processor attributes in separate config files (legacy) or in processor&#x2019;s config
file as a nested dict. Saving all attributes in a single dict will become the default in future versions.
Set to <code>legacy_serialization=True</code> until then.`,name:"legacy_serialization"},{anchor:"transformers.SpeechT5Processor.save_pretrained.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Additional key word arguments passed along to the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub">push_to_hub()</a> method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L653"}}),ge=new qo({props:{$$slots:{default:[da]},$$scope:{ctx:S}}}),nt=new w({props:{name:"batch_decode",anchor:"transformers.SpeechT5Processor.batch_decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1419"}}),st=new w({props:{name:"decode",anchor:"transformers.SpeechT5Processor.decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1428"}}),rt=new X({props:{title:"SpeechT5Model",local:"transformers.SpeechT5Model",headingTag:"h2"}}),at=new w({props:{name:"class transformers.SpeechT5Model",anchor:"transformers.SpeechT5Model",parameters:[{name:"config",val:": SpeechT5Config"},{name:"encoder",val:": typing.Optional[torch.nn.modules.module.Module] = None"},{name:"decoder",val:": typing.Optional[torch.nn.modules.module.Module] = None"}],parametersDescription:[{anchor:"transformers.SpeechT5Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config">SpeechT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.SpeechT5Model.encoder",description:`<strong>encoder</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The encoder model to use.`,name:"encoder"},{anchor:"transformers.SpeechT5Model.decoder",description:`<strong>decoder</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The decoder model to use.`,name:"decoder"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L1948"}}),ct=new w({props:{name:"forward",anchor:"transformers.SpeechT5Model.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_values",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"speaker_embeddings",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.SpeechT5Model.forward.input_values",description:`<strong>input_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Depending on which encoder is being used, the <code>input_values</code> are either: float values of the input raw
speech waveform, or indices of input sequence tokens in the vocabulary, or hidden states.`,name:"input_values"},{anchor:"transformers.SpeechT5Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SpeechT5Model.forward.decoder_input_values",description:`<strong>decoder_input_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Depending on which decoder is being used, the <code>decoder_input_values</code> are either: float values of log-mel
filterbank features extracted from the raw speech waveform, or indices of decoder input sequence tokens in
the vocabulary, or hidden states.`,name:"decoder_input_values"},{anchor:"transformers.SpeechT5Model.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_values</code>. Causal mask will
also be used by default.</p>
<p>If you want to change padding behavior, you should read <code>SpeechT5Decoder._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.SpeechT5Model.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SpeechT5Model.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.SpeechT5Model.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.SpeechT5Model.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.SpeechT5Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SpeechT5Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SpeechT5Model.forward.speaker_embeddings",description:`<strong>speaker_embeddings</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.speaker_embedding_dim)</code>, <em>optional</em>) &#x2014;
Tensor containing the speaker embeddings.`,name:"speaker_embeddings"},{anchor:"transformers.SpeechT5Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SpeechT5Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SpeechT5Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.SpeechT5Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L1993",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config"
>SpeechT5Config</a>) and inputs.</p>
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
`}}),ve=new qo({props:{$$slots:{default:[la]},$$scope:{ctx:S}}}),it=new X({props:{title:"SpeechT5ForSpeechToText",local:"transformers.SpeechT5ForSpeechToText",headingTag:"h2"}}),dt=new w({props:{name:"class transformers.SpeechT5ForSpeechToText",anchor:"transformers.SpeechT5ForSpeechToText",parameters:[{name:"config",val:": SpeechT5Config"}],parametersDescription:[{anchor:"transformers.SpeechT5ForSpeechToText.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config">SpeechT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2109"}}),lt=new w({props:{name:"forward",anchor:"transformers.SpeechT5ForSpeechToText.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.SpeechT5ForSpeechToText.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor">SpeechT5Processor</a> should be used for padding
and conversion into a tensor of type <code>torch.FloatTensor</code>. See <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__">SpeechT5Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer">SpeechT5Tokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>SpeechT5 uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_values</code>. Causal mask will
also be used by default.</p>
<p>If you want to change padding behavior, you should read <code>SpeechT5Decoder._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code>
or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored (masked), the loss is
only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.</p>
<p>Label indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer">SpeechT5Tokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"labels"},{anchor:"transformers.SpeechT5ForSpeechToText.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2151",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config"
>SpeechT5Config</a>) and inputs.</p>
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
`}}),be=new qo({props:{$$slots:{default:[pa]},$$scope:{ctx:S}}}),ye=new jo({props:{anchor:"transformers.SpeechT5ForSpeechToText.forward.example",$$slots:{default:[ha]},$$scope:{ctx:S}}}),ke=new jo({props:{anchor:"transformers.SpeechT5ForSpeechToText.forward.example-2",$$slots:{default:[ma]},$$scope:{ctx:S}}}),pt=new X({props:{title:"SpeechT5ForTextToSpeech",local:"transformers.SpeechT5ForTextToSpeech",headingTag:"h2"}}),ht=new w({props:{name:"class transformers.SpeechT5ForTextToSpeech",anchor:"transformers.SpeechT5ForTextToSpeech",parameters:[{name:"config",val:": SpeechT5Config"}],parametersDescription:[{anchor:"transformers.SpeechT5ForTextToSpeech.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config">SpeechT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2439"}}),mt=new w({props:{name:"forward",anchor:"transformers.SpeechT5ForTextToSpeech.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"speaker_embeddings",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.FloatTensor] = None"},{name:"stop_labels",val:": typing.Optional[torch.Tensor] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.SpeechT5ForTextToSpeech.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer">SpeechT5Tokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__"><strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.decoder_input_values",description:`<strong>decoder_input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_mel_bins)</code>) &#x2014;
Float values of input mel spectrogram.</p>
<p>SpeechT5 uses an all-zero spectrum as the starting token for <code>decoder_input_values</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_values</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_values"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_values</code>. Causal mask will
also be used by default.</p>
<p>If you want to change padding behavior, you should read <code>SpeechT5Decoder._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.speaker_embeddings",description:`<strong>speaker_embeddings</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.speaker_embedding_dim)</code>, <em>optional</em>) &#x2014;
Tensor containing the speaker embeddings.`,name:"speaker_embeddings"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.labels",description:`<strong>labels</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_mel_bins)</code>, <em>optional</em>) &#x2014;
Float values of target mel spectrogram. Timesteps set to <code>-100.0</code> are ignored (masked) for the loss
computation. Spectrograms can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor">SpeechT5Processor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__">SpeechT5Processor.<strong>call</strong>()</a>
for details.`,name:"labels"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.stop_labels",description:`<strong>stop_labels</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Binary tensor indicating the position of the stop token in the sequence.`,name:"stop_labels"},{anchor:"transformers.SpeechT5ForTextToSpeech.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2475",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput"
>transformers.modeling_outputs.Seq2SeqSpectrogramOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config"
>SpeechT5Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Spectrogram generation loss.</p>
</li>
<li>
<p><strong>spectrogram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, num_bins)</code>) — The predicted spectrogram.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput"
>transformers.modeling_outputs.Seq2SeqSpectrogramOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),xe=new qo({props:{$$slots:{default:[ua]},$$scope:{ctx:S}}}),we=new jo({props:{anchor:"transformers.SpeechT5ForTextToSpeech.forward.example",$$slots:{default:[fa]},$$scope:{ctx:S}}}),ut=new w({props:{name:"generate",anchor:"transformers.SpeechT5ForTextToSpeech.generate",parameters:[{name:"input_ids",val:": LongTensor"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"speaker_embeddings",val:": typing.Optional[torch.FloatTensor] = None"},{name:"threshold",val:": float = 0.5"},{name:"minlenratio",val:": float = 0.0"},{name:"maxlenratio",val:": float = 20.0"},{name:"vocoder",val:": typing.Optional[torch.nn.modules.module.Module] = None"},{name:"output_cross_attentions",val:": bool = False"},{name:"return_output_lengths",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SpeechT5ForTextToSpeech.generate.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer">SpeechT5Tokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__"><strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SpeechT5ForTextToSpeech.generate.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Attention mask from the tokenizer, required for batched inference to signal to the model where to
ignore padded tokens from the input_ids.`,name:"attention_mask"},{anchor:"transformers.SpeechT5ForTextToSpeech.generate.speaker_embeddings",description:`<strong>speaker_embeddings</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.speaker_embedding_dim)</code>, <em>optional</em>) &#x2014;
Tensor containing the speaker embeddings.`,name:"speaker_embeddings"},{anchor:"transformers.SpeechT5ForTextToSpeech.generate.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The generated sequence ends when the predicted stop token probability exceeds this value.`,name:"threshold"},{anchor:"transformers.SpeechT5ForTextToSpeech.generate.minlenratio",description:`<strong>minlenratio</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Used to calculate the minimum required length for the output sequence.`,name:"minlenratio"},{anchor:"transformers.SpeechT5ForTextToSpeech.generate.maxlenratio",description:`<strong>maxlenratio</strong> (<code>float</code>, <em>optional</em>, defaults to 20.0) &#x2014;
Used to calculate the maximum allowed length for the output sequence.`,name:"maxlenratio"},{anchor:"transformers.SpeechT5ForTextToSpeech.generate.vocoder",description:`<strong>vocoder</strong> (<code>nn.Module</code>, <em>optional</em>) &#x2014;
The vocoder that converts the mel spectrogram into a speech waveform. If <code>None</code>, the output is the mel
spectrogram.`,name:"vocoder"},{anchor:"transformers.SpeechT5ForTextToSpeech.generate.output_cross_attentions",description:`<strong>output_cross_attentions</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the attentions tensors of the decoder&#x2019;s cross-attention layers.`,name:"output_cross_attentions"},{anchor:"transformers.SpeechT5ForTextToSpeech.generate.return_output_lengths",description:`<strong>return_output_lengths</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the concrete spectrogram/waveform lengths.`,name:"return_output_lengths"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2610",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<ul>
<li>when <code>return_output_lengths</code> is False<ul>
<li><strong>spectrogram</strong> (<em>optional</em>, returned when no <code>vocoder</code> is provided) <code>torch.FloatTensor</code> of shape
<code>(output_sequence_length, config.num_mel_bins)</code> — The predicted log-mel spectrogram.</li>
<li><strong>waveform</strong> (<em>optional</em>, returned when a <code>vocoder</code> is provided) <code>torch.FloatTensor</code> of shape
<code>(num_frames,)</code> — The predicted speech waveform.</li>
<li><strong>cross_attentions</strong> (<em>optional</em>, returned when <code>output_cross_attentions</code> is <code>True</code>)
<code>torch.FloatTensor</code> of shape <code>(config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)</code> — The outputs of the decoder’s cross-attention layers.</li>
</ul></li>
<li>when <code>return_output_lengths</code> is True<ul>
<li><strong>spectrograms</strong> (<em>optional</em>, returned when no <code>vocoder</code> is provided) <code>torch.FloatTensor</code> of shape
<code>(batch_size, output_sequence_length, config.num_mel_bins)</code> — The predicted log-mel spectrograms that
are padded to the maximum length.</li>
<li><strong>spectrogram_lengths</strong> (<em>optional</em>, returned when no <code>vocoder</code> is provided) <code>list[Int]</code> — A list of
all the concrete lengths for each spectrogram.</li>
<li><strong>waveforms</strong> (<em>optional</em>, returned when a <code>vocoder</code> is provided) <code>torch.FloatTensor</code> of shape
<code>(batch_size, num_frames)</code> — The predicted speech waveforms that are padded to the maximum length.</li>
<li><strong>waveform_lengths</strong> (<em>optional</em>, returned when a <code>vocoder</code> is provided) <code>list[Int]</code> — A list of all
the concrete lengths for each waveform.</li>
<li><strong>cross_attentions</strong> (<em>optional</em>, returned when <code>output_cross_attentions</code> is <code>True</code>)
<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)</code> — The outputs of the decoder’s cross-attention layers.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>tuple(torch.FloatTensor)</code> comprising various elements depending on the inputs</p>
`}}),ft=new X({props:{title:"SpeechT5ForSpeechToSpeech",local:"transformers.SpeechT5ForSpeechToSpeech",headingTag:"h2"}}),gt=new w({props:{name:"class transformers.SpeechT5ForSpeechToSpeech",anchor:"transformers.SpeechT5ForSpeechToSpeech",parameters:[{name:"config",val:": SpeechT5Config"}],parametersDescription:[{anchor:"transformers.SpeechT5ForSpeechToSpeech.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config">SpeechT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2804"}}),_t=new w({props:{name:"forward",anchor:"transformers.SpeechT5ForSpeechToSpeech.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"speaker_embeddings",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.FloatTensor] = None"},{name:"stop_labels",val:": typing.Optional[torch.Tensor] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor">SpeechT5Processor</a> should be used for padding and conversion into
a tensor of type <code>torch.FloatTensor</code>. See <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__">SpeechT5Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.decoder_input_values",description:`<strong>decoder_input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_mel_bins)</code>) &#x2014;
Float values of input mel spectrogram.</p>
<p>SpeechT5 uses an all-zero spectrum as the starting token for <code>decoder_input_values</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_values</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_values"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_values</code>. Causal mask will
also be used by default.</p>
<p>If you want to change padding behavior, you should read <code>SpeechT5Decoder._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.speaker_embeddings",description:`<strong>speaker_embeddings</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.speaker_embedding_dim)</code>, <em>optional</em>) &#x2014;
Tensor containing the speaker embeddings.`,name:"speaker_embeddings"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.labels",description:`<strong>labels</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_mel_bins)</code>, <em>optional</em>) &#x2014;
Float values of target mel spectrogram. Spectrograms can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor">SpeechT5Processor</a>. See
<a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__">SpeechT5Processor.<strong>call</strong>()</a> for details.`,name:"labels"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.stop_labels",description:`<strong>stop_labels</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Binary tensor indicating the position of the stop token in the sequence.`,name:"stop_labels"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2830",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput"
>transformers.modeling_outputs.Seq2SeqSpectrogramOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config"
>SpeechT5Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Spectrogram generation loss.</p>
</li>
<li>
<p><strong>spectrogram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, num_bins)</code>) — The predicted spectrogram.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput"
>transformers.modeling_outputs.Seq2SeqSpectrogramOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),$e=new qo({props:{$$slots:{default:[ga]},$$scope:{ctx:S}}}),Me=new jo({props:{anchor:"transformers.SpeechT5ForSpeechToSpeech.forward.example",$$slots:{default:[_a]},$$scope:{ctx:S}}}),Tt=new w({props:{name:"generate_speech",anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech",parameters:[{name:"input_values",val:": FloatTensor"},{name:"speaker_embeddings",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"threshold",val:": float = 0.5"},{name:"minlenratio",val:": float = 0.0"},{name:"maxlenratio",val:": float = 20.0"},{name:"vocoder",val:": typing.Optional[torch.nn.modules.module.Module] = None"},{name:"output_cross_attentions",val:": bool = False"},{name:"return_output_lengths",val:": bool = False"}],parametersDescription:[{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform.</p>
<p>Values can be obtained by loading a <em>.flac</em> or <em>.wav</em> audio file into an array of type <code>list[float]</code>,
a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library (<code>pip install torchcodec</code>)
or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor">SpeechT5Processor</a> should be used for padding and
conversion into a tensor of type <code>torch.FloatTensor</code>. See <a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__">SpeechT5Processor.<strong>call</strong>()</a> for details.`,name:"input_values"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.speaker_embeddings",description:`<strong>speaker_embeddings</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.speaker_embedding_dim)</code>, <em>optional</em>) &#x2014;
Tensor containing the speaker embeddings.`,name:"speaker_embeddings"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The generated sequence ends when the predicted stop token probability exceeds this value.`,name:"threshold"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.minlenratio",description:`<strong>minlenratio</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Used to calculate the minimum required length for the output sequence.`,name:"minlenratio"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.maxlenratio",description:`<strong>maxlenratio</strong> (<code>float</code>, <em>optional</em>, defaults to 20.0) &#x2014;
Used to calculate the maximum allowed length for the output sequence.`,name:"maxlenratio"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.vocoder",description:`<strong>vocoder</strong> (<code>nn.Module</code>, <em>optional</em>, defaults to <code>None</code>) &#x2014;
The vocoder that converts the mel spectrogram into a speech waveform. If <code>None</code>, the output is the mel
spectrogram.`,name:"vocoder"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.output_cross_attentions",description:`<strong>output_cross_attentions</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the attentions tensors of the decoder&#x2019;s cross-attention layers.`,name:"output_cross_attentions"},{anchor:"transformers.SpeechT5ForSpeechToSpeech.generate_speech.return_output_lengths",description:`<strong>return_output_lengths</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the concrete spectrogram/waveform lengths.`,name:"return_output_lengths"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2960",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<ul>
<li>when <code>return_output_lengths</code> is False<ul>
<li><strong>spectrogram</strong> (<em>optional</em>, returned when no <code>vocoder</code> is provided) <code>torch.FloatTensor</code> of shape
<code>(output_sequence_length, config.num_mel_bins)</code> — The predicted log-mel spectrogram.</li>
<li><strong>waveform</strong> (<em>optional</em>, returned when a <code>vocoder</code> is provided) <code>torch.FloatTensor</code> of shape
<code>(num_frames,)</code> — The predicted speech waveform.</li>
<li><strong>cross_attentions</strong> (<em>optional</em>, returned when <code>output_cross_attentions</code> is <code>True</code>)
<code>torch.FloatTensor</code> of shape <code>(config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)</code> — The outputs of the decoder’s cross-attention layers.</li>
</ul></li>
<li>when <code>return_output_lengths</code> is True<ul>
<li><strong>spectrograms</strong> (<em>optional</em>, returned when no <code>vocoder</code> is provided) <code>torch.FloatTensor</code> of shape
<code>(batch_size, output_sequence_length, config.num_mel_bins)</code> — The predicted log-mel spectrograms that
are padded to the maximum length.</li>
<li><strong>spectrogram_lengths</strong> (<em>optional</em>, returned when no <code>vocoder</code> is provided) <code>list[Int]</code> — A list of
all the concrete lengths for each spectrogram.</li>
<li><strong>waveforms</strong> (<em>optional</em>, returned when a <code>vocoder</code> is provided) <code>torch.FloatTensor</code> of shape
<code>(batch_size, num_frames)</code> — The predicted speech waveforms that are padded to the maximum length.</li>
<li><strong>waveform_lengths</strong> (<em>optional</em>, returned when a <code>vocoder</code> is provided) <code>list[Int]</code> — A list of all
the concrete lengths for each waveform.</li>
<li><strong>cross_attentions</strong> (<em>optional</em>, returned when <code>output_cross_attentions</code> is <code>True</code>)
<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)</code> — The outputs of the decoder’s cross-attention layers.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>tuple(torch.FloatTensor)</code> comprising various elements depending on the inputs</p>
`}}),vt=new X({props:{title:"SpeechT5HifiGan",local:"transformers.SpeechT5HifiGan",headingTag:"h2"}}),bt=new w({props:{name:"class transformers.SpeechT5HifiGan",anchor:"transformers.SpeechT5HifiGan",parameters:[{name:"config",val:": SpeechT5HifiGanConfig"}],parametersDescription:[{anchor:"transformers.SpeechT5HifiGan.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5HifiGanConfig">SpeechT5HifiGanConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L3118"}}),yt=new w({props:{name:"forward",anchor:"transformers.SpeechT5HifiGan.forward",parameters:[{name:"spectrogram",val:": FloatTensor"}],parametersDescription:[{anchor:"transformers.SpeechT5HifiGan.forward.spectrogram",description:`<strong>spectrogram</strong> (<code>torch.FloatTensor</code>) &#x2014;
Tensor containing the log-mel spectrograms. Can be batched and of shape <code>(batch_size, sequence_length, config.model_in_dim)</code>, or un-batched and of shape <code>(sequence_length, config.model_in_dim)</code>.`,name:"spectrogram"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L3187",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Tensor containing the speech waveform. If the input spectrogram is batched, will be of
shape <code>(batch_size, num_frames,)</code>. If un-batched, will be of shape <code>(num_frames,)</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>torch.FloatTensor</code></p>
`}}),kt=new ra({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/speecht5.md"}}),{c(){s=r("meta"),b=o(),l=r("p"),v=o(),y=r("p"),y.innerHTML=d,$=o(),m(je.$$.fragment),Io=o(),le=r("div"),le.innerHTML=Os,Po=o(),m(Ne.$$.fragment),Wo=o(),Ue=r("p"),Ue.innerHTML=Xs,Zo=o(),Je=r("p"),Je.textContent=Ys,Go=o(),Ie=r("p"),Ie.innerHTML=Qs,Ho=o(),Pe=r("p"),Pe.innerHTML=As,Lo=o(),m(We.$$.fragment),Eo=o(),E=r("div"),m(Ze.$$.fragment),_n=o(),St=r("p"),St.innerHTML=Ks,Tn=o(),$t=r("p"),$t.innerHTML=er,vn=o(),m(pe.$$.fragment),Vo=o(),m(Ge.$$.fragment),Do=o(),V=r("div"),m(He.$$.fragment),bn=o(),Mt=r("p"),Mt.innerHTML=tr,yn=o(),Ft=r("p"),Ft.innerHTML=or,kn=o(),m(he.$$.fragment),Bo=o(),m(Le.$$.fragment),Ro=o(),z=r("div"),m(Ee.$$.fragment),xn=o(),zt=r("p"),zt.innerHTML=nr,wn=o(),Ct=r("p"),Ct.innerHTML=sr,Sn=o(),me=r("div"),m(Ve.$$.fragment),$n=o(),qt=r("p"),qt.textContent=rr,Mn=o(),jt=r("div"),m(De.$$.fragment),Fn=o(),K=r("div"),m(Be.$$.fragment),zn=o(),Nt=r("p"),Nt.textContent=ar,Cn=o(),Ut=r("p"),Ut.innerHTML=cr,qn=o(),ue=r("div"),m(Re.$$.fragment),jn=o(),Jt=r("p"),Jt.textContent=ir,Oo=o(),m(Oe.$$.fragment),Xo=o(),q=r("div"),m(Xe.$$.fragment),Nn=o(),It=r("p"),It.textContent=dr,Un=o(),Pt=r("p"),Pt.textContent=lr,Jn=o(),Wt=r("p"),Wt.textContent=pr,In=o(),Zt=r("p"),Zt.innerHTML=hr,Pn=o(),ee=r("div"),m(Ye.$$.fragment),Wn=o(),Gt=r("p"),Gt.textContent=mr,Zn=o(),Ht=r("p"),Ht.innerHTML=ur,Yo=o(),m(Qe.$$.fragment),Qo=o(),M=r("div"),m(Ae.$$.fragment),Gn=o(),Lt=r("p"),Lt.textContent=fr,Hn=o(),Et=r("p"),Et.innerHTML=gr,Ln=o(),C=r("div"),m(Ke.$$.fragment),En=o(),Vt=r("p"),Vt.textContent=_r,Vn=o(),Dt=r("p"),Dt.innerHTML=Tr,Dn=o(),Bt=r("p"),Bt.innerHTML=vr,Bn=o(),Rt=r("p"),Rt.textContent=br,Rn=o(),Ot=r("ul"),Ot.innerHTML=yr,On=o(),Xt=r("p"),Xt.textContent=kr,Xn=o(),I=r("div"),m(et.$$.fragment),Yn=o(),Yt=r("p"),Yt.textContent=xr,Qn=o(),Qt=r("p"),Qt.innerHTML=wr,An=o(),At=r("p"),At.textContent=Sr,Kn=o(),Kt=r("ul"),Kt.innerHTML=$r,es=o(),eo=r("p"),eo.textContent=Mr,ts=o(),te=r("div"),m(tt.$$.fragment),os=o(),to=r("p"),to.textContent=Fr,ns=o(),m(fe.$$.fragment),ss=o(),oe=r("div"),m(ot.$$.fragment),rs=o(),oo=r("p"),oo.innerHTML=zr,as=o(),m(ge.$$.fragment),cs=o(),_e=r("div"),m(nt.$$.fragment),is=o(),no=r("p"),no.innerHTML=Cr,ds=o(),Te=r("div"),m(st.$$.fragment),ls=o(),so=r("p"),so.innerHTML=qr,Ao=o(),m(rt.$$.fragment),Ko=o(),P=r("div"),m(at.$$.fragment),ps=o(),ro=r("p"),ro.textContent=jr,hs=o(),ao=r("p"),ao.innerHTML=Nr,ms=o(),co=r("p"),co.innerHTML=Ur,us=o(),ne=r("div"),m(ct.$$.fragment),fs=o(),io=r("p"),io.innerHTML=Jr,gs=o(),m(ve.$$.fragment),en=o(),m(it.$$.fragment),tn=o(),W=r("div"),m(dt.$$.fragment),_s=o(),lo=r("p"),lo.textContent=Ir,Ts=o(),po=r("p"),po.innerHTML=Pr,vs=o(),ho=r("p"),ho.innerHTML=Wr,bs=o(),L=r("div"),m(lt.$$.fragment),ys=o(),mo=r("p"),mo.innerHTML=Zr,ks=o(),m(be.$$.fragment),xs=o(),m(ye.$$.fragment),ws=o(),m(ke.$$.fragment),on=o(),m(pt.$$.fragment),nn=o(),j=r("div"),m(ht.$$.fragment),Ss=o(),uo=r("p"),uo.textContent=Gr,$s=o(),fo=r("p"),fo.innerHTML=Hr,Ms=o(),go=r("p"),go.innerHTML=Lr,Fs=o(),D=r("div"),m(mt.$$.fragment),zs=o(),_o=r("p"),_o.innerHTML=Er,Cs=o(),m(xe.$$.fragment),qs=o(),m(we.$$.fragment),js=o(),Se=r("div"),m(ut.$$.fragment),Ns=o(),To=r("p"),To.textContent=Vr,sn=o(),m(ft.$$.fragment),rn=o(),N=r("div"),m(gt.$$.fragment),Us=o(),vo=r("p"),vo.textContent=Dr,Js=o(),bo=r("p"),bo.innerHTML=Br,Is=o(),yo=r("p"),yo.innerHTML=Rr,Ps=o(),B=r("div"),m(_t.$$.fragment),Ws=o(),ko=r("p"),ko.innerHTML=Or,Zs=o(),m($e.$$.fragment),Gs=o(),m(Me.$$.fragment),Hs=o(),Fe=r("div"),m(Tt.$$.fragment),Ls=o(),xo=r("p"),xo.textContent=Xr,an=o(),m(vt.$$.fragment),cn=o(),Z=r("div"),m(bt.$$.fragment),Es=o(),wo=r("p"),wo.textContent=Yr,Vs=o(),So=r("p"),So.innerHTML=Qr,Ds=o(),$o=r("p"),$o.innerHTML=Ar,Bs=o(),ze=r("div"),m(yt.$$.fragment),Rs=o(),Mo=r("p"),Mo.textContent=Kr,dn=o(),m(kt.$$.fragment),ln=o(),Uo=r("p"),this.h()},l(e){const i=sa("svelte-u9bgzb",document.head);s=a(i,"META",{name:!0,content:!0}),i.forEach(c),b=n(e),l=a(e,"P",{}),x(l).forEach(c),v=n(e),y=a(e,"P",{"data-svelte-h":!0}),p(y)!=="svelte-xgtd95"&&(y.innerHTML=d),$=n(e),u(je.$$.fragment,e),Io=n(e),le=a(e,"DIV",{class:!0,"data-svelte-h":!0}),p(le)!=="svelte-13t8s2t"&&(le.innerHTML=Os),Po=n(e),u(Ne.$$.fragment,e),Wo=n(e),Ue=a(e,"P",{"data-svelte-h":!0}),p(Ue)!=="svelte-1lj9jtc"&&(Ue.innerHTML=Xs),Zo=n(e),Je=a(e,"P",{"data-svelte-h":!0}),p(Je)!=="svelte-vfdo9a"&&(Je.textContent=Ys),Go=n(e),Ie=a(e,"P",{"data-svelte-h":!0}),p(Ie)!=="svelte-y5xjl8"&&(Ie.innerHTML=Qs),Ho=n(e),Pe=a(e,"P",{"data-svelte-h":!0}),p(Pe)!=="svelte-th5lle"&&(Pe.innerHTML=As),Lo=n(e),u(We.$$.fragment,e),Eo=n(e),E=a(e,"DIV",{class:!0});var Q=x(E);u(Ze.$$.fragment,Q),_n=n(Q),St=a(Q,"P",{"data-svelte-h":!0}),p(St)!=="svelte-1xfmv56"&&(St.innerHTML=Ks),Tn=n(Q),$t=a(Q,"P",{"data-svelte-h":!0}),p($t)!=="svelte-1ek1ss9"&&($t.innerHTML=er),vn=n(Q),u(pe.$$.fragment,Q),Q.forEach(c),Vo=n(e),u(Ge.$$.fragment,e),Do=n(e),V=a(e,"DIV",{class:!0});var A=x(V);u(He.$$.fragment,A),bn=n(A),Mt=a(A,"P",{"data-svelte-h":!0}),p(Mt)!=="svelte-q7odiy"&&(Mt.innerHTML=tr),yn=n(A),Ft=a(A,"P",{"data-svelte-h":!0}),p(Ft)!=="svelte-1ek1ss9"&&(Ft.innerHTML=or),kn=n(A),u(he.$$.fragment,A),A.forEach(c),Bo=n(e),u(Le.$$.fragment,e),Ro=n(e),z=a(e,"DIV",{class:!0});var U=x(z);u(Ee.$$.fragment,U),xn=n(U),zt=a(U,"P",{"data-svelte-h":!0}),p(zt)!=="svelte-13guiap"&&(zt.innerHTML=nr),wn=n(U),Ct=a(U,"P",{"data-svelte-h":!0}),p(Ct)!=="svelte-ntrhio"&&(Ct.innerHTML=sr),Sn=n(U),me=a(U,"DIV",{class:!0});var xt=x(me);u(Ve.$$.fragment,xt),$n=n(xt),qt=a(xt,"P",{"data-svelte-h":!0}),p(qt)!=="svelte-kpxj0c"&&(qt.textContent=rr),xt.forEach(c),Mn=n(U),jt=a(U,"DIV",{class:!0});var Jo=x(jt);u(De.$$.fragment,Jo),Jo.forEach(c),Fn=n(U),K=a(U,"DIV",{class:!0});var ie=x(K);u(Be.$$.fragment,ie),zn=n(ie),Nt=a(ie,"P",{"data-svelte-h":!0}),p(Nt)!=="svelte-vbfkpu"&&(Nt.textContent=ar),Cn=n(ie),Ut=a(ie,"P",{"data-svelte-h":!0}),p(Ut)!=="svelte-125uxon"&&(Ut.innerHTML=cr),ie.forEach(c),qn=n(U),ue=a(U,"DIV",{class:!0});var wt=x(ue);u(Re.$$.fragment,wt),jn=n(wt),Jt=a(wt,"P",{"data-svelte-h":!0}),p(Jt)!=="svelte-1deng2j"&&(Jt.textContent=ir),wt.forEach(c),U.forEach(c),Oo=n(e),u(Oe.$$.fragment,e),Xo=n(e),q=a(e,"DIV",{class:!0});var G=x(q);u(Xe.$$.fragment,G),Nn=n(G),It=a(G,"P",{"data-svelte-h":!0}),p(It)!=="svelte-hytttm"&&(It.textContent=dr),Un=n(G),Pt=a(G,"P",{"data-svelte-h":!0}),p(Pt)!=="svelte-9t836v"&&(Pt.textContent=lr),Jn=n(G),Wt=a(G,"P",{"data-svelte-h":!0}),p(Wt)!=="svelte-10431qd"&&(Wt.textContent=pr),In=n(G),Zt=a(G,"P",{"data-svelte-h":!0}),p(Zt)!=="svelte-ue5gbv"&&(Zt.innerHTML=hr),Pn=n(G),ee=a(G,"DIV",{class:!0});var de=x(ee);u(Ye.$$.fragment,de),Wn=n(de),Gt=a(de,"P",{"data-svelte-h":!0}),p(Gt)!=="svelte-1a6wgfx"&&(Gt.textContent=mr),Zn=n(de),Ht=a(de,"P",{"data-svelte-h":!0}),p(Ht)!=="svelte-8yitl9"&&(Ht.innerHTML=ur),de.forEach(c),G.forEach(c),Yo=n(e),u(Qe.$$.fragment,e),Qo=n(e),M=a(e,"DIV",{class:!0});var F=x(M);u(Ae.$$.fragment,F),Gn=n(F),Lt=a(F,"P",{"data-svelte-h":!0}),p(Lt)!=="svelte-1b2vd31"&&(Lt.textContent=fr),Hn=n(F),Et=a(F,"P",{"data-svelte-h":!0}),p(Et)!=="svelte-1ason94"&&(Et.innerHTML=gr),Ln=n(F),C=a(F,"DIV",{class:!0});var J=x(C);u(Ke.$$.fragment,J),En=n(J),Vt=a(J,"P",{"data-svelte-h":!0}),p(Vt)!=="svelte-ocqtg9"&&(Vt.textContent=_r),Vn=n(J),Dt=a(J,"P",{"data-svelte-h":!0}),p(Dt)!=="svelte-h2j9sw"&&(Dt.innerHTML=Tr),Dn=n(J),Bt=a(J,"P",{"data-svelte-h":!0}),p(Bt)!=="svelte-1psugse"&&(Bt.innerHTML=vr),Bn=n(J),Rt=a(J,"P",{"data-svelte-h":!0}),p(Rt)!=="svelte-bjsvki"&&(Rt.textContent=br),Rn=n(J),Ot=a(J,"UL",{"data-svelte-h":!0}),p(Ot)!=="svelte-1gayspi"&&(Ot.innerHTML=yr),On=n(J),Xt=a(J,"P",{"data-svelte-h":!0}),p(Xt)!=="svelte-ws0hzs"&&(Xt.textContent=kr),J.forEach(c),Xn=n(F),I=a(F,"DIV",{class:!0});var H=x(I);u(et.$$.fragment,H),Yn=n(H),Yt=a(H,"P",{"data-svelte-h":!0}),p(Yt)!=="svelte-1n59xk4"&&(Yt.textContent=xr),Qn=n(H),Qt=a(H,"P",{"data-svelte-h":!0}),p(Qt)!=="svelte-1ydpoay"&&(Qt.innerHTML=wr),An=n(H),At=a(H,"P",{"data-svelte-h":!0}),p(At)!=="svelte-bjsvki"&&(At.textContent=Sr),Kn=n(H),Kt=a(H,"UL",{"data-svelte-h":!0}),p(Kt)!=="svelte-25scwy"&&(Kt.innerHTML=$r),es=n(H),eo=a(H,"P",{"data-svelte-h":!0}),p(eo)!=="svelte-ws0hzs"&&(eo.textContent=Mr),H.forEach(c),ts=n(F),te=a(F,"DIV",{class:!0});var Fo=x(te);u(tt.$$.fragment,Fo),os=n(Fo),to=a(Fo,"P",{"data-svelte-h":!0}),p(to)!=="svelte-1cj8dcb"&&(to.textContent=Fr),ns=n(Fo),u(fe.$$.fragment,Fo),Fo.forEach(c),ss=n(F),oe=a(F,"DIV",{class:!0});var zo=x(oe);u(ot.$$.fragment,zo),rs=n(zo),oo=a(zo,"P",{"data-svelte-h":!0}),p(oo)!=="svelte-1fjnvpp"&&(oo.innerHTML=zr),as=n(zo),u(ge.$$.fragment,zo),zo.forEach(c),cs=n(F),_e=a(F,"DIV",{class:!0});var hn=x(_e);u(nt.$$.fragment,hn),is=n(hn),no=a(hn,"P",{"data-svelte-h":!0}),p(no)!=="svelte-njenc7"&&(no.innerHTML=Cr),hn.forEach(c),ds=n(F),Te=a(F,"DIV",{class:!0});var mn=x(Te);u(st.$$.fragment,mn),ls=n(mn),so=a(mn,"P",{"data-svelte-h":!0}),p(so)!=="svelte-f8t9ud"&&(so.innerHTML=qr),mn.forEach(c),F.forEach(c),Ao=n(e),u(rt.$$.fragment,e),Ko=n(e),P=a(e,"DIV",{class:!0});var se=x(P);u(at.$$.fragment,se),ps=n(se),ro=a(se,"P",{"data-svelte-h":!0}),p(ro)!=="svelte-bzoa48"&&(ro.textContent=jr),hs=n(se),ao=a(se,"P",{"data-svelte-h":!0}),p(ao)!=="svelte-q52n56"&&(ao.innerHTML=Nr),ms=n(se),co=a(se,"P",{"data-svelte-h":!0}),p(co)!=="svelte-hswkmf"&&(co.innerHTML=Ur),us=n(se),ne=a(se,"DIV",{class:!0});var Co=x(ne);u(ct.$$.fragment,Co),fs=n(Co),io=a(Co,"P",{"data-svelte-h":!0}),p(io)!=="svelte-u4rjue"&&(io.innerHTML=Jr),gs=n(Co),u(ve.$$.fragment,Co),Co.forEach(c),se.forEach(c),en=n(e),u(it.$$.fragment,e),tn=n(e),W=a(e,"DIV",{class:!0});var re=x(W);u(dt.$$.fragment,re),_s=n(re),lo=a(re,"P",{"data-svelte-h":!0}),p(lo)!=="svelte-128k33u"&&(lo.textContent=Ir),Ts=n(re),po=a(re,"P",{"data-svelte-h":!0}),p(po)!=="svelte-q52n56"&&(po.innerHTML=Pr),vs=n(re),ho=a(re,"P",{"data-svelte-h":!0}),p(ho)!=="svelte-hswkmf"&&(ho.innerHTML=Wr),bs=n(re),L=a(re,"DIV",{class:!0});var ae=x(L);u(lt.$$.fragment,ae),ys=n(ae),mo=a(ae,"P",{"data-svelte-h":!0}),p(mo)!=="svelte-64q3s6"&&(mo.innerHTML=Zr),ks=n(ae),u(be.$$.fragment,ae),xs=n(ae),u(ye.$$.fragment,ae),ws=n(ae),u(ke.$$.fragment,ae),ae.forEach(c),re.forEach(c),on=n(e),u(pt.$$.fragment,e),nn=n(e),j=a(e,"DIV",{class:!0});var R=x(j);u(ht.$$.fragment,R),Ss=n(R),uo=a(R,"P",{"data-svelte-h":!0}),p(uo)!=="svelte-ue58lo"&&(uo.textContent=Gr),$s=n(R),fo=a(R,"P",{"data-svelte-h":!0}),p(fo)!=="svelte-q52n56"&&(fo.innerHTML=Hr),Ms=n(R),go=a(R,"P",{"data-svelte-h":!0}),p(go)!=="svelte-hswkmf"&&(go.innerHTML=Lr),Fs=n(R),D=a(R,"DIV",{class:!0});var Ce=x(D);u(mt.$$.fragment,Ce),zs=n(Ce),_o=a(Ce,"P",{"data-svelte-h":!0}),p(_o)!=="svelte-1ogc2ry"&&(_o.innerHTML=Er),Cs=n(Ce),u(xe.$$.fragment,Ce),qs=n(Ce),u(we.$$.fragment,Ce),Ce.forEach(c),js=n(R),Se=a(R,"DIV",{class:!0});var un=x(Se);u(ut.$$.fragment,un),Ns=n(un),To=a(un,"P",{"data-svelte-h":!0}),p(To)!=="svelte-1bjnl76"&&(To.textContent=Vr),un.forEach(c),R.forEach(c),sn=n(e),u(ft.$$.fragment,e),rn=n(e),N=a(e,"DIV",{class:!0});var O=x(N);u(gt.$$.fragment,O),Us=n(O),vo=a(O,"P",{"data-svelte-h":!0}),p(vo)!=="svelte-vef7j"&&(vo.textContent=Dr),Js=n(O),bo=a(O,"P",{"data-svelte-h":!0}),p(bo)!=="svelte-q52n56"&&(bo.innerHTML=Br),Is=n(O),yo=a(O,"P",{"data-svelte-h":!0}),p(yo)!=="svelte-hswkmf"&&(yo.innerHTML=Rr),Ps=n(O),B=a(O,"DIV",{class:!0});var qe=x(B);u(_t.$$.fragment,qe),Ws=n(qe),ko=a(qe,"P",{"data-svelte-h":!0}),p(ko)!=="svelte-4otugu"&&(ko.innerHTML=Or),Zs=n(qe),u($e.$$.fragment,qe),Gs=n(qe),u(Me.$$.fragment,qe),qe.forEach(c),Hs=n(O),Fe=a(O,"DIV",{class:!0});var fn=x(Fe);u(Tt.$$.fragment,fn),Ls=n(fn),xo=a(fn,"P",{"data-svelte-h":!0}),p(xo)!=="svelte-mx15yu"&&(xo.textContent=Xr),fn.forEach(c),O.forEach(c),an=n(e),u(vt.$$.fragment,e),cn=n(e),Z=a(e,"DIV",{class:!0});var ce=x(Z);u(bt.$$.fragment,ce),Es=n(ce),wo=a(ce,"P",{"data-svelte-h":!0}),p(wo)!=="svelte-xe6dkd"&&(wo.textContent=Yr),Vs=n(ce),So=a(ce,"P",{"data-svelte-h":!0}),p(So)!=="svelte-q52n56"&&(So.innerHTML=Qr),Ds=n(ce),$o=a(ce,"P",{"data-svelte-h":!0}),p($o)!=="svelte-hswkmf"&&($o.innerHTML=Ar),Bs=n(ce),ze=a(ce,"DIV",{class:!0});var gn=x(ze);u(yt.$$.fragment,gn),Rs=n(gn),Mo=a(gn,"P",{"data-svelte-h":!0}),p(Mo)!=="svelte-qef8w5"&&(Mo.textContent=Kr),gn.forEach(c),ce.forEach(c),dn=n(e),u(kt.$$.fragment,e),ln=n(e),Uo=a(e,"P",{}),x(Uo).forEach(c),this.h()},h(){k(s,"name","hf:doc:metadata"),k(s,"content",va),k(le,"class","flex flex-wrap space-x-1"),k(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(jt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(_e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,i){t(document.head,s),h(e,b,i),h(e,l,i),h(e,v,i),h(e,y,i),h(e,$,i),f(je,e,i),h(e,Io,i),h(e,le,i),h(e,Po,i),f(Ne,e,i),h(e,Wo,i),h(e,Ue,i),h(e,Zo,i),h(e,Je,i),h(e,Go,i),h(e,Ie,i),h(e,Ho,i),h(e,Pe,i),h(e,Lo,i),f(We,e,i),h(e,Eo,i),h(e,E,i),f(Ze,E,null),t(E,_n),t(E,St),t(E,Tn),t(E,$t),t(E,vn),f(pe,E,null),h(e,Vo,i),f(Ge,e,i),h(e,Do,i),h(e,V,i),f(He,V,null),t(V,bn),t(V,Mt),t(V,yn),t(V,Ft),t(V,kn),f(he,V,null),h(e,Bo,i),f(Le,e,i),h(e,Ro,i),h(e,z,i),f(Ee,z,null),t(z,xn),t(z,zt),t(z,wn),t(z,Ct),t(z,Sn),t(z,me),f(Ve,me,null),t(me,$n),t(me,qt),t(z,Mn),t(z,jt),f(De,jt,null),t(z,Fn),t(z,K),f(Be,K,null),t(K,zn),t(K,Nt),t(K,Cn),t(K,Ut),t(z,qn),t(z,ue),f(Re,ue,null),t(ue,jn),t(ue,Jt),h(e,Oo,i),f(Oe,e,i),h(e,Xo,i),h(e,q,i),f(Xe,q,null),t(q,Nn),t(q,It),t(q,Un),t(q,Pt),t(q,Jn),t(q,Wt),t(q,In),t(q,Zt),t(q,Pn),t(q,ee),f(Ye,ee,null),t(ee,Wn),t(ee,Gt),t(ee,Zn),t(ee,Ht),h(e,Yo,i),f(Qe,e,i),h(e,Qo,i),h(e,M,i),f(Ae,M,null),t(M,Gn),t(M,Lt),t(M,Hn),t(M,Et),t(M,Ln),t(M,C),f(Ke,C,null),t(C,En),t(C,Vt),t(C,Vn),t(C,Dt),t(C,Dn),t(C,Bt),t(C,Bn),t(C,Rt),t(C,Rn),t(C,Ot),t(C,On),t(C,Xt),t(M,Xn),t(M,I),f(et,I,null),t(I,Yn),t(I,Yt),t(I,Qn),t(I,Qt),t(I,An),t(I,At),t(I,Kn),t(I,Kt),t(I,es),t(I,eo),t(M,ts),t(M,te),f(tt,te,null),t(te,os),t(te,to),t(te,ns),f(fe,te,null),t(M,ss),t(M,oe),f(ot,oe,null),t(oe,rs),t(oe,oo),t(oe,as),f(ge,oe,null),t(M,cs),t(M,_e),f(nt,_e,null),t(_e,is),t(_e,no),t(M,ds),t(M,Te),f(st,Te,null),t(Te,ls),t(Te,so),h(e,Ao,i),f(rt,e,i),h(e,Ko,i),h(e,P,i),f(at,P,null),t(P,ps),t(P,ro),t(P,hs),t(P,ao),t(P,ms),t(P,co),t(P,us),t(P,ne),f(ct,ne,null),t(ne,fs),t(ne,io),t(ne,gs),f(ve,ne,null),h(e,en,i),f(it,e,i),h(e,tn,i),h(e,W,i),f(dt,W,null),t(W,_s),t(W,lo),t(W,Ts),t(W,po),t(W,vs),t(W,ho),t(W,bs),t(W,L),f(lt,L,null),t(L,ys),t(L,mo),t(L,ks),f(be,L,null),t(L,xs),f(ye,L,null),t(L,ws),f(ke,L,null),h(e,on,i),f(pt,e,i),h(e,nn,i),h(e,j,i),f(ht,j,null),t(j,Ss),t(j,uo),t(j,$s),t(j,fo),t(j,Ms),t(j,go),t(j,Fs),t(j,D),f(mt,D,null),t(D,zs),t(D,_o),t(D,Cs),f(xe,D,null),t(D,qs),f(we,D,null),t(j,js),t(j,Se),f(ut,Se,null),t(Se,Ns),t(Se,To),h(e,sn,i),f(ft,e,i),h(e,rn,i),h(e,N,i),f(gt,N,null),t(N,Us),t(N,vo),t(N,Js),t(N,bo),t(N,Is),t(N,yo),t(N,Ps),t(N,B),f(_t,B,null),t(B,Ws),t(B,ko),t(B,Zs),f($e,B,null),t(B,Gs),f(Me,B,null),t(N,Hs),t(N,Fe),f(Tt,Fe,null),t(Fe,Ls),t(Fe,xo),h(e,an,i),f(vt,e,i),h(e,cn,i),h(e,Z,i),f(bt,Z,null),t(Z,Es),t(Z,wo),t(Z,Vs),t(Z,So),t(Z,Ds),t(Z,$o),t(Z,Bs),t(Z,ze),f(yt,ze,null),t(ze,Rs),t(ze,Mo),h(e,dn,i),f(kt,e,i),h(e,ln,i),h(e,Uo,i),pn=!0},p(e,[i]){const Q={};i&2&&(Q.$$scope={dirty:i,ctx:e}),pe.$set(Q);const A={};i&2&&(A.$$scope={dirty:i,ctx:e}),he.$set(A);const U={};i&2&&(U.$$scope={dirty:i,ctx:e}),fe.$set(U);const xt={};i&2&&(xt.$$scope={dirty:i,ctx:e}),ge.$set(xt);const Jo={};i&2&&(Jo.$$scope={dirty:i,ctx:e}),ve.$set(Jo);const ie={};i&2&&(ie.$$scope={dirty:i,ctx:e}),be.$set(ie);const wt={};i&2&&(wt.$$scope={dirty:i,ctx:e}),ye.$set(wt);const G={};i&2&&(G.$$scope={dirty:i,ctx:e}),ke.$set(G);const de={};i&2&&(de.$$scope={dirty:i,ctx:e}),xe.$set(de);const F={};i&2&&(F.$$scope={dirty:i,ctx:e}),we.$set(F);const J={};i&2&&(J.$$scope={dirty:i,ctx:e}),$e.$set(J);const H={};i&2&&(H.$$scope={dirty:i,ctx:e}),Me.$set(H)},i(e){pn||(g(je.$$.fragment,e),g(Ne.$$.fragment,e),g(We.$$.fragment,e),g(Ze.$$.fragment,e),g(pe.$$.fragment,e),g(Ge.$$.fragment,e),g(He.$$.fragment,e),g(he.$$.fragment,e),g(Le.$$.fragment,e),g(Ee.$$.fragment,e),g(Ve.$$.fragment,e),g(De.$$.fragment,e),g(Be.$$.fragment,e),g(Re.$$.fragment,e),g(Oe.$$.fragment,e),g(Xe.$$.fragment,e),g(Ye.$$.fragment,e),g(Qe.$$.fragment,e),g(Ae.$$.fragment,e),g(Ke.$$.fragment,e),g(et.$$.fragment,e),g(tt.$$.fragment,e),g(fe.$$.fragment,e),g(ot.$$.fragment,e),g(ge.$$.fragment,e),g(nt.$$.fragment,e),g(st.$$.fragment,e),g(rt.$$.fragment,e),g(at.$$.fragment,e),g(ct.$$.fragment,e),g(ve.$$.fragment,e),g(it.$$.fragment,e),g(dt.$$.fragment,e),g(lt.$$.fragment,e),g(be.$$.fragment,e),g(ye.$$.fragment,e),g(ke.$$.fragment,e),g(pt.$$.fragment,e),g(ht.$$.fragment,e),g(mt.$$.fragment,e),g(xe.$$.fragment,e),g(we.$$.fragment,e),g(ut.$$.fragment,e),g(ft.$$.fragment,e),g(gt.$$.fragment,e),g(_t.$$.fragment,e),g($e.$$.fragment,e),g(Me.$$.fragment,e),g(Tt.$$.fragment,e),g(vt.$$.fragment,e),g(bt.$$.fragment,e),g(yt.$$.fragment,e),g(kt.$$.fragment,e),pn=!0)},o(e){_(je.$$.fragment,e),_(Ne.$$.fragment,e),_(We.$$.fragment,e),_(Ze.$$.fragment,e),_(pe.$$.fragment,e),_(Ge.$$.fragment,e),_(He.$$.fragment,e),_(he.$$.fragment,e),_(Le.$$.fragment,e),_(Ee.$$.fragment,e),_(Ve.$$.fragment,e),_(De.$$.fragment,e),_(Be.$$.fragment,e),_(Re.$$.fragment,e),_(Oe.$$.fragment,e),_(Xe.$$.fragment,e),_(Ye.$$.fragment,e),_(Qe.$$.fragment,e),_(Ae.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(fe.$$.fragment,e),_(ot.$$.fragment,e),_(ge.$$.fragment,e),_(nt.$$.fragment,e),_(st.$$.fragment,e),_(rt.$$.fragment,e),_(at.$$.fragment,e),_(ct.$$.fragment,e),_(ve.$$.fragment,e),_(it.$$.fragment,e),_(dt.$$.fragment,e),_(lt.$$.fragment,e),_(be.$$.fragment,e),_(ye.$$.fragment,e),_(ke.$$.fragment,e),_(pt.$$.fragment,e),_(ht.$$.fragment,e),_(mt.$$.fragment,e),_(xe.$$.fragment,e),_(we.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(gt.$$.fragment,e),_(_t.$$.fragment,e),_($e.$$.fragment,e),_(Me.$$.fragment,e),_(Tt.$$.fragment,e),_(vt.$$.fragment,e),_(bt.$$.fragment,e),_(yt.$$.fragment,e),_(kt.$$.fragment,e),pn=!1},d(e){e&&(c(b),c(l),c(v),c(y),c($),c(Io),c(le),c(Po),c(Wo),c(Ue),c(Zo),c(Je),c(Go),c(Ie),c(Ho),c(Pe),c(Lo),c(Eo),c(E),c(Vo),c(Do),c(V),c(Bo),c(Ro),c(z),c(Oo),c(Xo),c(q),c(Yo),c(Qo),c(M),c(Ao),c(Ko),c(P),c(en),c(tn),c(W),c(on),c(nn),c(j),c(sn),c(rn),c(N),c(an),c(cn),c(Z),c(dn),c(ln),c(Uo)),c(s),T(je,e),T(Ne,e),T(We,e),T(Ze),T(pe),T(Ge,e),T(He),T(he),T(Le,e),T(Ee),T(Ve),T(De),T(Be),T(Re),T(Oe,e),T(Xe),T(Ye),T(Qe,e),T(Ae),T(Ke),T(et),T(tt),T(fe),T(ot),T(ge),T(nt),T(st),T(rt,e),T(at),T(ct),T(ve),T(it,e),T(dt),T(lt),T(be),T(ye),T(ke),T(pt,e),T(ht),T(mt),T(xe),T(we),T(ut),T(ft,e),T(gt),T(_t),T($e),T(Me),T(Tt),T(vt,e),T(bt),T(yt),T(kt,e)}}}const va='{"title":"SpeechT5","local":"speecht5","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"SpeechT5Config","local":"transformers.SpeechT5Config","sections":[],"depth":2},{"title":"SpeechT5HifiGanConfig","local":"transformers.SpeechT5HifiGanConfig","sections":[],"depth":2},{"title":"SpeechT5Tokenizer","local":"transformers.SpeechT5Tokenizer","sections":[],"depth":2},{"title":"SpeechT5FeatureExtractor","local":"transformers.SpeechT5FeatureExtractor","sections":[],"depth":2},{"title":"SpeechT5Processor","local":"transformers.SpeechT5Processor","sections":[],"depth":2},{"title":"SpeechT5Model","local":"transformers.SpeechT5Model","sections":[],"depth":2},{"title":"SpeechT5ForSpeechToText","local":"transformers.SpeechT5ForSpeechToText","sections":[],"depth":2},{"title":"SpeechT5ForTextToSpeech","local":"transformers.SpeechT5ForTextToSpeech","sections":[],"depth":2},{"title":"SpeechT5ForSpeechToSpeech","local":"transformers.SpeechT5ForSpeechToSpeech","sections":[],"depth":2},{"title":"SpeechT5HifiGan","local":"transformers.SpeechT5HifiGan","sections":[],"depth":2}],"depth":1}';function ba(S){return ta(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Fa extends oa{constructor(s){super(),na(this,s,ba,Ta,ea,{})}}export{Fa as component};
