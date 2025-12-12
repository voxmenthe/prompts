import{s as Xn,o as Vn,n as L}from"../chunks/scheduler.18a86fab.js";import{S as Dn,i as Sn,g as c,s as o,r as p,A as Hn,h as d,f as n,c as a,j as q,x as f,u as m,k as U,y as r,a as i,v as h,d as u,t as M,w as g}from"../chunks/index.98837b22.js";import{T as xt}from"../chunks/Tip.77304350.js";import{D as E}from"../chunks/Docstring.a1ef7999.js";import{C as Y}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as qt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Z,E as Yn}from"../chunks/getInferenceSnippets.06c2775f.js";function Ln(j){let s,y;return s=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyMENzbUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBDc21Db25maWclMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwQ3NtQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMEFtb2RlbCUyMCUzRCUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbihjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmForConditionalGeneration, CsmConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a CsmConfig</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = CsmConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CsmForConditionalGeneration(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){p(s.$$.fragment)},l(l){m(s.$$.fragment,l)},m(l,_){h(s,l,_),y=!0},p:L,i(l){y||(u(s.$$.fragment,l),y=!0)},o(l){M(s.$$.fragment,l),y=!1},d(l){g(s,l)}}}function Pn(j){let s,y;return s=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENzbURlcHRoRGVjb2RlciUyQyUyMENzbURlcHRoRGVjb2RlckNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBDc21EZXB0aERlY29kZXIlMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwQ3NtRGVwdGhEZWNvZGVyQ29uZmlnKCklMEFtb2RlbCUyMCUzRCUyMENzbURlcHRoRGVjb2Rlck1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmDepthDecoder, CsmDepthDecoderConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a CsmDepthDecoder</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = CsmDepthDecoderConfig()
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CsmDepthDecoderModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){p(s.$$.fragment)},l(l){m(s.$$.fragment,l)},m(l,_){h(s,l,_),y=!0},p:L,i(l){y||(u(s.$$.fragment,l),y=!0)},o(l){M(s.$$.fragment,l),y=!1},d(l){g(s,l)}}}function Kn(j){let s,y;return s=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENzbVByb2Nlc3NvciUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQSUwQWRzJTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZkYWlseXRhbGstZHVtbXklMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQWF1ZGlvJTIwJTNEJTIwZHMlNUIwJTVEJTVCJTIyYXVkaW8lMjIlNUQlNUIlMjJhcnJheSUyMiU1RCUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMENzbVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyc2VzYW1lJTJGY3NtLTFiJTIyKSUwQSUwQXByb2Nlc3NvciglMEElMjAlMjAlMjAlMjB0ZXh0JTNEJTVCJTIyJTNDJTdDYmVnaW5fb2ZfdGV4dCU3QyUzRSU1QjAlNURXaGF0JTIwYXJlJTIweW91JTIwd29ya2luZyUyMG9uJTNGJTNDJTdDZW5kX29mX3RleHQlN0MlM0UlM0MlN0NBVURJTyU3QyUzRSUzQyU3Q2F1ZGlvX2VvcyU3QyUzRSUzQyU3Q2JlZ2luX29mX3RleHQlN0MlM0UlNUIxJTVESSdtJTIwZmlndXJpbmclMjBvdXQlMjBteSUyMGJ1ZGdldC4lM0MlN0NlbmRfb2ZfdGV4dCU3QyUzRSUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGF1ZGlvJTNEYXVkaW8lMkMlMEElMjAlMjAlMjAlMjB0ZXh0X2t3YXJncyUyMCUzRCUyMCU3QiUyMnBhZGRpbmclMjIlM0ElMjBGYWxzZSU3RCUyQyUwQSUyMCUyMCUyMCUyMGF1ZGlvX2t3YXJncyUyMCUzRCUyMCU3QiUyMnNhbXBsaW5nX3JhdGUlMjIlM0ElMjAxNjAwMCU3RCUyQyUwQSUyMCUyMCUyMCUyMGNvbW1vbl9rd2FyZ3MlMjAlM0QlMjAlN0IlMjJyZXR1cm5fdGVuc29ycyUyMiUzQSUyMCUyMnB0JTIyJTdEJTJDJTBBKSUwQSUyMyUyMHRoaXMlMjBzaG91bGQlMjBlcnJvciUyMG91dCUyMGJlY2F1c2UlMjBFbmNvZGVjRmVhdHVyZUV4dHJhY3RvciUyMGV4cGVjdHMlMjBhJTIwMjRrSHolMjBhdWRpbyUyMCUzQSk=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmProcessor
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
audio = ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]

processor = CsmProcessor.from_pretrained(<span class="hljs-string">&quot;sesame/csm-1b&quot;</span>)

processor(
    text=[<span class="hljs-string">&quot;&lt;|begin_of_text|&gt;[0]What are you working on?&lt;|end_of_text|&gt;&lt;|AUDIO|&gt;&lt;|audio_eos|&gt;&lt;|begin_of_text|&gt;[1]I&#x27;m figuring out my budget.&lt;|end_of_text|&gt;&quot;</span>],
    audio=audio,
    text_kwargs = {<span class="hljs-string">&quot;padding&quot;</span>: <span class="hljs-literal">False</span>},
    audio_kwargs = {<span class="hljs-string">&quot;sampling_rate&quot;</span>: <span class="hljs-number">16000</span>},
    common_kwargs = {<span class="hljs-string">&quot;return_tensors&quot;</span>: <span class="hljs-string">&quot;pt&quot;</span>},
)
<span class="hljs-comment"># this should error out because EncodecFeatureExtractor expects a 24kHz audio :)</span>`,wrap:!1}}),{c(){p(s.$$.fragment)},l(l){m(s.$$.fragment,l)},m(l,_){h(s,l,_),y=!0},p:L,i(l){y||(u(s.$$.fragment,l),y=!0)},o(l){M(s.$$.fragment,l),y=!1},d(l){g(s,l)}}}function On(j){let s,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=c("p"),s.innerHTML=y},l(l){s=d(l,"P",{"data-svelte-h":!0}),f(s)!=="svelte-fincs2"&&(s.innerHTML=y)},m(l,_){i(l,s,_)},p:L,d(l){l&&n(s)}}}function eo(j){let s,y="Example:",l,_,b;return _=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQ3NtRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTJDJTIwQXV0b1Byb2Nlc3NvciUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUyQyUyMEF1ZGlvJTBBJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJzZXNhbWUlMkZjc20tMWIlMjIlMEF0b3JjaF9kZXZpY2UlMjAlM0QlMjAlMjJjdWRhJTIyJTIwaWYlMjB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpJTIwZWxzZSUyMCUyMmNwdSUyMiUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkKSUwQSUwQWRzJTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZkYWlseXRhbGstZHVtbXklMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQSUyMyUyMGVuc3VyZSUyMHRoZSUyMGF1ZGlvJTIwaXMlMjAyNGtIeiUwQWRzJTIwJTNEJTIwZHMuY2FzdF9jb2x1bW4oJTIyYXVkaW8lMjIlMkMlMjBBdWRpbyhzYW1wbGluZ19yYXRlJTNEMjQwMDApKSUwQSUwQWNvbnZlcnNhdGlvbiUyMCUzRCUyMCU1QiU1RCUwQSUyMyUyMHByZXBhcmUlMjBhJTIwY29udmVyc2F0aW9uJTIwd2l0aCUyMHRleHQlMjBhbmQlMjBjb3JyZXNwb25kaW5nJTIwYXVkaW8lMEFmb3IlMjB0ZXh0JTJDJTIwYXVkaW8lMkMlMjBzcGVha2VyX2lkJTIwaW4lMjB6aXAoZHMlNUIlM0E0JTVEJTVCJTIydGV4dCUyMiU1RCUyQyUyMGRzJTVCJTNBNCU1RCU1QiUyMmF1ZGlvJTIyJTVEJTJDJTIwZHMlNUIlM0E0JTVEJTVCJTIyc3BlYWtlcl9pZCUyMiU1RCklM0ElMEElMjAlMjAlMjAlMjBjb252ZXJzYXRpb24uYXBwZW5kKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjBmJTIyJTdCc3BlYWtlcl9pZCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjB0ZXh0JTdEJTJDJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmF1ZGlvJTIyJTJDJTIwJTIycGF0aCUyMiUzQSUyMGF1ZGlvJTVCJTIyYXJyYXklMjIlNUQlN0QlNUQlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMEElMjAlMjAlMjAlMjApJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwY29udmVyc2F0aW9uJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwb3V0cHV0X2xhYmVscyUzRFRydWUlMkMlMEEpLnRvKHRvcmNoX2RldmljZSklMEElMEFtb2RlbCUyMCUzRCUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQlMkMlMjBkZXZpY2VfbWFwJTNEdG9yY2hfZGV2aWNlKSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQW91dHB1dC5sb3NzLmJhY2t3YXJkKCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmForConditionalGeneration, AutoProcessor
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio

<span class="hljs-meta">&gt;&gt;&gt; </span>model_id = <span class="hljs-string">&quot;sesame/csm-1b&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>torch_device = <span class="hljs-string">&quot;cuda&quot;</span> <span class="hljs-keyword">if</span> torch.cuda.is_available() <span class="hljs-keyword">else</span> <span class="hljs-string">&quot;cpu&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(model_id)

<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># ensure the audio is 24kHz</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">24000</span>))

<span class="hljs-meta">&gt;&gt;&gt; </span>conversation = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare a conversation with text and corresponding audio</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> text, audio, speaker_id <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;text&quot;</span>], ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;audio&quot;</span>], ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;speaker_id&quot;</span>]):
<span class="hljs-meta">... </span>    conversation.append(
<span class="hljs-meta">... </span>        {
<span class="hljs-meta">... </span>            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{speaker_id}</span>&quot;</span>,
<span class="hljs-meta">... </span>            <span class="hljs-string">&quot;content&quot;</span>: [{<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: text}, {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: audio[<span class="hljs-string">&quot;array&quot;</span>]}],
<span class="hljs-meta">... </span>        }
<span class="hljs-meta">... </span>    )

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor.apply_chat_template(
<span class="hljs-meta">... </span>    conversation,
<span class="hljs-meta">... </span>    tokenize=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    return_dict=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    output_labels=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>).to(torch_device)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)
<span class="hljs-meta">&gt;&gt;&gt; </span>output = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>output.loss.backward()`,wrap:!1}}),{c(){s=c("p"),s.textContent=y,l=o(),p(_.$$.fragment)},l(T){s=d(T,"P",{"data-svelte-h":!0}),f(s)!=="svelte-11lpom8"&&(s.textContent=y),l=a(T),m(_.$$.fragment,T)},m(T,W){i(T,s,W),i(T,l,W),h(_,T,W),b=!0},p:L,i(T){b||(u(_.$$.fragment,T),b=!0)},o(T){M(_.$$.fragment,T),b=!1},d(T){T&&(n(s),n(l)),g(_,T)}}}function to(j){let s,y=`Most generation-controlling parameters are set in <code>generation_config</code> which, if not passed, will be set to the
model’s default generation configuration. You can override any <code>generation_config</code> by passing the corresponding
parameters to generate(), e.g. <code>.generate(inputs, do_sample=True)</code>.`;return{c(){s=c("p"),s.innerHTML=y},l(l){s=d(l,"P",{"data-svelte-h":!0}),f(s)!=="svelte-1qx4kgv"&&(s.innerHTML=y)},m(l,_){i(l,s,_)},p:L,d(l){l&&n(s)}}}function so(j){let s,y="Example:",l,_,b;return _=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENzbVByb2Nlc3NvciUyQyUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUyQyUyMEF1ZGlvJTBBJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJzZXNhbWUlMkZjc20tMWIlMjIlMEF0b3JjaF9kZXZpY2UlMjAlM0QlMjAlMjJjdWRhJTIyJTIwaWYlMjB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpJTIwZWxzZSUyMCUyMmNwdSUyMiUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkKSUwQSUwQWRzJTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZkYWlseXRhbGstZHVtbXklMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQSUyMyUyMGVuc3VyZSUyMHRoZSUyMGF1ZGlvJTIwaXMlMjAyNGtIeiUwQWRzJTIwJTNEJTIwZHMuY2FzdF9jb2x1bW4oJTIyYXVkaW8lMjIlMkMlMjBBdWRpbyhzYW1wbGluZ19yYXRlJTNEMjQwMDApKSUwQSUwQWNvbnZlcnNhdGlvbiUyMCUzRCUyMCU1QiU1RCUwQSUyMyUyMHByZXBhcmUlMjBhJTIwY29udmVyc2F0aW9uJTIwd2l0aCUyMHRleHQlMjBhbmQlMjBjb3JyZXNwb25kaW5nJTIwYXVkaW8lMEFmb3IlMjB0ZXh0JTJDJTIwYXVkaW8lMkMlMjBzcGVha2VyX2lkJTIwaW4lMjB6aXAoZHMlNUIlM0E0JTVEJTVCJTIydGV4dCUyMiU1RCUyQyUyMGRzJTVCJTNBNCU1RCU1QiUyMmF1ZGlvJTIyJTVEJTJDJTIwZHMlNUIlM0E0JTVEJTVCJTIyc3BlYWtlcl9pZCUyMiU1RCklM0ElMEElMjAlMjAlMjAlMjBjb252ZXJzYXRpb24uYXBwZW5kKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjBmJTIyJTdCc3BlYWtlcl9pZCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjB0ZXh0JTdEJTJDJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmF1ZGlvJTIyJTJDJTIwJTIycGF0aCUyMiUzQSUyMGF1ZGlvJTVCJTIyYXJyYXklMjIlNUQlN0QlNUQlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMEElMjAlMjAlMjAlMjApJTBBJTBBJTIzJTIwdGV4dCUyMHByb21wdCUwQWNvbnZlcnNhdGlvbi5hcHBlbmQoJTdCJTIycm9sZSUyMiUzQSUyMGYlMjIlN0JkcyU1QjQlNUQlNUInc3BlYWtlcl9pZCclNUQlN0QlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwZHMlNUI0JTVEJTVCJTIydGV4dCUyMiU1RCU3RCU1RCU3RCklMEElMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjBjb252ZXJzYXRpb24lMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMEEpLnRvKHRvcmNoX2RldmljZSklMEElMEFtb2RlbCUyMCUzRCUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQlMkMlMjBkZXZpY2VfbWFwJTNEdG9yY2hfZGV2aWNlKSUwQWF1ZGlvJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBvdXRwdXRfYXVkaW8lM0RUcnVlKSUwQXByb2Nlc3Nvci5zYXZlX2F1ZGlvKGF1ZGlvJTJDJTIwJTIyb3V0cHV0LndhdiUyMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmProcessor, CsmForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio

<span class="hljs-meta">&gt;&gt;&gt; </span>model_id = <span class="hljs-string">&quot;sesame/csm-1b&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>torch_device = <span class="hljs-string">&quot;cuda&quot;</span> <span class="hljs-keyword">if</span> torch.cuda.is_available() <span class="hljs-keyword">else</span> <span class="hljs-string">&quot;cpu&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(model_id)

<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># ensure the audio is 24kHz</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">24000</span>))

<span class="hljs-meta">&gt;&gt;&gt; </span>conversation = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare a conversation with text and corresponding audio</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> text, audio, speaker_id <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;text&quot;</span>], ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;audio&quot;</span>], ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;speaker_id&quot;</span>]):
<span class="hljs-meta">... </span>    conversation.append(
<span class="hljs-meta">... </span>        {
<span class="hljs-meta">... </span>            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{speaker_id}</span>&quot;</span>,
<span class="hljs-meta">... </span>            <span class="hljs-string">&quot;content&quot;</span>: [{<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: text}, {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: audio[<span class="hljs-string">&quot;array&quot;</span>]}],
<span class="hljs-meta">... </span>        }
<span class="hljs-meta">... </span>    )

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># text prompt</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>conversation.append({<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">4</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [{<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">4</span>][<span class="hljs-string">&quot;text&quot;</span>]}]})

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor.apply_chat_template(
<span class="hljs-meta">... </span>    conversation,
<span class="hljs-meta">... </span>    tokenize=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    return_dict=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>).to(torch_device)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)
<span class="hljs-meta">&gt;&gt;&gt; </span>audio = model.generate(**inputs, output_audio=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.save_audio(audio, <span class="hljs-string">&quot;output.wav&quot;</span>)`,wrap:!1}}),{c(){s=c("p"),s.textContent=y,l=o(),p(_.$$.fragment)},l(T){s=d(T,"P",{"data-svelte-h":!0}),f(s)!=="svelte-11lpom8"&&(s.textContent=y),l=a(T),m(_.$$.fragment,T)},m(T,W){i(T,s,W),i(T,l,W),h(_,T,W),b=!0},p:L,i(T){b||(u(_.$$.fragment,T),b=!0)},o(T){M(_.$$.fragment,T),b=!1},d(T){T&&(n(s),n(l)),g(_,T)}}}function no(j){let s,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=c("p"),s.innerHTML=y},l(l){s=d(l,"P",{"data-svelte-h":!0}),f(s)!=="svelte-fincs2"&&(s.innerHTML=y)},m(l,_){i(l,s,_)},p:L,d(l){l&&n(s)}}}function oo(j){let s,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=c("p"),s.innerHTML=y},l(l){s=d(l,"P",{"data-svelte-h":!0}),f(s)!=="svelte-fincs2"&&(s.innerHTML=y)},m(l,_){i(l,s,_)},p:L,d(l){l&&n(s)}}}function ao(j){let s,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=c("p"),s.innerHTML=y},l(l){s=d(l,"P",{"data-svelte-h":!0}),f(s)!=="svelte-fincs2"&&(s.innerHTML=y)},m(l,_){i(l,s,_)},p:L,d(l){l&&n(s)}}}function lo(j){let s,y,l,_,b,T="<em>This model was released on 2025-02-27 and added to Hugging Face Transformers on 2025-05-07.</em>",W,pe,Zt,me,Bt,he,ln='The Conversational Speech Model (CSM) is the first open-source contextual text-to-speech model <a href="https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice" rel="nofollow">released by Sesame</a>. It is designed to generate natural-sounding speech with or without conversational context. This context typically consists of multi-turn dialogue between speakers, represented as sequences of text and corresponding spoken audio.',At,ue,rn=`<strong>Model Architecture:</strong>
CSM is composed of two LLaMA-style auto-regressive transformer decoders: a backbone decoder that predicts the first codebook token and a depth decoder that generates the remaining tokens. It uses the pretrained codec model <a href="./mimi">Mimi</a>, introduced by Kyutai, to encode speech into discrete codebook tokens and decode them back into audio.`,Gt,Me,cn='The original csm-1b checkpoint is available under the <a href="https://huggingface.co/sesame/csm-1b" rel="nofollow">Sesame</a> organization on Hugging Face.',Qt,K,dn='<img src="https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/csm_architecture.png"/>',Rt,ge,Nt,fe,zt,ye,pn="CSM can be used to simply generate speech from a text prompt:",Et,_e,Wt,Te,Ft,je,mn="CSM can be used to generate speech given a conversation, allowing consistency in the voices and content-aware generation:",$t,be,Xt,Je,Vt,Ue,hn="CSM supports batched inference!",Dt,we,St,Ce,Ht,Ie,un="CSM supports full-graph compilation with CUDA graphs!",Yt,ve,Lt,ke,Pt,xe,Mn="CSM Transformers integration supports training!",Kt,qe,Ot,Ze,gn=`This model was contributed by <a href="https://huggingface.co/eustlb" rel="nofollow">Eustache Le Bihan</a>.
The original code can be found <a href="https://github.com/SesameAILabs/csm" rel="nofollow">here</a>.`,es,Be,ts,w,Ae,ys,tt,fn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmForConditionalGeneration">CsmForConditionalGeneration</a>. It is used to instantiate an CSM
model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the csm-1b.`,_s,st,yn='e.g. <a href="https://huggingface.co/sesame/csm-1b" rel="nofollow">sesame/csm-1b</a>',Ts,nt,_n=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,js,O,ss,Ge,ns,C,Qe,bs,ot,Tn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderModel">CsmDepthDecoderModel</a>. It is used to instantiate an CSM depth decoder
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
a similar configuration to that of the csm-1b.`,Js,at,jn='e.g. <a href="https://huggingface.co/sesame/csm-1b" rel="nofollow">sesame/csm-1b</a>',Us,lt,bn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,ws,ee,os,Re,as,te,Jn='<img src="https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/fig1.jpg"/>',ls,A,Ne,Cs,rt,Un=`Constructs a Csm processor which wraps <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor">EncodecFeatureExtractor</a> and
<code>PretrainedTokenizerFast</code> into a single processor that inherits both the audio feature extraction and
tokenizer functionalities. See the <a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmProcessor.__call__"><strong>call</strong>()</a> for more
information.
The preferred way of passing kwargs is as a dictionary per modality, see usage example below.`,Is,se,vs,ne,ze,ks,it,wn=`Main method to prepare text(s) and audio to be fed as input to the model. This method forwards the <code>text</code>
arguments to PreTrainedTokenizerFast’s <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__"><strong>call</strong>()</a> to encode
the text. To prepare the audio, this method forwards the <code>audio</code> arguments to
EncodecFeatureExtractor’s <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor.__call__"><strong>call</strong>()</a>. Please refer
to the docstring of the above two methods for more information.`,rs,Ee,is,J,We,xs,ct,Cn="The Csm model consists of two llama-like auto-regressive transformer models: a backbone model that predicts the first codebook token and a depth decoder that predicts the other codebook tokens.",qs,dt,In=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Zs,pt,vn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bs,F,Fe,As,mt,kn='The <a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmForConditionalGeneration">CsmForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Gs,oe,Qs,ae,Rs,B,$e,Ns,ht,xn=`This method overrides <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate">generate()</a> to match the specifics of the Csm model.
Indeed, Csm model requires a custom generation sampling step:`,zs,ut,qn="<li>Infer the backbone model to sample the first codebook token</li> <li>Call generate on the depth decoder with the first codebook token as <code>input_ids</code> to sample the next codebook tokens</li> <li>Use these generated codebook tokens as <code>input_ids</code> to sample the next first codebook token using the backbone model</li> <li>Repeat until stopping criteria is met</li>",Es,le,Ws,re,cs,Xe,ds,I,Ve,Fs,Mt,Zn=`The CsmDepthDecoder Model transformer, with a <code>CsmCodebooksHead</code> on top,
which can be seen a position-specific language modeling head, allowing to use a different linear layer for each codebook
(e.g. position 0 is the first codebook and uses the first codebook head, etc.)`,$s,gt,Bn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Xs,ft,An=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Vs,V,De,Ds,yt,Gn='The <a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderForCausalLM">CsmDepthDecoderForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ss,ie,ps,Se,ms,v,He,Hs,_t,Qn="The bare Csm Model outputting raw hidden-states without any specific head on top.",Ys,Tt,Rn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ls,jt,Nn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ps,D,Ye,Ks,bt,zn='The <a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderModel">CsmDepthDecoderModel</a> forward method, overrides the <code>__call__</code> special method.',Os,ce,hs,Le,us,k,Pe,en,Jt,En="The bare Csm Model outputting raw hidden-states without any specific head on top.",tn,Ut,Wn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,sn,wt,Fn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,nn,S,Ke,on,Ct,$n='The <a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmBackboneModel">CsmBackboneModel</a> forward method, overrides the <code>__call__</code> special method.',an,de,Ms,Oe,gs,kt,fs;return pe=new Z({props:{title:"Csm",local:"csm",headingTag:"h1"}}),me=new Z({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ge=new Z({props:{title:"Usage Tips",local:"usage-tips",headingTag:"h2"}}),fe=new Z({props:{title:"Without Conversational Context",local:"without-conversational-context",headingTag:"h3"}}),_e=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQ3NtRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTJDJTIwQXV0b1Byb2Nlc3NvciUyQyUyMGluZmVyX2RldmljZSUwQSUwQW1vZGVsX2lkJTIwJTNEJTIwJTIyc2VzYW1lJTJGY3NtLTFiJTIyJTBBZGV2aWNlJTIwJTNEJTIwaW5mZXJfZGV2aWNlKCklMEElMEElMjMlMjBsb2FkJTIwdGhlJTIwbW9kZWwlMjBhbmQlMjB0aGUlMjBwcm9jZXNzb3IlMEFwcm9jZXNzb3IlMjAlM0QlMjBBdXRvUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZChtb2RlbF9pZCklMEFtb2RlbCUyMCUzRCUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQlMkMlMjBkZXZpY2VfbWFwJTNEZGV2aWNlKSUwQSUwQSUyMyUyMHByZXBhcmUlMjB0aGUlMjBpbnB1dHMlMEF0ZXh0JTIwJTNEJTIwJTIyJTVCMCU1RFRoZSUyMHBhc3QlMjBpcyUyMGp1c3QlMjBhJTIwc3RvcnklMjB3ZSUyMHRlbGwlMjBvdXJzZWx2ZXMuJTIyJTIwJTIzJTIwJTYwJTVCMCU1RCU2MCUyMGZvciUyMHNwZWFrZXIlMjBpZCUyMDAlMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IodGV4dCUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzRFRydWUpLnRvKGRldmljZSklMEElMEElMjMlMjBhbm90aGVyJTIwZXF1aXZhbGVudCUyMHdheSUyMHRvJTIwcHJlcGFyZSUyMHRoZSUyMGlucHV0cyUwQWNvbnZlcnNhdGlvbiUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjIwJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMlRoZSUyMHBhc3QlMjBpcyUyMGp1c3QlMjBhJTIwc3RvcnklMjB3ZSUyMHRlbGwlMjBvdXJzZWx2ZXMuJTIyJTdEJTVEJTdEJTJDJTBBJTVEJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwY29udmVyc2F0aW9uJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBJTIzJTIwaW5mZXIlMjB0aGUlMjBtb2RlbCUwQWF1ZGlvJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBvdXRwdXRfYXVkaW8lM0RUcnVlKSUwQXByb2Nlc3Nvci5zYXZlX2F1ZGlvKGF1ZGlvJTJDJTIwJTIyZXhhbXBsZV93aXRob3V0X2NvbnRleHQud2F2JTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmForConditionalGeneration, AutoProcessor, infer_device

model_id = <span class="hljs-string">&quot;sesame/csm-1b&quot;</span>
device = infer_device()

<span class="hljs-comment"># load the model and the processor</span>
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

<span class="hljs-comment"># prepare the inputs</span>
text = <span class="hljs-string">&quot;[0]The past is just a story we tell ourselves.&quot;</span> <span class="hljs-comment"># \`[0]\` for speaker id 0</span>
inputs = processor(text, add_special_tokens=<span class="hljs-literal">True</span>).to(device)

<span class="hljs-comment"># another equivalent way to prepare the inputs</span>
conversation = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;0&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [{<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;The past is just a story we tell ourselves.&quot;</span>}]},
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=<span class="hljs-literal">True</span>,
    return_dict=<span class="hljs-literal">True</span>,
).to(model.device)

<span class="hljs-comment"># infer the model</span>
audio = model.generate(**inputs, output_audio=<span class="hljs-literal">True</span>)
processor.save_audio(audio, <span class="hljs-string">&quot;example_without_context.wav&quot;</span>)`,wrap:!1}}),Te=new Z({props:{title:"With Conversational Context",local:"with-conversational-context",headingTag:"h3"}}),be=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQ3NtRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTJDJTIwQXV0b1Byb2Nlc3NvciUyQyUyMGluZmVyX2RldmljZSUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUyQyUyMEF1ZGlvJTBBJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJzZXNhbWUlMkZjc20tMWIlMjIlMEFkZXZpY2UlMjAlM0QlMjBpbmZlcl9kZXZpY2UoKSUwQSUwQSUyMyUyMGxvYWQlMjB0aGUlMjBtb2RlbCUyMGFuZCUyMHRoZSUyMHByb2Nlc3NvciUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkKSUwQW1vZGVsJTIwJTNEJTIwQ3NtRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZChtb2RlbF9pZCUyQyUyMGRldmljZV9tYXAlM0RkZXZpY2UpJTBBJTBBJTIzJTIwcHJlcGFyZSUyMHRoZSUyMGlucHV0cyUwQWRzJTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZkYWlseXRhbGstZHVtbXklMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQSUyMyUyMGVuc3VyZSUyMHRoZSUyMGF1ZGlvJTIwaXMlMjAyNGtIeiUwQWRzJTIwJTNEJTIwZHMuY2FzdF9jb2x1bW4oJTIyYXVkaW8lMjIlMkMlMjBBdWRpbyhzYW1wbGluZ19yYXRlJTNEMjQwMDApKSUwQWNvbnZlcnNhdGlvbiUyMCUzRCUyMCU1QiU1RCUwQSUwQSUyMyUyMDEuJTIwY29udGV4dCUwQWZvciUyMHRleHQlMkMlMjBhdWRpbyUyQyUyMHNwZWFrZXJfaWQlMjBpbiUyMHppcChkcyU1QiUzQTQlNUQlNUIlMjJ0ZXh0JTIyJTVEJTJDJTIwZHMlNUIlM0E0JTVEJTVCJTIyYXVkaW8lMjIlNUQlMkMlMjBkcyU1QiUzQTQlNUQlNUIlMjJzcGVha2VyX2lkJTIyJTVEKSUzQSUwQSUyMCUyMCUyMCUyMGNvbnZlcnNhdGlvbi5hcHBlbmQoJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycm9sZSUyMiUzQSUyMGYlMjIlN0JzcGVha2VyX2lkJTdEJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMHRleHQlN0QlMkMlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyYXVkaW8lMjIlMkMlMjAlMjJwYXRoJTIyJTNBJTIwYXVkaW8lNUIlMjJhcnJheSUyMiU1RCU3RCU1RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3RCUwQSUyMCUyMCUyMCUyMCklMEElMEElMjMlMjAyLiUyMHRleHQlMjBwcm9tcHQlMEFjb252ZXJzYXRpb24uYXBwZW5kKCU3QiUyMnJvbGUlMjIlM0ElMjBmJTIyJTdCZHMlNUI0JTVEJTVCJ3NwZWFrZXJfaWQnJTVEJTdEJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMGRzJTVCNCU1RCU1QiUyMnRleHQlMjIlNUQlN0QlNUQlN0QpJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwY29udmVyc2F0aW9uJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBJTIzJTIwaW5mZXIlMjB0aGUlMjBtb2RlbCUwQWF1ZGlvJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBvdXRwdXRfYXVkaW8lM0RUcnVlKSUwQXByb2Nlc3Nvci5zYXZlX2F1ZGlvKGF1ZGlvJTJDJTIwJTIyZXhhbXBsZV93aXRoX2NvbnRleHQud2F2JTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmForConditionalGeneration, AutoProcessor, infer_device
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio

model_id = <span class="hljs-string">&quot;sesame/csm-1b&quot;</span>
device = infer_device()

<span class="hljs-comment"># load the model and the processor</span>
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

<span class="hljs-comment"># prepare the inputs</span>
ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-comment"># ensure the audio is 24kHz</span>
ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">24000</span>))
conversation = []

<span class="hljs-comment"># 1. context</span>
<span class="hljs-keyword">for</span> text, audio, speaker_id <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;text&quot;</span>], ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;audio&quot;</span>], ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;speaker_id&quot;</span>]):
    conversation.append(
        {
            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{speaker_id}</span>&quot;</span>,
            <span class="hljs-string">&quot;content&quot;</span>: [{<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: text}, {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: audio[<span class="hljs-string">&quot;array&quot;</span>]}],
        }
    )

<span class="hljs-comment"># 2. text prompt</span>
conversation.append({<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">4</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [{<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">4</span>][<span class="hljs-string">&quot;text&quot;</span>]}]})

inputs = processor.apply_chat_template(
    conversation,
    tokenize=<span class="hljs-literal">True</span>,
    return_dict=<span class="hljs-literal">True</span>,
).to(model.device)

<span class="hljs-comment"># infer the model</span>
audio = model.generate(**inputs, output_audio=<span class="hljs-literal">True</span>)
processor.save_audio(audio, <span class="hljs-string">&quot;example_with_context.wav&quot;</span>)`,wrap:!1}}),Je=new Z({props:{title:"Batched Inference",local:"batched-inference",headingTag:"h3"}}),we=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQ3NtRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTJDJTIwQXV0b1Byb2Nlc3NvciUyQyUyMGluZmVyX2RldmljZSUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUyQyUyMEF1ZGlvJTBBJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJzZXNhbWUlMkZjc20tMWIlMjIlMEFkZXZpY2UlMjAlM0QlMjBpbmZlcl9kZXZpY2UoKSUwQSUwQSUyMyUyMGxvYWQlMjB0aGUlMjBtb2RlbCUyMGFuZCUyMHRoZSUyMHByb2Nlc3NvciUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkKSUwQW1vZGVsJTIwJTNEJTIwQ3NtRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZChtb2RlbF9pZCUyQyUyMGRldmljZV9tYXAlM0RkZXZpY2UpJTBBJTBBJTIzJTIwcHJlcGFyZSUyMHRoZSUyMGlucHV0cyUyMCUwQWRzJTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZkYWlseXRhbGstZHVtbXklMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQSUyMyUyMGVuc3VyZSUyMHRoZSUyMGF1ZGlvJTIwaXMlMjAyNGtIeiUwQWRzJTIwJTNEJTIwZHMuY2FzdF9jb2x1bW4oJTIyYXVkaW8lMjIlMkMlMjBBdWRpbyhzYW1wbGluZ19yYXRlJTNEMjQwMDApKSUwQSUyMyUyMGhlcmUlMjBhJTIwYmF0Y2glMjB3aXRoJTIwdHdvJTIwcHJvbXB0cyUwQWNvbnZlcnNhdGlvbiUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjBmJTIyJTdCZHMlNUIwJTVEJTVCJ3NwZWFrZXJfaWQnJTVEJTdEJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMGRzJTVCMCU1RCU1QiUyMnRleHQlMjIlNUQlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyYXVkaW8lMjIlMkMlMjAlMjJwYXRoJTIyJTNBJTIwZHMlNUIwJTVEJTVCJTIyYXVkaW8lMjIlNUQlNUIlMjJhcnJheSUyMiU1RCU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU1RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjBmJTIyJTdCZHMlNUIxJTVEJTVCJ3NwZWFrZXJfaWQnJTVEJTdEJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMGRzJTVCMSU1RCU1QiUyMnRleHQlMjIlNUQlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwZiUyMiU3QmRzJTVCMCU1RCU1QidzcGVha2VyX2lkJyU1RCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjBkcyU1QjAlNUQlNUIlMjJ0ZXh0JTIyJTVEJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTBBJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTVEJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwY29udmVyc2F0aW9uJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBYXVkaW8lMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG91dHB1dF9hdWRpbyUzRFRydWUpJTBBcHJvY2Vzc29yLnNhdmVfYXVkaW8oYXVkaW8lMkMlMjAlNUJmJTIyc3BlZWNoX2JhdGNoX2lkeF8lN0JpJTdELndhdiUyMiUyMGZvciUyMGklMjBpbiUyMHJhbmdlKGxlbihhdWRpbykpJTVEKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmForConditionalGeneration, AutoProcessor, infer_device
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio

model_id = <span class="hljs-string">&quot;sesame/csm-1b&quot;</span>
device = infer_device()

<span class="hljs-comment"># load the model and the processor</span>
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

<span class="hljs-comment"># prepare the inputs </span>
ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-comment"># ensure the audio is 24kHz</span>
ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">24000</span>))
<span class="hljs-comment"># here a batch with two prompts</span>
conversation = [
    [
        {
            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">0</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
            <span class="hljs-string">&quot;content&quot;</span>: [
                {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;text&quot;</span>]},
                {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]},
            ],
        },
        {
            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">1</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
            <span class="hljs-string">&quot;content&quot;</span>: [
                {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">1</span>][<span class="hljs-string">&quot;text&quot;</span>]},
            ],
        },
    ],
    [
        {
            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">0</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
            <span class="hljs-string">&quot;content&quot;</span>: [
                {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;text&quot;</span>]},
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=<span class="hljs-literal">True</span>,
    return_dict=<span class="hljs-literal">True</span>,
).to(model.device)

audio = model.generate(**inputs, output_audio=<span class="hljs-literal">True</span>)
processor.save_audio(audio, [<span class="hljs-string">f&quot;speech_batch_idx_<span class="hljs-subst">{i}</span>.wav&quot;</span> <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(<span class="hljs-built_in">len</span>(audio))])`,wrap:!1}}),Ce=new Z({props:{title:"Making The Model Go Brrr",local:"making-the-model-go-brrr",headingTag:"h3"}}),ve=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFpbXBvcnQlMjBjb3B5JTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyMEF1dG9Qcm9jZXNzb3IlMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMEElMEFtb2RlbF9pZCUyMCUzRCUyMCUyMnNlc2FtZSUyRmNzbS0xYiUyMiUwQWRldmljZSUyMCUzRCUyMCUyMmN1ZGElMjIlMEElMEElMjMlMjBzZXQlMjBsb2dzJTIwdG8lMjBlbnN1cmUlMjBubyUyMHJlY29tcGlsYXRpb24lMjBhbmQlMjBncmFwaCUyMGJyZWFrcyUwQXRvcmNoLl9sb2dnaW5nLnNldF9sb2dzKGdyYXBoX2JyZWFrcyUzRFRydWUlMkMlMjByZWNvbXBpbGVzJTNEVHJ1ZSUyQyUyMGN1ZGFncmFwaHMlM0RUcnVlKSUwQSUwQSUyMyUyMGxvYWQlMjB0aGUlMjBtb2RlbCUyMGFuZCUyMHRoZSUyMHByb2Nlc3NvciUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkKSUwQW1vZGVsJTIwJTNEJTIwQ3NtRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZChtb2RlbF9pZCUyQyUyMGRldmljZV9tYXAlM0RkZXZpY2UpJTBBJTBBJTIzJTIwdXNlJTIwc3RhdGljJTIwY2FjaGUlMkMlMjBlbmFibGluZyUyMGF1dG9tYXRpY2FsbHklMjB0b3JjaCUyMGNvbXBpbGUlMjB3aXRoJTIwZnVsbGdyYXBoJTIwYW5kJTIwcmVkdWNlLW92ZXJoZWFkJTBBbW9kZWwuZ2VuZXJhdGlvbl9jb25maWcubWF4X2xlbmd0aCUyMCUzRCUyMDI1MCUyMCUyMyUyMGJpZyUyMGVub3VnaCUyMHRvJTIwYXZvaWQlMjByZWNvbXBpbGF0aW9uJTBBbW9kZWwuZ2VuZXJhdGlvbl9jb25maWcubWF4X25ld190b2tlbnMlMjAlM0QlMjBOb25lJTIwJTIzJTIwd291bGQlMjB0YWtlJTIwcHJlY2VkZW5jZSUyMG92ZXIlMjBtYXhfbGVuZ3RoJTBBbW9kZWwuZ2VuZXJhdGlvbl9jb25maWcuY2FjaGVfaW1wbGVtZW50YXRpb24lMjAlM0QlMjAlMjJzdGF0aWMlMjIlMEFtb2RlbC5kZXB0aF9kZWNvZGVyLmdlbmVyYXRpb25fY29uZmlnLmNhY2hlX2ltcGxlbWVudGF0aW9uJTIwJTNEJTIwJTIyc3RhdGljJTIyJTBBJTBBJTIzJTIwZ2VuZXJhdGlvbiUyMGt3YXJncyUwQWdlbl9rd2FyZ3MlMjAlM0QlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjJkb19zYW1wbGUlMjIlM0ElMjBGYWxzZSUyQyUwQSUyMCUyMCUyMCUyMCUyMmRlcHRoX2RlY29kZXJfZG9fc2FtcGxlJTIyJTNBJTIwRmFsc2UlMkMlMEElMjAlMjAlMjAlMjAlMjJ0ZW1wZXJhdHVyZSUyMiUzQSUyMDEuMCUyQyUwQSUyMCUyMCUyMCUyMCUyMmRlcHRoX2RlY29kZXJfdGVtcGVyYXR1cmUlMjIlM0ElMjAxLjAlMkMlMEElN0QlMEElMEElMjMlMjBEZWZpbmUlMjBhJTIwdGltaW5nJTIwZGVjb3JhdG9yJTBBY2xhc3MlMjBUaW1lckNvbnRleHQlM0ElMEElMjAlMjAlMjAlMjBkZWYlMjBfX2luaXRfXyhzZWxmJTJDJTIwbmFtZSUzRCUyMkV4ZWN1dGlvbiUyMiklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBzZWxmLm5hbWUlMjAlM0QlMjBuYW1lJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc2VsZi5zdGFydF9ldmVudCUyMCUzRCUyME5vbmUlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBzZWxmLmVuZF9ldmVudCUyMCUzRCUyME5vbmUlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMEElMjAlMjAlMjAlMjBkZWYlMjBfX2VudGVyX18oc2VsZiklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjMlMjBVc2UlMjBDVURBJTIwZXZlbnRzJTIwZm9yJTIwbW9yZSUyMGFjY3VyYXRlJTIwR1BVJTIwdGltaW5nJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc2VsZi5zdGFydF9ldmVudCUyMCUzRCUyMHRvcmNoLmN1ZGEuRXZlbnQoZW5hYmxlX3RpbWluZyUzRFRydWUpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc2VsZi5lbmRfZXZlbnQlMjAlM0QlMjB0b3JjaC5jdWRhLkV2ZW50KGVuYWJsZV90aW1pbmclM0RUcnVlKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHNlbGYuc3RhcnRfZXZlbnQucmVjb3JkKCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjByZXR1cm4lMjBzZWxmJTBBJTBBJTIwJTIwJTIwJTIwZGVmJTIwX19leGl0X18oc2VsZiUyQyUyMCphcmdzKSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHNlbGYuZW5kX2V2ZW50LnJlY29yZCgpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdG9yY2guY3VkYS5zeW5jaHJvbml6ZSgpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZWxhcHNlZF90aW1lJTIwJTNEJTIwc2VsZi5zdGFydF9ldmVudC5lbGFwc2VkX3RpbWUoc2VsZi5lbmRfZXZlbnQpJTIwJTJGJTIwMTAwMC4wJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcHJpbnQoZiUyMiU3QnNlbGYubmFtZSU3RCUyMHRpbWUlM0ElMjAlN0JlbGFwc2VkX3RpbWUlM0EuNGYlN0QlMjBzZWNvbmRzJTIyKSUwQSUwQSUyMyUyMHByZXBhcmUlMjB0aGUlMjBpbnB1dHMlMjAlMEFkcyUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGZGFpbHl0YWxrLWR1bW15JTIyJTJDJTIwc3BsaXQlM0QlMjJ0cmFpbiUyMiklMEElMEFjb252ZXJzYXRpb24lMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwZiUyMiU3QmRzJTVCMCU1RCU1QidzcGVha2VyX2lkJyU1RCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjBkcyU1QjAlNUQlNUIlMjJ0ZXh0JTIyJTVEJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmF1ZGlvJTIyJTJDJTIwJTIycGF0aCUyMiUzQSUyMGRzJTVCMCU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwZiUyMiU3QmRzJTVCMSU1RCU1QidzcGVha2VyX2lkJyU1RCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjBkcyU1QjElNUQlNUIlMjJ0ZXh0JTIyJTVEJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmF1ZGlvJTIyJTJDJTIwJTIycGF0aCUyMiUzQSUyMGRzJTVCMSU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwZiUyMiU3QmRzJTVCMiU1RCU1QidzcGVha2VyX2lkJyU1RCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjBkcyU1QjIlNUQlNUIlMjJ0ZXh0JTIyJTVEJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTVEJTBBJTBBcGFkZGVkX2lucHV0c18xJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwY29udmVyc2F0aW9uJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBcHJpbnQoJTIyJTVDbiUyMiUyMCUyQiUyMCUyMiUzRCUyMio1MCklMEFwcmludCglMjJGaXJzdCUyMGdlbmVyYXRpb24lMjAtJTIwY29tcGlsaW5nJTIwYW5kJTIwcmVjb3JkaW5nJTIwQ1VEQSUyMGdyYXBocy4uLiUyMiklMEF3aXRoJTIwVGltZXJDb250ZXh0KCUyMkZpcnN0JTIwZ2VuZXJhdGlvbiUyMiklM0ElMEElMjAlMjAlMjAlMjBfJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKipwYWRkZWRfaW5wdXRzXzElMkMlMjAqKmdlbl9rd2FyZ3MpJTBBcHJpbnQoJTIyJTNEJTIyKjUwKSUwQSUwQXByaW50KCUyMiU1Q24lMjIlMjAlMkIlMjAlMjIlM0QlMjIqNTApJTBBcHJpbnQoJTIyU2Vjb25kJTIwZ2VuZXJhdGlvbiUyMC0lMjBmYXN0JTIwISEhJTIyKSUwQXdpdGglMjBUaW1lckNvbnRleHQoJTIyU2Vjb25kJTIwZ2VuZXJhdGlvbiUyMiklM0ElMEElMjAlMjAlMjAlMjBfJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKipwYWRkZWRfaW5wdXRzXzElMkMlMjAqKmdlbl9rd2FyZ3MpJTBBcHJpbnQoJTIyJTNEJTIyKjUwKSUwQSUwQSUyMyUyMG5vdyUyMHdpdGglMjBkaWZmZXJlbnQlMjBpbnB1dHMlMEFjb252ZXJzYXRpb24lMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwZiUyMiU3QmRzJTVCMCU1RCU1QidzcGVha2VyX2lkJyU1RCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjBkcyU1QjIlNUQlNUIlMjJ0ZXh0JTIyJTVEJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmF1ZGlvJTIyJTJDJTIwJTIycGF0aCUyMiUzQSUyMGRzJTVCMiU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwZiUyMiU3QmRzJTVCMSU1RCU1QidzcGVha2VyX2lkJyU1RCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjBkcyU1QjMlNUQlNUIlMjJ0ZXh0JTIyJTVEJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmF1ZGlvJTIyJTJDJTIwJTIycGF0aCUyMiUzQSUyMGRzJTVCMyU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwZiUyMiU3QmRzJTVCMiU1RCU1QidzcGVha2VyX2lkJyU1RCU3RCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjBkcyU1QjQlNUQlNUIlMjJ0ZXh0JTIyJTVEJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTVEJTBBcGFkZGVkX2lucHV0c18yJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwY29udmVyc2F0aW9uJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBcHJpbnQoJTIyJTVDbiUyMiUyMCUyQiUyMCUyMiUzRCUyMio1MCklMEFwcmludCglMjJHZW5lcmF0aW9uJTIwd2l0aCUyMG90aGVyJTIwaW5wdXRzISUyMiklMEF3aXRoJTIwVGltZXJDb250ZXh0KCUyMkdlbmVyYXRpb24lMjB3aXRoJTIwZGlmZmVyZW50JTIwaW5wdXRzJTIyKSUzQSUwQSUyMCUyMCUyMCUyMF8lMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKnBhZGRlZF9pbnB1dHNfMiUyQyUyMCoqZ2VuX2t3YXJncyklMEFwcmludCglMjIlM0QlMjIqNTAp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">import</span> copy
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmForConditionalGeneration, AutoProcessor
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

model_id = <span class="hljs-string">&quot;sesame/csm-1b&quot;</span>
device = <span class="hljs-string">&quot;cuda&quot;</span>

<span class="hljs-comment"># set logs to ensure no recompilation and graph breaks</span>
torch._logging.set_logs(graph_breaks=<span class="hljs-literal">True</span>, recompiles=<span class="hljs-literal">True</span>, cudagraphs=<span class="hljs-literal">True</span>)

<span class="hljs-comment"># load the model and the processor</span>
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

<span class="hljs-comment"># use static cache, enabling automatically torch compile with fullgraph and reduce-overhead</span>
model.generation_config.max_length = <span class="hljs-number">250</span> <span class="hljs-comment"># big enough to avoid recompilation</span>
model.generation_config.max_new_tokens = <span class="hljs-literal">None</span> <span class="hljs-comment"># would take precedence over max_length</span>
model.generation_config.cache_implementation = <span class="hljs-string">&quot;static&quot;</span>
model.depth_decoder.generation_config.cache_implementation = <span class="hljs-string">&quot;static&quot;</span>

<span class="hljs-comment"># generation kwargs</span>
gen_kwargs = {
    <span class="hljs-string">&quot;do_sample&quot;</span>: <span class="hljs-literal">False</span>,
    <span class="hljs-string">&quot;depth_decoder_do_sample&quot;</span>: <span class="hljs-literal">False</span>,
    <span class="hljs-string">&quot;temperature&quot;</span>: <span class="hljs-number">1.0</span>,
    <span class="hljs-string">&quot;depth_decoder_temperature&quot;</span>: <span class="hljs-number">1.0</span>,
}

<span class="hljs-comment"># Define a timing decorator</span>
<span class="hljs-keyword">class</span> <span class="hljs-title class_">TimerContext</span>:
    <span class="hljs-keyword">def</span> <span class="hljs-title function_">__init__</span>(<span class="hljs-params">self, name=<span class="hljs-string">&quot;Execution&quot;</span></span>):
        self.name = name
        self.start_event = <span class="hljs-literal">None</span>
        self.end_event = <span class="hljs-literal">None</span>
        
    <span class="hljs-keyword">def</span> <span class="hljs-title function_">__enter__</span>(<span class="hljs-params">self</span>):
        <span class="hljs-comment"># Use CUDA events for more accurate GPU timing</span>
        self.start_event = torch.cuda.Event(enable_timing=<span class="hljs-literal">True</span>)
        self.end_event = torch.cuda.Event(enable_timing=<span class="hljs-literal">True</span>)
        self.start_event.record()
        <span class="hljs-keyword">return</span> self

    <span class="hljs-keyword">def</span> <span class="hljs-title function_">__exit__</span>(<span class="hljs-params">self, *args</span>):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event) / <span class="hljs-number">1000.0</span>
        <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;<span class="hljs-subst">{self.name}</span> time: <span class="hljs-subst">{elapsed_time:<span class="hljs-number">.4</span>f}</span> seconds&quot;</span>)

<span class="hljs-comment"># prepare the inputs </span>
ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)

conversation = [
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">0</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;text&quot;</span>]},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]},
        ],
    },
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">1</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">1</span>][<span class="hljs-string">&quot;text&quot;</span>]},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: ds[<span class="hljs-number">1</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]},
        ],
    },
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">2</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">2</span>][<span class="hljs-string">&quot;text&quot;</span>]},
        ],
    },
]

padded_inputs_1 = processor.apply_chat_template(
    conversation,
    tokenize=<span class="hljs-literal">True</span>,
    return_dict=<span class="hljs-literal">True</span>,
).to(model.device)

<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;\\n&quot;</span> + <span class="hljs-string">&quot;=&quot;</span>*<span class="hljs-number">50</span>)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;First generation - compiling and recording CUDA graphs...&quot;</span>)
<span class="hljs-keyword">with</span> TimerContext(<span class="hljs-string">&quot;First generation&quot;</span>):
    _ = model.generate(**padded_inputs_1, **gen_kwargs)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;=&quot;</span>*<span class="hljs-number">50</span>)

<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;\\n&quot;</span> + <span class="hljs-string">&quot;=&quot;</span>*<span class="hljs-number">50</span>)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Second generation - fast !!!&quot;</span>)
<span class="hljs-keyword">with</span> TimerContext(<span class="hljs-string">&quot;Second generation&quot;</span>):
    _ = model.generate(**padded_inputs_1, **gen_kwargs)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;=&quot;</span>*<span class="hljs-number">50</span>)

<span class="hljs-comment"># now with different inputs</span>
conversation = [
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">0</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">2</span>][<span class="hljs-string">&quot;text&quot;</span>]},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: ds[<span class="hljs-number">2</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]},
        ],
    },
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">1</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">3</span>][<span class="hljs-string">&quot;text&quot;</span>]},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: ds[<span class="hljs-number">3</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]},
        ],
    },
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{ds[<span class="hljs-number">2</span>][<span class="hljs-string">&#x27;speaker_id&#x27;</span>]}</span>&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: ds[<span class="hljs-number">4</span>][<span class="hljs-string">&quot;text&quot;</span>]},
        ],
    },
]
padded_inputs_2 = processor.apply_chat_template(
    conversation,
    tokenize=<span class="hljs-literal">True</span>,
    return_dict=<span class="hljs-literal">True</span>,
).to(model.device)

<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;\\n&quot;</span> + <span class="hljs-string">&quot;=&quot;</span>*<span class="hljs-number">50</span>)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Generation with other inputs!&quot;</span>)
<span class="hljs-keyword">with</span> TimerContext(<span class="hljs-string">&quot;Generation with different inputs&quot;</span>):
    _ = model.generate(**padded_inputs_2, **gen_kwargs)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;=&quot;</span>*<span class="hljs-number">50</span>)`,wrap:!1}}),ke=new Z({props:{title:"Training",local:"training",headingTag:"h3"}}),qe=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBpbmZlcl9kZXZpY2UlMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMkMlMjBBdWRpbyUwQSUwQW1vZGVsX2lkJTIwJTNEJTIwJTIyc2VzYW1lJTJGY3NtLTFiJTIyJTBBZGV2aWNlJTIwJTNEJTIwaW5mZXJfZGV2aWNlKCklMEElMEElMjMlMjBsb2FkJTIwdGhlJTIwbW9kZWwlMjBhbmQlMjB0aGUlMjBwcm9jZXNzb3IlMEFwcm9jZXNzb3IlMjAlM0QlMjBBdXRvUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZChtb2RlbF9pZCklMEFtb2RlbCUyMCUzRCUyMENzbUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQlMkMlMjBkZXZpY2VfbWFwJTNEZGV2aWNlKSUwQW1vZGVsLnRyYWluKCklMEFtb2RlbC5jb2RlY19tb2RlbC5ldmFsKCklMEElMEFkcyUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGZGFpbHl0YWxrLWR1bW15JTIyJTJDJTIwc3BsaXQlM0QlMjJ0cmFpbiUyMiklMEElMjMlMjBlbnN1cmUlMjB0aGUlMjBhdWRpbyUyMGlzJTIwMjRrSHolMEFkcyUyMCUzRCUyMGRzLmNhc3RfY29sdW1uKCUyMmF1ZGlvJTIyJTJDJTIwQXVkaW8oc2FtcGxpbmdfcmF0ZSUzRDI0MDAwKSklMEFjb252ZXJzYXRpb24lMjAlM0QlMjAlNUIlNUQlMEElMEElMjMlMjBjb250ZXh0JTBBZm9yJTIwdGV4dCUyQyUyMGF1ZGlvJTJDJTIwc3BlYWtlcl9pZCUyMGluJTIwemlwKGRzJTVCJTNBNCU1RCU1QiUyMnRleHQlMjIlNUQlMkMlMjBkcyU1QiUzQTQlNUQlNUIlMjJhdWRpbyUyMiU1RCUyQyUyMGRzJTVCJTNBNCU1RCU1QiUyMnNwZWFrZXJfaWQlMjIlNUQpJTNBJTBBJTIwJTIwJTIwJTIwY29udmVyc2F0aW9uLmFwcGVuZCglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwZiUyMiU3QnNwZWFrZXJfaWQlN0QlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwdGV4dCU3RCUyQyUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJhdWRpbyUyMiUyQyUyMCUyMnBhdGglMjIlM0ElMjBhdWRpbyU1QiUyMmFycmF5JTIyJTVEJTdEJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTBBJTIwJTIwJTIwJTIwKSUwQSUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3Nvci5hcHBseV9jaGF0X3RlbXBsYXRlKCUwQSUyMCUyMCUyMCUyMGNvbnZlcnNhdGlvbiUyQyUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMHJldHVybl9kaWN0JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMG91dHB1dF9sYWJlbHMlM0RUcnVlJTJDJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBb3V0JTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBb3V0Lmxvc3MuYmFja3dhcmQoKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CsmForConditionalGeneration, AutoProcessor, infer_device
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio

model_id = <span class="hljs-string">&quot;sesame/csm-1b&quot;</span>
device = infer_device()

<span class="hljs-comment"># load the model and the processor</span>
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)
model.train()
model.codec_model.<span class="hljs-built_in">eval</span>()

ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-comment"># ensure the audio is 24kHz</span>
ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">24000</span>))
conversation = []

<span class="hljs-comment"># context</span>
<span class="hljs-keyword">for</span> text, audio, speaker_id <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;text&quot;</span>], ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;audio&quot;</span>], ds[:<span class="hljs-number">4</span>][<span class="hljs-string">&quot;speaker_id&quot;</span>]):
    conversation.append(
        {
            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">f&quot;<span class="hljs-subst">{speaker_id}</span>&quot;</span>,
            <span class="hljs-string">&quot;content&quot;</span>: [{<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: text}, {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;audio&quot;</span>, <span class="hljs-string">&quot;path&quot;</span>: audio[<span class="hljs-string">&quot;array&quot;</span>]}],
        }
    )

inputs = processor.apply_chat_template(
    conversation,
    tokenize=<span class="hljs-literal">True</span>,
    return_dict=<span class="hljs-literal">True</span>,
    output_labels=<span class="hljs-literal">True</span>,
).to(model.device)

out = model(**inputs)
out.loss.backward()`,wrap:!1}}),Be=new Z({props:{title:"CsmConfig",local:"transformers.CsmConfig",headingTag:"h2"}}),Ae=new E({props:{name:"class transformers.CsmConfig",anchor:"transformers.CsmConfig",parameters:[{name:"num_codebooks",val:" = 32"},{name:"vocab_size",val:" = 2051"},{name:"text_vocab_size",val:" = 128256"},{name:"hidden_size",val:" = 2048"},{name:"intermediate_size",val:" = 8192"},{name:"num_hidden_layers",val:" = 16"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 8"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 128002"},{name:"codebook_pad_token_id",val:" = 2050"},{name:"codebook_eos_token_id",val:" = 0"},{name:"bos_token_id",val:" = 128000"},{name:"eos_token_id",val:" = None"},{name:"audio_token_id",val:" = 128002"},{name:"audio_eos_token_id",val:" = 128003"},{name:"rope_theta",val:" = 500000"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"head_dim",val:" = None"},{name:"tie_codebooks_embeddings",val:" = True"},{name:"depth_decoder_config",val:" = None"},{name:"codec_config",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CsmConfig.num_codebooks",description:`<strong>num_codebooks</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of codebooks used in the underlying codec model responsible for tokenizing the audio.`,name:"num_codebooks"},{anchor:"transformers.CsmConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2051) &#x2014;
Vocabulary size of the Csm model. Defines the number of different audio tokens that can be represented by each codebook.`,name:"vocab_size"},{anchor:"transformers.CsmConfig.text_vocab_size",description:`<strong>text_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128256) &#x2014;
Vocabulary size of the text input for the Csm model. Defines the number of different text tokens that can be represented.`,name:"text_vocab_size"},{anchor:"transformers.CsmConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations of the backbone model.`,name:"hidden_size"},{anchor:"transformers.CsmConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimension of the MLP representations of the backbone model.`,name:"intermediate_size"},{anchor:"transformers.CsmConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of hidden layers in the backbone model Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.CsmConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the backbone model Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.CsmConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>.`,name:"num_key_value_heads"},{anchor:"transformers.CsmConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the backbone model Transformer decoder.`,name:"hidden_act"},{anchor:"transformers.CsmConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.CsmConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.CsmConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.CsmConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.CsmConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128002) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.CsmConfig.codebook_pad_token_id",description:`<strong>codebook_pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2050) &#x2014;
Padding token id for codebook tokens.`,name:"codebook_pad_token_id"},{anchor:"transformers.CsmConfig.codebook_eos_token_id",description:`<strong>codebook_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
End of stream token id for codebook tokens.`,name:"codebook_eos_token_id"},{anchor:"transformers.CsmConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128000) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.CsmConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.CsmConfig.audio_token_id",description:`<strong>audio_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128002) &#x2014;
Audio token id in the text input.`,name:"audio_token_id"},{anchor:"transformers.CsmConfig.audio_eos_token_id",description:`<strong>audio_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128003) &#x2014;
End of stream token id for audio in the text input.`,name:"audio_eos_token_id"},{anchor:"transformers.CsmConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 500000) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.CsmConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>, defaults to <code>{&apos;factor&apos; -- 32.0, &apos;high_freq_factor&apos;: 0.5, &apos;low_freq_factor&apos;: 0.125, &apos;original_max_position_embeddings&apos;: 1024, &apos;rope_type&apos;: &apos;llama3&apos;}</code>):
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
<code>short_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.CsmConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.CsmConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.CsmConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.CsmConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The attention head dimension. If None, it will default to hidden_size // num_attention_heads`,name:"head_dim"},{anchor:"transformers.CsmConfig.tie_codebooks_embeddings",description:`<strong>tie_codebooks_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to tie the codebook tokens embeddings of the backbone model to the codebook tokens embeddings of the depth decoder.`,name:"tie_codebooks_embeddings"},{anchor:"transformers.CsmConfig.depth_decoder_config",description:`<strong>depth_decoder_config</strong> (<code>CsmDepthDecoderConfig</code>, <em>optional</em>) &#x2014;
Configuration for the depth decoder.`,name:"depth_decoder_config"},{anchor:"transformers.CsmConfig.codec_config",description:`<strong>codec_config</strong> (<code>PretrainedConfig</code>, <em>optional</em>) &#x2014;
Configuration for the codec.`,name:"codec_config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/configuration_csm.py#L207"}}),O=new qt({props:{anchor:"transformers.CsmConfig.example",$$slots:{default:[Ln]},$$scope:{ctx:j}}}),Ge=new Z({props:{title:"CsmDepthDecoderConfig",local:"transformers.CsmDepthDecoderConfig",headingTag:"h2"}}),Qe=new E({props:{name:"class transformers.CsmDepthDecoderConfig",anchor:"transformers.CsmDepthDecoderConfig",parameters:[{name:"num_codebooks",val:" = 32"},{name:"backbone_hidden_size",val:" = 2048"},{name:"vocab_size",val:" = 2051"},{name:"hidden_size",val:" = 1024"},{name:"intermediate_size",val:" = 8192"},{name:"num_hidden_layers",val:" = 4"},{name:"num_attention_heads",val:" = 8"},{name:"num_key_value_heads",val:" = 2"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 33"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = None"},{name:"eos_token_id",val:" = None"},{name:"rope_theta",val:" = 500000"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"head_dim",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CsmDepthDecoderConfig.num_codebooks",description:`<strong>num_codebooks</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of codebooks used in the underlying codec model responsible for tokenizing the audio.`,name:"num_codebooks"},{anchor:"transformers.CsmDepthDecoderConfig.backbone_hidden_size",description:`<strong>backbone_hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations of the backbone model used with this depth decoder.`,name:"backbone_hidden_size"},{anchor:"transformers.CsmDepthDecoderConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2051) &#x2014;
Vocabulary size of the CsmDepthDecoder model. Defines the number of different audio tokens that can be represented by each codebook.`,name:"vocab_size"},{anchor:"transformers.CsmDepthDecoderConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.CsmDepthDecoderConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.CsmDepthDecoderConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.CsmDepthDecoderConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.CsmDepthDecoderConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.CsmDepthDecoderConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.CsmDepthDecoderConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 33) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.CsmDepthDecoderConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.CsmDepthDecoderConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.CsmDepthDecoderConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.CsmDepthDecoderConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2050) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.CsmDepthDecoderConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.CsmDepthDecoderConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.CsmDepthDecoderConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 500000) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.CsmDepthDecoderConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
<code>short_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.CsmDepthDecoderConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.CsmDepthDecoderConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.CsmDepthDecoderConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.CsmDepthDecoderConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The attention head dimension. If None, it will default to hidden_size // num_attention_heads`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/configuration_csm.py#L25"}}),ee=new qt({props:{anchor:"transformers.CsmDepthDecoderConfig.example",$$slots:{default:[Pn]},$$scope:{ctx:j}}}),Re=new Z({props:{title:"CsmProcessor",local:"transformers.CsmProcessor",headingTag:"h2"}}),Ne=new E({props:{name:"class transformers.CsmProcessor",anchor:"transformers.CsmProcessor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""},{name:"chat_template",val:" = None"}],parametersDescription:[{anchor:"transformers.CsmProcessor.feature_extractor",description:`<strong>feature_extractor</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor">EncodecFeatureExtractor</a>) &#x2014;
The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.CsmProcessor.tokenizer",description:`<strong>tokenizer</strong> ([<code>PreTrainedTokenizer</code>, <code>PreTrainedTokenizerFast</code>]) &#x2014;
The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.CsmProcessor.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014; A Jinja template which will be used to convert lists of messages
in a chat into a tokenizable string.`,name:"chat_template"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/processing_csm.py#L62"}}),se=new qt({props:{anchor:"transformers.CsmProcessor.example",$$slots:{default:[Kn]},$$scope:{ctx:j}}}),ze=new E({props:{name:"__call__",anchor:"transformers.CsmProcessor.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType]"},{name:"audio",val:": typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), typing.Sequence[numpy.ndarray], typing.Sequence[ForwardRef('torch.Tensor')], NoneType] = None"},{name:"output_labels",val:": typing.Optional[bool] = False"},{name:"depth_decoder_labels_ratio",val:": typing.Optional[float] = 1.0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.csm.processing_csm.CsmProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.CsmProcessor.__call__.audio",description:`<strong>audio</strong> (<code>np.ndarray</code>, <code>torch.Tensor</code>, <code>list[np.ndarray]</code>, <code>list[torch.Tensor]</code>) &#x2014;
The audio or batch of audio to be prepared. Each audio can be a NumPy array or PyTorch
tensor.`,name:"audio"},{anchor:"transformers.CsmProcessor.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.CsmProcessor.__call__.output_labels",description:`<strong>output_labels</strong> (bool, <em>optional</em>, default=False) &#x2014;
Whether to return labels for training. Indices will be in <code>[config.audio_token_id, -100, -101]</code>.<ul>
<li><code>config.audio_token_id</code> indicates an audio frame (considering sequence length elements as frames)</li>
<li><code>-100</code> will be ignored in the loss computation</li>
<li><code>-101</code> indicates the audio frame will be used only for the backbone model (using the first codebook token as labels)</li>
</ul>`,name:"output_labels"},{anchor:"transformers.CsmProcessor.__call__.depth_decoder_labels_ratio",description:`<strong>depth_decoder_labels_ratio</strong> (float, <em>optional</em>, default=1.0) &#x2014;
The ratio of audio frames to keep for the depth decoder labels.`,name:"depth_decoder_labels_ratio"},{anchor:"transformers.CsmProcessor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/processing_csm.py#L197",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>input_ids</strong> — List of token ids to be fed to a model. Returned when <code>text</code> is not <code>None</code>.</li>
<li><strong>input_values</strong> — List of audio values to be fed to a model. Returned when <code>audio</code> is not <code>None</code>.</li>
<li><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code> and if <code>text</code> is not
<code>None</code>).</li>
<li><strong>labels</strong> — List of labels for the audio frames. Returned when <code>output_labels=True</code>.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),Ee=new Z({props:{title:"CsmForConditionalGeneration",local:"transformers.CsmForConditionalGeneration",headingTag:"h2"}}),We=new E({props:{name:"class transformers.CsmForConditionalGeneration",anchor:"transformers.CsmForConditionalGeneration",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CsmForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmForConditionalGeneration">CsmForConditionalGeneration</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L745"}}),Fe=new E({props:{name:"forward",anchor:"transformers.CsmForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"input_values",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"input_values_cutoffs",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.CsmForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, num_codebooks) or (batch_size, sequence_length)</code>) &#x2014;</p>
<ol>
<li>
<p>(batch_size, sequence_length): corresponds to the input sequence prepared with the processor from the text prompt. Such input
requires <code>input_values</code> to be provided so that audio can be encoded in codebook tokens and then merged with the text tokens.</p>
</li>
<li>
<p>(batch_size, sequence_length, num_codebooks): codebook tokens generated during the autoregressive decoding. Such input is not meant to be used by end users.</p>
</li>
</ol>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CsmForConditionalGeneration.forward.input_values",description:`<strong>input_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <code>.flac</code> or <code>.wav</code> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor">AutoProcessor</a> should be used for padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <code>processor_class.__call__</code> for details.`,name:"input_values"},{anchor:"transformers.CsmForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CsmForConditionalGeneration.forward.input_values_cutoffs",description:`<strong>input_values_cutoffs</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, max_num_audio)</code>, <em>optional</em>) &#x2014;
Specify the end positions of audio segments within each batch entry, relative to the concatenated audio input.
If a batch entry has fewer segments than the maximum, it is padded with -1. For example, in a batch of 2 sequences
where the first contains 2 audio segments of length l1, and the second contains 1 audio segment of length l2,
the input_values_cutoffs would be: [[l1, 2 * l1], [l2, -1]].`,name:"input_values_cutoffs"},{anchor:"transformers.CsmForConditionalGeneration.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CsmForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CsmForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CsmForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[config.audio_token_id, -100, -101]</code>.
Requires targeted <code>input_values</code> to be provided as audio tokens will be inferred from it using the <code>codec_model</code>.</p>
<ul>
<li><code>config.audio_token_id</code> indicates an audio frames (considering sequence length elements as frames)</li>
<li><code>-100</code> will be ignored in the loss computation</li>
<li><code>-101</code> indicates the audio frame will be used only for the backbone model (using the first codebook token as labels)</li>
</ul>
<p>Such labels can be prepared using <code>output_labels=True</code> when calling <a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmProcessor">CsmProcessor</a>.`,name:"labels"},{anchor:"transformers.CsmForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.CsmForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.CsmForConditionalGeneration.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>int</code> or <code>torch.Tensor</code>, <em>optional</em>) &#x2014;
Kept for compatibility. Does not support another value than:</p>
<ol>
<li><code>0</code>, which is equivalent to keeping all logits, used in the training regime</li>
<li><code>1</code>, which is equivalent to keeping only the last logit, used in the generation regime</li>
</ol>`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L924",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.csm.modeling_csm.CsmOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig"
>CsmConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>)</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
<li>
<p><strong>depth_decoder_loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction) of the depth decoder model.</p>
</li>
<li>
<p><strong>depth_decoder_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the depth decoder (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>depth_decoder_past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>)</p>
</li>
<li>
<p><strong>depth_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>depth_decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
</li>
<li>
<p><strong>backbone_loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction) of the backbone model.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.csm.modeling_csm.CsmOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),oe=new xt({props:{$$slots:{default:[On]},$$scope:{ctx:j}}}),ae=new qt({props:{anchor:"transformers.CsmForConditionalGeneration.forward.example",$$slots:{default:[eo]},$$scope:{ctx:j}}}),$e=new E({props:{name:"generate",anchor:"transformers.CsmForConditionalGeneration.generate",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_values",val:": typing.Optional[torch.Tensor] = None"},{name:"input_values_cutoffs",val:": typing.Optional[torch.Tensor] = None"},{name:"generation_config",val:": typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None"},{name:"logits_processor",val:": typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None"},{name:"stopping_criteria",val:": typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None"},{name:"synced_gpus",val:": typing.Optional[bool] = None"},{name:"streamer",val:": typing.Optional[ForwardRef('BaseStreamer')] = None"},{name:"output_audio",val:": typing.Optional[bool] = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CsmForConditionalGeneration.generate.inputs_ids",description:`<strong>inputs_ids</strong> (<code>torch.Tensor</code> of shape (batch_size, seq_length), <em>optional</em>) &#x2014;
The sequence used as a prompt for the backbone model.`,name:"inputs_ids"},{anchor:"transformers.CsmForConditionalGeneration.generate.input_values",description:`<strong>input_values</strong> (<code>torch.Tensor</code> of shape (batch_size, channels, max_concatenated_audio_length), <em>optional</em>) &#x2014;
The batched audio input values, where each batch entry contains the concatenation of all audio segments for that entry.
These values will be encoded into codebook tokens using the codec model and merged with the text input ids provided in <code>input_ids</code>.`,name:"input_values"},{anchor:"transformers.CsmForConditionalGeneration.generate.input_values_cutoffs",description:`<strong>input_values_cutoffs</strong> (<code>torch.Tensor</code> of shape (batch_size, max_num_audio), <em>optional</em>) &#x2014;
Specify the end positions of audio segments within each batch entry, relative to the concatenated audio input.
If a batch entry has fewer segments than the maximum, it is padded with -1. For example, in a batch of 2 sequences
where the first contains 2 audio segments of length l1, and the second contains 1 audio segment of length l2,
the input_values_cutoffs would be: [[l1, 2 * l1], [l2, -1]].`,name:"input_values_cutoffs"},{anchor:"transformers.CsmForConditionalGeneration.generate.generation_config",description:`<strong>generation_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>, <em>optional</em>) &#x2014;
The generation configuration to be used as base parametrization for the generation call. <code>**kwargs</code>
passed to generate matching the attributes of <code>generation_config</code> will override them. If
<code>generation_config</code> is not provided, the default will be used, which has the following loading
priority: 1) from the <code>generation_config.json</code> model file, if it exists; 2) from the model
configuration. Please note that unspecified parameters will inherit <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>&#x2019;s
default values, whose documentation should be checked to parameterize generation.`,name:"generation_config"},{anchor:"transformers.CsmForConditionalGeneration.generate.logits_processor",description:`<strong>logits_processor</strong> (<code>LogitsProcessorList</code>, <em>optional</em>) &#x2014;
Custom logits processors that complement the default logits processors built from arguments and
generation config. If a logit processor is passed that is already created with the arguments or a
generation config an error is thrown. This feature is intended for advanced users.`,name:"logits_processor"},{anchor:"transformers.CsmForConditionalGeneration.generate.stopping_criteria",description:`<strong>stopping_criteria</strong> (<code>StoppingCriteriaList</code>, <em>optional</em>) &#x2014;
Custom stopping criteria that complements the default stopping criteria built from arguments and a
generation config. If a stopping criteria is passed that is already created with the arguments or a
generation config an error is thrown. If your stopping criteria depends on the <code>scores</code> input, make
sure you pass <code>return_dict_in_generate=True, output_scores=True</code> to <code>generate</code>. This feature is
intended for advanced users.`,name:"stopping_criteria"},{anchor:"transformers.CsmForConditionalGeneration.generate.synced_gpus",description:`<strong>synced_gpus</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
to <code>True</code> if using <code>FullyShardedDataParallel</code> or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to <code>False</code>.`,name:"synced_gpus"},{anchor:"transformers.CsmForConditionalGeneration.generate.streamer",description:`<strong>streamer</strong> (<code>BaseStreamer</code>, <em>optional</em>) &#x2014;
Streamer object that will be used to stream the generated sequences. Generated tokens are passed
through <code>streamer.put(token_ids)</code> and the streamer is responsible for any further processing.`,name:"streamer"},{anchor:"transformers.CsmForConditionalGeneration.generate.output_audio",description:`<strong>output_audio</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the generated audio.`,name:"output_audio"},{anchor:"transformers.CsmForConditionalGeneration.generate.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Ad hoc parametrization of <code>generation_config</code> and/or additional model-specific kwargs that will be
forwarded to the <code>forward</code> function of the model. Depth decoder specific kwargs should be prefixed with <em>depth<em>decoder</em></em>.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/generation_csm.py#L338",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>CsmGenerateOutput</code>
(if <code>return_dict_in_generate=True</code> or when <code>config.return_dict_in_generate=True</code>) or a <code>torch.LongTensor</code> when <code>output_audio=False</code>
or a <code>list[torch.FloatTensor]</code> otherwise.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>CsmGenerateOutput</code> or <code>torch.LongTensor</code> or <code>list[torch.FloatTensor]</code></p>
`}}),le=new xt({props:{warning:!0,$$slots:{default:[to]},$$scope:{ctx:j}}}),re=new qt({props:{anchor:"transformers.CsmForConditionalGeneration.generate.example",$$slots:{default:[so]},$$scope:{ctx:j}}}),Xe=new Z({props:{title:"CsmDepthDecoderForCausalLM",local:"transformers.CsmDepthDecoderForCausalLM",headingTag:"h2"}}),Ve=new E({props:{name:"class transformers.CsmDepthDecoderForCausalLM",anchor:"transformers.CsmDepthDecoderForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CsmDepthDecoderForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderForCausalLM">CsmDepthDecoderForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L531"}}),De=new E({props:{name:"forward",anchor:"transformers.CsmDepthDecoderForCausalLM.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"backbone_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.backbone_last_hidden_state",description:`<strong>backbone_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, backbone_hidden_size)</code>, <em>optional</em>) &#x2014;
The last hidden state of the backbone model. Such input is required when the first codebook token (the one generated by the backbone model)
is provided in the <code>input_ids</code> argument.`,name:"backbone_last_hidden_state"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.CsmDepthDecoderForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L545",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig"
>CsmConfig</a>) and inputs.</p>
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
`}}),ie=new xt({props:{$$slots:{default:[no]},$$scope:{ctx:j}}}),Se=new Z({props:{title:"CsmDepthDecoderModel",local:"transformers.CsmDepthDecoderModel",headingTag:"h2"}}),He=new E({props:{name:"class transformers.CsmDepthDecoderModel",anchor:"transformers.CsmDepthDecoderModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CsmDepthDecoderModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmDepthDecoderModel">CsmDepthDecoderModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L395"}}),Ye=new E({props:{name:"forward",anchor:"transformers.CsmDepthDecoderModel.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"backbone_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.CsmDepthDecoderModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CsmDepthDecoderModel.forward.backbone_last_hidden_state",description:`<strong>backbone_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, backbone_hidden_size)</code>, <em>optional</em>) &#x2014;
The last hidden state of the backbone model. Such input is required when the first codebook token (the one generated by the backbone model)
is provided in the <code>input_ids</code> argument.`,name:"backbone_last_hidden_state"},{anchor:"transformers.CsmDepthDecoderModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CsmDepthDecoderModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CsmDepthDecoderModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CsmDepthDecoderModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CsmDepthDecoderModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.CsmDepthDecoderModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L414",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig"
>CsmConfig</a>) and inputs.</p>
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
`}}),ce=new xt({props:{$$slots:{default:[oo]},$$scope:{ctx:j}}}),Le=new Z({props:{title:"CsmBackboneModel",local:"transformers.CsmBackboneModel",headingTag:"h2"}}),Pe=new E({props:{name:"class transformers.CsmBackboneModel",anchor:"transformers.CsmBackboneModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CsmBackboneModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmBackboneModel">CsmBackboneModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L651"}}),Ke=new E({props:{name:"forward",anchor:"transformers.CsmBackboneModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.CsmBackboneModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, num_codebooks) or (batch_size, sequence_length)</code>) &#x2014;</p>
<ol>
<li>
<p>(batch_size, sequence_length): corresponds to the input sequence prepared with the processor from the text prompt. Such input
requires <code>input_values</code> to be provided so that audio can be encoded in codebook tokens and then merged with the text tokens.</p>
</li>
<li>
<p>(batch_size, sequence_length, num_codebooks): codebook tokens generated during the autoregressive decoding. Such input is not meant to be used by end users.</p>
</li>
</ol>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CsmBackboneModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CsmBackboneModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CsmBackboneModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CsmBackboneModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CsmBackboneModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.CsmBackboneModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/csm/modeling_csm.py#L667",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/csm#transformers.CsmConfig"
>CsmConfig</a>) and inputs.</p>
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
`}}),de=new xt({props:{$$slots:{default:[ao]},$$scope:{ctx:j}}}),Oe=new Yn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/csm.md"}}),{c(){s=c("meta"),y=o(),l=c("p"),_=o(),b=c("p"),b.innerHTML=T,W=o(),p(pe.$$.fragment),Zt=o(),p(me.$$.fragment),Bt=o(),he=c("p"),he.innerHTML=ln,At=o(),ue=c("p"),ue.innerHTML=rn,Gt=o(),Me=c("p"),Me.innerHTML=cn,Qt=o(),K=c("div"),K.innerHTML=dn,Rt=o(),p(ge.$$.fragment),Nt=o(),p(fe.$$.fragment),zt=o(),ye=c("p"),ye.textContent=pn,Et=o(),p(_e.$$.fragment),Wt=o(),p(Te.$$.fragment),Ft=o(),je=c("p"),je.textContent=mn,$t=o(),p(be.$$.fragment),Xt=o(),p(Je.$$.fragment),Vt=o(),Ue=c("p"),Ue.textContent=hn,Dt=o(),p(we.$$.fragment),St=o(),p(Ce.$$.fragment),Ht=o(),Ie=c("p"),Ie.textContent=un,Yt=o(),p(ve.$$.fragment),Lt=o(),p(ke.$$.fragment),Pt=o(),xe=c("p"),xe.textContent=Mn,Kt=o(),p(qe.$$.fragment),Ot=o(),Ze=c("p"),Ze.innerHTML=gn,es=o(),p(Be.$$.fragment),ts=o(),w=c("div"),p(Ae.$$.fragment),ys=o(),tt=c("p"),tt.innerHTML=fn,_s=o(),st=c("p"),st.innerHTML=yn,Ts=o(),nt=c("p"),nt.innerHTML=_n,js=o(),p(O.$$.fragment),ss=o(),p(Ge.$$.fragment),ns=o(),C=c("div"),p(Qe.$$.fragment),bs=o(),ot=c("p"),ot.innerHTML=Tn,Js=o(),at=c("p"),at.innerHTML=jn,Us=o(),lt=c("p"),lt.innerHTML=bn,ws=o(),p(ee.$$.fragment),os=o(),p(Re.$$.fragment),as=o(),te=c("div"),te.innerHTML=Jn,ls=o(),A=c("div"),p(Ne.$$.fragment),Cs=o(),rt=c("p"),rt.innerHTML=Un,Is=o(),p(se.$$.fragment),vs=o(),ne=c("div"),p(ze.$$.fragment),ks=o(),it=c("p"),it.innerHTML=wn,rs=o(),p(Ee.$$.fragment),is=o(),J=c("div"),p(We.$$.fragment),xs=o(),ct=c("p"),ct.textContent=Cn,qs=o(),dt=c("p"),dt.innerHTML=In,Zs=o(),pt=c("p"),pt.innerHTML=vn,Bs=o(),F=c("div"),p(Fe.$$.fragment),As=o(),mt=c("p"),mt.innerHTML=kn,Gs=o(),p(oe.$$.fragment),Qs=o(),p(ae.$$.fragment),Rs=o(),B=c("div"),p($e.$$.fragment),Ns=o(),ht=c("p"),ht.innerHTML=xn,zs=o(),ut=c("ol"),ut.innerHTML=qn,Es=o(),p(le.$$.fragment),Ws=o(),p(re.$$.fragment),cs=o(),p(Xe.$$.fragment),ds=o(),I=c("div"),p(Ve.$$.fragment),Fs=o(),Mt=c("p"),Mt.innerHTML=Zn,$s=o(),gt=c("p"),gt.innerHTML=Bn,Xs=o(),ft=c("p"),ft.innerHTML=An,Vs=o(),V=c("div"),p(De.$$.fragment),Ds=o(),yt=c("p"),yt.innerHTML=Gn,Ss=o(),p(ie.$$.fragment),ps=o(),p(Se.$$.fragment),ms=o(),v=c("div"),p(He.$$.fragment),Hs=o(),_t=c("p"),_t.textContent=Qn,Ys=o(),Tt=c("p"),Tt.innerHTML=Rn,Ls=o(),jt=c("p"),jt.innerHTML=Nn,Ps=o(),D=c("div"),p(Ye.$$.fragment),Ks=o(),bt=c("p"),bt.innerHTML=zn,Os=o(),p(ce.$$.fragment),hs=o(),p(Le.$$.fragment),us=o(),k=c("div"),p(Pe.$$.fragment),en=o(),Jt=c("p"),Jt.textContent=En,tn=o(),Ut=c("p"),Ut.innerHTML=Wn,sn=o(),wt=c("p"),wt.innerHTML=Fn,nn=o(),S=c("div"),p(Ke.$$.fragment),on=o(),Ct=c("p"),Ct.innerHTML=$n,an=o(),p(de.$$.fragment),Ms=o(),p(Oe.$$.fragment),gs=o(),kt=c("p"),this.h()},l(e){const t=Hn("svelte-u9bgzb",document.head);s=d(t,"META",{name:!0,content:!0}),t.forEach(n),y=a(e),l=d(e,"P",{}),q(l).forEach(n),_=a(e),b=d(e,"P",{"data-svelte-h":!0}),f(b)!=="svelte-14zvlx1"&&(b.innerHTML=T),W=a(e),m(pe.$$.fragment,e),Zt=a(e),m(me.$$.fragment,e),Bt=a(e),he=d(e,"P",{"data-svelte-h":!0}),f(he)!=="svelte-15v86h2"&&(he.innerHTML=ln),At=a(e),ue=d(e,"P",{"data-svelte-h":!0}),f(ue)!=="svelte-inql9"&&(ue.innerHTML=rn),Gt=a(e),Me=d(e,"P",{"data-svelte-h":!0}),f(Me)!=="svelte-4lmnqg"&&(Me.innerHTML=cn),Qt=a(e),K=d(e,"DIV",{class:!0,"data-svelte-h":!0}),f(K)!=="svelte-fz2ig7"&&(K.innerHTML=dn),Rt=a(e),m(ge.$$.fragment,e),Nt=a(e),m(fe.$$.fragment,e),zt=a(e),ye=d(e,"P",{"data-svelte-h":!0}),f(ye)!=="svelte-1xmq5mx"&&(ye.textContent=pn),Et=a(e),m(_e.$$.fragment,e),Wt=a(e),m(Te.$$.fragment,e),Ft=a(e),je=d(e,"P",{"data-svelte-h":!0}),f(je)!=="svelte-b2th4x"&&(je.textContent=mn),$t=a(e),m(be.$$.fragment,e),Xt=a(e),m(Je.$$.fragment,e),Vt=a(e),Ue=d(e,"P",{"data-svelte-h":!0}),f(Ue)!=="svelte-1vzzgbo"&&(Ue.textContent=hn),Dt=a(e),m(we.$$.fragment,e),St=a(e),m(Ce.$$.fragment,e),Ht=a(e),Ie=d(e,"P",{"data-svelte-h":!0}),f(Ie)!=="svelte-1i8fflp"&&(Ie.textContent=un),Yt=a(e),m(ve.$$.fragment,e),Lt=a(e),m(ke.$$.fragment,e),Pt=a(e),xe=d(e,"P",{"data-svelte-h":!0}),f(xe)!=="svelte-1hmyckm"&&(xe.textContent=Mn),Kt=a(e),m(qe.$$.fragment,e),Ot=a(e),Ze=d(e,"P",{"data-svelte-h":!0}),f(Ze)!=="svelte-175b7kk"&&(Ze.innerHTML=gn),es=a(e),m(Be.$$.fragment,e),ts=a(e),w=d(e,"DIV",{class:!0});var G=q(w);m(Ae.$$.fragment,G),ys=a(G),tt=d(G,"P",{"data-svelte-h":!0}),f(tt)!=="svelte-r9up6m"&&(tt.innerHTML=fn),_s=a(G),st=d(G,"P",{"data-svelte-h":!0}),f(st)!=="svelte-pwwfqb"&&(st.innerHTML=yn),Ts=a(G),nt=d(G,"P",{"data-svelte-h":!0}),f(nt)!=="svelte-1ek1ss9"&&(nt.innerHTML=_n),js=a(G),m(O.$$.fragment,G),G.forEach(n),ss=a(e),m(Ge.$$.fragment,e),ns=a(e),C=d(e,"DIV",{class:!0});var Q=q(C);m(Qe.$$.fragment,Q),bs=a(Q),ot=d(Q,"P",{"data-svelte-h":!0}),f(ot)!=="svelte-kbbntd"&&(ot.innerHTML=Tn),Js=a(Q),at=d(Q,"P",{"data-svelte-h":!0}),f(at)!=="svelte-pwwfqb"&&(at.innerHTML=jn),Us=a(Q),lt=d(Q,"P",{"data-svelte-h":!0}),f(lt)!=="svelte-1ek1ss9"&&(lt.innerHTML=bn),ws=a(Q),m(ee.$$.fragment,Q),Q.forEach(n),os=a(e),m(Re.$$.fragment,e),as=a(e),te=d(e,"DIV",{class:!0,"data-svelte-h":!0}),f(te)!=="svelte-quqljr"&&(te.innerHTML=Jn),ls=a(e),A=d(e,"DIV",{class:!0});var $=q(A);m(Ne.$$.fragment,$),Cs=a($),rt=d($,"P",{"data-svelte-h":!0}),f(rt)!=="svelte-1wwtpu6"&&(rt.innerHTML=Un),Is=a($),m(se.$$.fragment,$),vs=a($),ne=d($,"DIV",{class:!0});var et=q(ne);m(ze.$$.fragment,et),ks=a(et),it=d(et,"P",{"data-svelte-h":!0}),f(it)!=="svelte-17nn2et"&&(it.innerHTML=wn),et.forEach(n),$.forEach(n),rs=a(e),m(Ee.$$.fragment,e),is=a(e),J=d(e,"DIV",{class:!0});var x=q(J);m(We.$$.fragment,x),xs=a(x),ct=d(x,"P",{"data-svelte-h":!0}),f(ct)!=="svelte-15xzei"&&(ct.textContent=Cn),qs=a(x),dt=d(x,"P",{"data-svelte-h":!0}),f(dt)!=="svelte-q52n56"&&(dt.innerHTML=In),Zs=a(x),pt=d(x,"P",{"data-svelte-h":!0}),f(pt)!=="svelte-hswkmf"&&(pt.innerHTML=vn),Bs=a(x),F=d(x,"DIV",{class:!0});var X=q(F);m(Fe.$$.fragment,X),As=a(X),mt=d(X,"P",{"data-svelte-h":!0}),f(mt)!=="svelte-m1r0vw"&&(mt.innerHTML=kn),Gs=a(X),m(oe.$$.fragment,X),Qs=a(X),m(ae.$$.fragment,X),X.forEach(n),Rs=a(x),B=d(x,"DIV",{class:!0});var R=q(B);m($e.$$.fragment,R),Ns=a(R),ht=d(R,"P",{"data-svelte-h":!0}),f(ht)!=="svelte-9wxgdd"&&(ht.innerHTML=xn),zs=a(R),ut=d(R,"OL",{"data-svelte-h":!0}),f(ut)!=="svelte-1dezhe3"&&(ut.innerHTML=qn),Es=a(R),m(le.$$.fragment,R),Ws=a(R),m(re.$$.fragment,R),R.forEach(n),x.forEach(n),cs=a(e),m(Xe.$$.fragment,e),ds=a(e),I=d(e,"DIV",{class:!0});var N=q(I);m(Ve.$$.fragment,N),Fs=a(N),Mt=d(N,"P",{"data-svelte-h":!0}),f(Mt)!=="svelte-1rjjo9y"&&(Mt.innerHTML=Zn),$s=a(N),gt=d(N,"P",{"data-svelte-h":!0}),f(gt)!=="svelte-q52n56"&&(gt.innerHTML=Bn),Xs=a(N),ft=d(N,"P",{"data-svelte-h":!0}),f(ft)!=="svelte-hswkmf"&&(ft.innerHTML=An),Vs=a(N),V=d(N,"DIV",{class:!0});var P=q(V);m(De.$$.fragment,P),Ds=a(P),yt=d(P,"P",{"data-svelte-h":!0}),f(yt)!=="svelte-18hwysu"&&(yt.innerHTML=Gn),Ss=a(P),m(ie.$$.fragment,P),P.forEach(n),N.forEach(n),ps=a(e),m(Se.$$.fragment,e),ms=a(e),v=d(e,"DIV",{class:!0});var z=q(v);m(He.$$.fragment,z),Hs=a(z),_t=d(z,"P",{"data-svelte-h":!0}),f(_t)!=="svelte-1q6c4il"&&(_t.textContent=Qn),Ys=a(z),Tt=d(z,"P",{"data-svelte-h":!0}),f(Tt)!=="svelte-q52n56"&&(Tt.innerHTML=Rn),Ls=a(z),jt=d(z,"P",{"data-svelte-h":!0}),f(jt)!=="svelte-hswkmf"&&(jt.innerHTML=Nn),Ps=a(z),D=d(z,"DIV",{class:!0});var It=q(D);m(Ye.$$.fragment,It),Ks=a(It),bt=d(It,"P",{"data-svelte-h":!0}),f(bt)!=="svelte-1hw2riq"&&(bt.innerHTML=zn),Os=a(It),m(ce.$$.fragment,It),It.forEach(n),z.forEach(n),hs=a(e),m(Le.$$.fragment,e),us=a(e),k=d(e,"DIV",{class:!0});var H=q(k);m(Pe.$$.fragment,H),en=a(H),Jt=d(H,"P",{"data-svelte-h":!0}),f(Jt)!=="svelte-1q6c4il"&&(Jt.textContent=En),tn=a(H),Ut=d(H,"P",{"data-svelte-h":!0}),f(Ut)!=="svelte-q52n56"&&(Ut.innerHTML=Wn),sn=a(H),wt=d(H,"P",{"data-svelte-h":!0}),f(wt)!=="svelte-hswkmf"&&(wt.innerHTML=Fn),nn=a(H),S=d(H,"DIV",{class:!0});var vt=q(S);m(Ke.$$.fragment,vt),on=a(vt),Ct=d(vt,"P",{"data-svelte-h":!0}),f(Ct)!=="svelte-99v5mm"&&(Ct.innerHTML=$n),an=a(vt),m(de.$$.fragment,vt),vt.forEach(n),H.forEach(n),Ms=a(e),m(Oe.$$.fragment,e),gs=a(e),kt=d(e,"P",{}),q(kt).forEach(n),this.h()},h(){U(s,"name","hf:doc:metadata"),U(s,"content",ro),U(K,"class","flex justify-center"),U(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(te,"class","flex justify-center"),U(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){r(document.head,s),i(e,y,t),i(e,l,t),i(e,_,t),i(e,b,t),i(e,W,t),h(pe,e,t),i(e,Zt,t),h(me,e,t),i(e,Bt,t),i(e,he,t),i(e,At,t),i(e,ue,t),i(e,Gt,t),i(e,Me,t),i(e,Qt,t),i(e,K,t),i(e,Rt,t),h(ge,e,t),i(e,Nt,t),h(fe,e,t),i(e,zt,t),i(e,ye,t),i(e,Et,t),h(_e,e,t),i(e,Wt,t),h(Te,e,t),i(e,Ft,t),i(e,je,t),i(e,$t,t),h(be,e,t),i(e,Xt,t),h(Je,e,t),i(e,Vt,t),i(e,Ue,t),i(e,Dt,t),h(we,e,t),i(e,St,t),h(Ce,e,t),i(e,Ht,t),i(e,Ie,t),i(e,Yt,t),h(ve,e,t),i(e,Lt,t),h(ke,e,t),i(e,Pt,t),i(e,xe,t),i(e,Kt,t),h(qe,e,t),i(e,Ot,t),i(e,Ze,t),i(e,es,t),h(Be,e,t),i(e,ts,t),i(e,w,t),h(Ae,w,null),r(w,ys),r(w,tt),r(w,_s),r(w,st),r(w,Ts),r(w,nt),r(w,js),h(O,w,null),i(e,ss,t),h(Ge,e,t),i(e,ns,t),i(e,C,t),h(Qe,C,null),r(C,bs),r(C,ot),r(C,Js),r(C,at),r(C,Us),r(C,lt),r(C,ws),h(ee,C,null),i(e,os,t),h(Re,e,t),i(e,as,t),i(e,te,t),i(e,ls,t),i(e,A,t),h(Ne,A,null),r(A,Cs),r(A,rt),r(A,Is),h(se,A,null),r(A,vs),r(A,ne),h(ze,ne,null),r(ne,ks),r(ne,it),i(e,rs,t),h(Ee,e,t),i(e,is,t),i(e,J,t),h(We,J,null),r(J,xs),r(J,ct),r(J,qs),r(J,dt),r(J,Zs),r(J,pt),r(J,Bs),r(J,F),h(Fe,F,null),r(F,As),r(F,mt),r(F,Gs),h(oe,F,null),r(F,Qs),h(ae,F,null),r(J,Rs),r(J,B),h($e,B,null),r(B,Ns),r(B,ht),r(B,zs),r(B,ut),r(B,Es),h(le,B,null),r(B,Ws),h(re,B,null),i(e,cs,t),h(Xe,e,t),i(e,ds,t),i(e,I,t),h(Ve,I,null),r(I,Fs),r(I,Mt),r(I,$s),r(I,gt),r(I,Xs),r(I,ft),r(I,Vs),r(I,V),h(De,V,null),r(V,Ds),r(V,yt),r(V,Ss),h(ie,V,null),i(e,ps,t),h(Se,e,t),i(e,ms,t),i(e,v,t),h(He,v,null),r(v,Hs),r(v,_t),r(v,Ys),r(v,Tt),r(v,Ls),r(v,jt),r(v,Ps),r(v,D),h(Ye,D,null),r(D,Ks),r(D,bt),r(D,Os),h(ce,D,null),i(e,hs,t),h(Le,e,t),i(e,us,t),i(e,k,t),h(Pe,k,null),r(k,en),r(k,Jt),r(k,tn),r(k,Ut),r(k,sn),r(k,wt),r(k,nn),r(k,S),h(Ke,S,null),r(S,on),r(S,Ct),r(S,an),h(de,S,null),i(e,Ms,t),h(Oe,e,t),i(e,gs,t),i(e,kt,t),fs=!0},p(e,[t]){const G={};t&2&&(G.$$scope={dirty:t,ctx:e}),O.$set(G);const Q={};t&2&&(Q.$$scope={dirty:t,ctx:e}),ee.$set(Q);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),se.$set($);const et={};t&2&&(et.$$scope={dirty:t,ctx:e}),oe.$set(et);const x={};t&2&&(x.$$scope={dirty:t,ctx:e}),ae.$set(x);const X={};t&2&&(X.$$scope={dirty:t,ctx:e}),le.$set(X);const R={};t&2&&(R.$$scope={dirty:t,ctx:e}),re.$set(R);const N={};t&2&&(N.$$scope={dirty:t,ctx:e}),ie.$set(N);const P={};t&2&&(P.$$scope={dirty:t,ctx:e}),ce.$set(P);const z={};t&2&&(z.$$scope={dirty:t,ctx:e}),de.$set(z)},i(e){fs||(u(pe.$$.fragment,e),u(me.$$.fragment,e),u(ge.$$.fragment,e),u(fe.$$.fragment,e),u(_e.$$.fragment,e),u(Te.$$.fragment,e),u(be.$$.fragment,e),u(Je.$$.fragment,e),u(we.$$.fragment,e),u(Ce.$$.fragment,e),u(ve.$$.fragment,e),u(ke.$$.fragment,e),u(qe.$$.fragment,e),u(Be.$$.fragment,e),u(Ae.$$.fragment,e),u(O.$$.fragment,e),u(Ge.$$.fragment,e),u(Qe.$$.fragment,e),u(ee.$$.fragment,e),u(Re.$$.fragment,e),u(Ne.$$.fragment,e),u(se.$$.fragment,e),u(ze.$$.fragment,e),u(Ee.$$.fragment,e),u(We.$$.fragment,e),u(Fe.$$.fragment,e),u(oe.$$.fragment,e),u(ae.$$.fragment,e),u($e.$$.fragment,e),u(le.$$.fragment,e),u(re.$$.fragment,e),u(Xe.$$.fragment,e),u(Ve.$$.fragment,e),u(De.$$.fragment,e),u(ie.$$.fragment,e),u(Se.$$.fragment,e),u(He.$$.fragment,e),u(Ye.$$.fragment,e),u(ce.$$.fragment,e),u(Le.$$.fragment,e),u(Pe.$$.fragment,e),u(Ke.$$.fragment,e),u(de.$$.fragment,e),u(Oe.$$.fragment,e),fs=!0)},o(e){M(pe.$$.fragment,e),M(me.$$.fragment,e),M(ge.$$.fragment,e),M(fe.$$.fragment,e),M(_e.$$.fragment,e),M(Te.$$.fragment,e),M(be.$$.fragment,e),M(Je.$$.fragment,e),M(we.$$.fragment,e),M(Ce.$$.fragment,e),M(ve.$$.fragment,e),M(ke.$$.fragment,e),M(qe.$$.fragment,e),M(Be.$$.fragment,e),M(Ae.$$.fragment,e),M(O.$$.fragment,e),M(Ge.$$.fragment,e),M(Qe.$$.fragment,e),M(ee.$$.fragment,e),M(Re.$$.fragment,e),M(Ne.$$.fragment,e),M(se.$$.fragment,e),M(ze.$$.fragment,e),M(Ee.$$.fragment,e),M(We.$$.fragment,e),M(Fe.$$.fragment,e),M(oe.$$.fragment,e),M(ae.$$.fragment,e),M($e.$$.fragment,e),M(le.$$.fragment,e),M(re.$$.fragment,e),M(Xe.$$.fragment,e),M(Ve.$$.fragment,e),M(De.$$.fragment,e),M(ie.$$.fragment,e),M(Se.$$.fragment,e),M(He.$$.fragment,e),M(Ye.$$.fragment,e),M(ce.$$.fragment,e),M(Le.$$.fragment,e),M(Pe.$$.fragment,e),M(Ke.$$.fragment,e),M(de.$$.fragment,e),M(Oe.$$.fragment,e),fs=!1},d(e){e&&(n(y),n(l),n(_),n(b),n(W),n(Zt),n(Bt),n(he),n(At),n(ue),n(Gt),n(Me),n(Qt),n(K),n(Rt),n(Nt),n(zt),n(ye),n(Et),n(Wt),n(Ft),n(je),n($t),n(Xt),n(Vt),n(Ue),n(Dt),n(St),n(Ht),n(Ie),n(Yt),n(Lt),n(Pt),n(xe),n(Kt),n(Ot),n(Ze),n(es),n(ts),n(w),n(ss),n(ns),n(C),n(os),n(as),n(te),n(ls),n(A),n(rs),n(is),n(J),n(cs),n(ds),n(I),n(ps),n(ms),n(v),n(hs),n(us),n(k),n(Ms),n(gs),n(kt)),n(s),g(pe,e),g(me,e),g(ge,e),g(fe,e),g(_e,e),g(Te,e),g(be,e),g(Je,e),g(we,e),g(Ce,e),g(ve,e),g(ke,e),g(qe,e),g(Be,e),g(Ae),g(O),g(Ge,e),g(Qe),g(ee),g(Re,e),g(Ne),g(se),g(ze),g(Ee,e),g(We),g(Fe),g(oe),g(ae),g($e),g(le),g(re),g(Xe,e),g(Ve),g(De),g(ie),g(Se,e),g(He),g(Ye),g(ce),g(Le,e),g(Pe),g(Ke),g(de),g(Oe,e)}}}const ro='{"title":"Csm","local":"csm","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage Tips","local":"usage-tips","sections":[{"title":"Without Conversational Context","local":"without-conversational-context","sections":[],"depth":3},{"title":"With Conversational Context","local":"with-conversational-context","sections":[],"depth":3},{"title":"Batched Inference","local":"batched-inference","sections":[],"depth":3},{"title":"Making The Model Go Brrr","local":"making-the-model-go-brrr","sections":[],"depth":3},{"title":"Training","local":"training","sections":[],"depth":3}],"depth":2},{"title":"CsmConfig","local":"transformers.CsmConfig","sections":[],"depth":2},{"title":"CsmDepthDecoderConfig","local":"transformers.CsmDepthDecoderConfig","sections":[],"depth":2},{"title":"CsmProcessor","local":"transformers.CsmProcessor","sections":[],"depth":2},{"title":"CsmForConditionalGeneration","local":"transformers.CsmForConditionalGeneration","sections":[],"depth":2},{"title":"CsmDepthDecoderForCausalLM","local":"transformers.CsmDepthDecoderForCausalLM","sections":[],"depth":2},{"title":"CsmDepthDecoderModel","local":"transformers.CsmDepthDecoderModel","sections":[],"depth":2},{"title":"CsmBackboneModel","local":"transformers.CsmBackboneModel","sections":[],"depth":2}],"depth":1}';function io(j){return Vn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class fo extends Dn{constructor(s){super(),Sn(this,s,io,lo,Xn,{})}}export{fo as component};
