import{s as zo,o as jo,n as N}from"../chunks/scheduler.18a86fab.js";import{S as Go,i as Io,g as m,s as d,r as f,A as Zo,h as u,f as r,c,j as R,x as v,u as g,k as W,l as qo,y as h,a as p,v as _,d as y,t as b,w as M}from"../chunks/index.98837b22.js";import{T as Ve}from"../chunks/Tip.77304350.js";import{D as pe}from"../chunks/Docstring.a1ef7999.js";import{C as $e}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ho}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Ee,E as Ro}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Wo,a as Fo}from"../chunks/HfOption.6641485e.js";function No(w){let o,l="Click on the Moonshine models in the right sidebar for more examples of how to apply Moonshine to different speech recognition tasks.";return{c(){o=m("p"),o.textContent=l},l(n){o=u(n,"P",{"data-svelte-h":!0}),v(o)!=="svelte-1bcugae"&&(o.textContent=l)},m(n,a){p(n,o,a)},p:N,d(n){n&&r(o)}}}function Xo(w){let o,l;return o=new $e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJhdXRvbWF0aWMtc3BlZWNoLXJlY29nbml0aW9uJTIyJTJDJTBBJTIwJTIwJTIwJTIwbW9kZWwlM0QlMjJVc2VmdWxTZW5zb3JzJTJGbW9vbnNoaW5lLWJhc2UlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUyMmh0dHBzJTNBJTJGJTJGaHVnZ2luZ2ZhY2UuY28lMkZkYXRhc2V0cyUyRk5hcnNpbCUyRmFzcl9kdW1teSUyRnJlc29sdmUlMkZtYWluJTJGbWxrLmZsYWMlMjIp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;automatic-speech-recognition&quot;</span>,
    model=<span class="hljs-string">&quot;UsefulSensors/moonshine-base&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac&quot;</span>)`,wrap:!1}}),{c(){f(o.$$.fragment)},l(n){g(o.$$.fragment,n)},m(n,a){_(o,n,a),l=!0},p:N,i(n){l||(y(o.$$.fragment,n),l=!0)},o(n){b(o.$$.fragment,n),l=!1},d(n){M(o,n)}}}function Vo(w){let o,l;return o=new $e({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMGRhdGFzZXRzJTBBaW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyME1vb25zaGluZUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMlVzZWZ1bFNlbnNvcnMlMkZtb29uc2hpbmUtYmFzZSUyMiUyQyUwQSklMEFtb2RlbCUyMCUzRCUyME1vb25zaGluZUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyVXNlZnVsU2Vuc29ycyUyRm1vb25zaGluZS1iYXNlJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEElMEFkcyUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGbGlicmlzcGVlY2hfYXNyX2R1bW15JTIyJTJDJTIwc3BsaXQlM0QlMjJ2YWxpZGF0aW9uJTIyKSUwQWF1ZGlvX3NhbXBsZSUyMCUzRCUyMGRzJTVCMCU1RCU1QiUyMmF1ZGlvJTIyJTVEJTBBJTBBaW5wdXRfZmVhdHVyZXMlMjAlM0QlMjBwcm9jZXNzb3IoJTBBJTIwJTIwJTIwJTIwYXVkaW9fc2FtcGxlJTVCJTIyYXJyYXklMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBzYW1wbGluZ19yYXRlJTNEYXVkaW9fc2FtcGxlJTVCJTIyc2FtcGxpbmdfcmF0ZSUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBaW5wdXRfZmVhdHVyZXMlMjAlM0QlMjBpbnB1dF9mZWF0dXJlcy50byhtb2RlbC5kZXZpY2UlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYpJTBBJTBBcHJlZGljdGVkX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfZmVhdHVyZXMlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEF0cmFuc2NyaXB0aW9uJTIwJTNEJTIwcHJvY2Vzc29yLmJhdGNoX2RlY29kZShwcmVkaWN0ZWRfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBdHJhbnNjcmlwdGlvbiU1QjAlNUQ=",highlighted:`<span class="hljs-comment"># pip install datasets</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MoonshineForConditionalGeneration

processor = AutoProcessor.from_pretrained(
    <span class="hljs-string">&quot;UsefulSensors/moonshine-base&quot;</span>,
)
model = MoonshineForConditionalGeneration.from_pretrained(
    <span class="hljs-string">&quot;UsefulSensors/moonshine-base&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
audio_sample = ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>]

input_features = processor(
    audio_sample[<span class="hljs-string">&quot;array&quot;</span>],
    sampling_rate=audio_sample[<span class="hljs-string">&quot;sampling_rate&quot;</span>],
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
)
input_features = input_features.to(model.device, dtype=torch.float16)

predicted_ids = model.generate(**input_features, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
transcription[<span class="hljs-number">0</span>]`,wrap:!1}}),{c(){f(o.$$.fragment)},l(n){g(o.$$.fragment,n)},m(n,a){_(o,n,a),l=!0},p:N,i(n){l||(y(o.$$.fragment,n),l=!0)},o(n){b(o.$$.fragment,n),l=!1},d(n){M(o,n)}}}function Eo(w){let o,l,n,a;return o=new Fo({props:{id:"usage",option:"Pipeline",$$slots:{default:[Xo]},$$scope:{ctx:w}}}),n=new Fo({props:{id:"usage",option:"AutoModel",$$slots:{default:[Vo]},$$scope:{ctx:w}}}),{c(){f(o.$$.fragment),l=d(),f(n.$$.fragment)},l(i){g(o.$$.fragment,i),l=c(i),g(n.$$.fragment,i)},m(i,t){_(o,i,t),p(i,l,t),_(n,i,t),a=!0},p(i,t){const T={};t&2&&(T.$$scope={dirty:t,ctx:i}),o.$set(T);const z={};t&2&&(z.$$scope={dirty:t,ctx:i}),n.$set(z)},i(i){a||(y(o.$$.fragment,i),y(n.$$.fragment,i),a=!0)},o(i){b(o.$$.fragment,i),b(n.$$.fragment,i),a=!1},d(i){i&&r(l),M(o,i),M(n,i)}}}function Ho(w){let o,l="Example:",n,a,i;return a=new $e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1vb25zaGluZU1vZGVsJTJDJTIwTW9vbnNoaW5lQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyME1vb25zaGluZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBNb29uc2hpbmVDb25maWcoKS5mcm9tX3ByZXRyYWluZWQoJTIyVXNlZnVsU2Vuc29ycyUyRm1vb25zaGluZS10aW55JTIyKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBNb29uc2hpbmVNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MoonshineModel, MoonshineConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Moonshine style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MoonshineConfig().from_pretrained(<span class="hljs-string">&quot;UsefulSensors/moonshine-tiny&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MoonshineModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){o=m("p"),o.textContent=l,n=d(),f(a.$$.fragment)},l(t){o=u(t,"P",{"data-svelte-h":!0}),v(o)!=="svelte-11lpom8"&&(o.textContent=l),n=c(t),g(a.$$.fragment,t)},m(t,T){p(t,o,T),p(t,n,T),_(a,t,T),i=!0},p:N,i(t){i||(y(a.$$.fragment,t),i=!0)},o(t){b(a.$$.fragment,t),i=!1},d(t){t&&(r(o),r(n)),M(a,t)}}}function Bo(w){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=m("p"),o.innerHTML=l},l(n){o=u(n,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(n,a){p(n,o,a)},p:N,d(n){n&&r(o)}}}function Lo(w){let o,l="Example:",n,a,i;return a=new $e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b0ZlYXR1cmVFeHRyYWN0b3IlMkMlMjBNb29uc2hpbmVNb2RlbCUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQSUwQW1vZGVsJTIwJTNEJTIwTW9vbnNoaW5lTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMlVzZWZ1bFNlbnNvcnMlMkZtb29uc2hpbmUtdGlueSUyMiklMEFmZWF0dXJlX2V4dHJhY3RvciUyMCUzRCUyMEF1dG9GZWF0dXJlRXh0cmFjdG9yLmZyb21fcHJldHJhaW5lZCglMjJVc2VmdWxTZW5zb3JzJTJGbW9vbnNoaW5lLXRpbnklMjIpJTBBZHMlMjAlM0QlMjBsb2FkX2RhdGFzZXQoJTIyaGYtaW50ZXJuYWwtdGVzdGluZyUyRmxpYnJpc3BlZWNoX2Fzcl9kdW1teSUyMiUyQyUyMCUyMmNsZWFuJTIyJTJDJTIwc3BsaXQlM0QlMjJ2YWxpZGF0aW9uJTIyKSUwQWlucHV0cyUyMCUzRCUyMGZlYXR1cmVfZXh0cmFjdG9yKGRzJTVCMCU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWlucHV0X3ZhbHVlcyUyMCUzRCUyMGlucHV0cy5pbnB1dF92YWx1ZXMlMEFkZWNvZGVyX2lucHV0X2lkcyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIlNUIxJTJDJTIwMSU1RCU1RCklMjAqJTIwbW9kZWwuY29uZmlnLmRlY29kZXJfc3RhcnRfdG9rZW5faWQlMEFsYXN0X2hpZGRlbl9zdGF0ZSUyMCUzRCUyMG1vZGVsKGlucHV0X3ZhbHVlcyUyQyUyMGRlY29kZXJfaW5wdXRfaWRzJTNEZGVjb2Rlcl9pbnB1dF9pZHMpLmxhc3RfaGlkZGVuX3N0YXRlJTBBbGlzdChsYXN0X2hpZGRlbl9zdGF0ZS5zaGFwZSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoFeatureExtractor, MoonshineModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MoonshineModel.from_pretrained(<span class="hljs-string">&quot;UsefulSensors/moonshine-tiny&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = AutoFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;UsefulSensors/moonshine-tiny&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = inputs.input_values
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = torch.tensor([[<span class="hljs-number">1</span>, <span class="hljs-number">1</span>]]) * model.config.decoder_start_token_id
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = model(input_values, decoder_input_ids=decoder_input_ids).last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_state.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">2</span>, <span class="hljs-number">288</span>]`,wrap:!1}}),{c(){o=m("p"),o.textContent=l,n=d(),f(a.$$.fragment)},l(t){o=u(t,"P",{"data-svelte-h":!0}),v(o)!=="svelte-11lpom8"&&(o.textContent=l),n=c(t),g(a.$$.fragment,t)},m(t,T){p(t,o,T),p(t,n,T),_(a,t,T),i=!0},p:N,i(t){i||(y(a.$$.fragment,t),i=!0)},o(t){b(a.$$.fragment,t),i=!1},d(t){t&&(r(o),r(n)),M(a,t)}}}function So(w){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=m("p"),o.innerHTML=l},l(n){o=u(n,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(n,a){p(n,o,a)},p:N,d(n){n&&r(o)}}}function Po(w){let o,l="Example:",n,a,i;return a=new $e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyME1vb25zaGluZUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMlVzZWZ1bFNlbnNvcnMlMkZtb29uc2hpbmUtdGlueSUyMiklMEFtb2RlbCUyMCUzRCUyME1vb25zaGluZUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyVXNlZnVsU2Vuc29ycyUyRm1vb25zaGluZS10aW55JTIyKSUwQSUwQWRzJTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZsaWJyaXNwZWVjaF9hc3JfZHVtbXklMjIlMkMlMjAlMjJjbGVhbiUyMiUyQyUyMHNwbGl0JTNEJTIydmFsaWRhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IoZHMlNUIwJTVEJTVCJTIyYXVkaW8lMjIlNUQlNUIlMjJhcnJheSUyMiU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBaW5wdXRfdmFsdWVzJTIwJTNEJTIwaW5wdXRzLmlucHV0X3ZhbHVlcyUwQSUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dF92YWx1ZXMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDEwMCklMEElMEF0cmFuc2NyaXB0aW9uJTIwJTNEJTIwcHJvY2Vzc29yLmJhdGNoX2RlY29kZShnZW5lcmF0ZWRfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTVCMCU1RCUwQXRyYW5zY3JpcHRpb24=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MoonshineForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;UsefulSensors/moonshine-tiny&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MoonshineForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;UsefulSensors/moonshine-tiny&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_values = inputs.input_values

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(input_values, max_new_tokens=<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription
<span class="hljs-string">&#x27;Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.&#x27;</span>`,wrap:!1}}),{c(){o=m("p"),o.textContent=l,n=d(),f(a.$$.fragment)},l(t){o=u(t,"P",{"data-svelte-h":!0}),v(o)!=="svelte-11lpom8"&&(o.textContent=l),n=c(t),g(a.$$.fragment,t)},m(t,T){p(t,o,T),p(t,n,T),_(a,t,T),i=!0},p:N,i(t){i||(y(a.$$.fragment,t),i=!0)},o(t){b(a.$$.fragment,t),i=!1},d(t){t&&(r(o),r(n)),M(a,t)}}}function Qo(w){let o,l=`Most generation-controlling parameters are set in <code>generation_config</code> which, if not passed, will be set to the
model’s default generation configuration. You can override any <code>generation_config</code> by passing the corresponding
parameters to generate(), e.g. <code>.generate(inputs, num_beams=4, do_sample=True)</code>.`,n,a,i=`For an overview of generation strategies and code examples, check out the <a href="../generation_strategies">following
guide</a>.`;return{c(){o=m("p"),o.innerHTML=l,n=d(),a=m("p"),a.innerHTML=i},l(t){o=u(t,"P",{"data-svelte-h":!0}),v(o)!=="svelte-1c5u34l"&&(o.innerHTML=l),n=c(t),a=u(t,"P",{"data-svelte-h":!0}),v(a)!=="svelte-fvlq1g"&&(a.innerHTML=i)},m(t,T){p(t,o,T),p(t,n,T),p(t,a,T)},p:N,d(t){t&&(r(o),r(n),r(a))}}}function Ao(w){let o,l,n,a,i,t="<em>This model was released on 2024-10-21 and added to Hugging Face Transformers on 2025-01-10.</em>",T,z,mo='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',xe,A,Ce,Y,uo='<a href="https://huggingface.co/papers/2410.15608" rel="nofollow">Moonshine</a> is an encoder-decoder speech recognition model optimized for real-time transcription and recognizing voice command. Instead of using traditional absolute position embeddings, Moonshine uses Rotary Position Embedding (RoPE) to handle speech with varying lengths without using padding. This improves efficiency during inference, making it ideal for resource-constrained devices.',Je,O,fo='You can find all the original Moonshine checkpoints under the <a href="https://huggingface.co/UsefulSensors" rel="nofollow">Useful Sensors</a> organization.',Ue,X,Fe,D,go='The example below demonstrates how to transcribe speech into text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',ze,V,je,K,Ge,J,ee,He,he,_o=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineModel">MoonshineModel</a>. It is used to instantiate a Moonshine
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Moonshine
<a href="https://huggingface.co/UsefulSensors/moonshine-tiny" rel="nofollow">UsefulSensors/moonshine-tiny</a>.`,Be,me,yo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Le,E,Ie,oe,Ze,k,ne,Se,ue,bo="The bare Moonshine Model outputting raw hidden-states without any specific head on top.",Pe,fe,Mo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qe,ge,vo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ae,U,te,Ye,_e,To='The <a href="/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineModel">MoonshineModel</a> forward method, overrides the <code>__call__</code> special method.',Oe,H,De,B,Ke,L,se,eo,ye,wo=`Masks extracted features along time axis and/or along feature axis according to
<a href="https://huggingface.co/papers/1904.08779" rel="nofollow">SpecAugment</a>.`,qe,ae,Re,$,re,oo,be,ko="The Moonshine Model with a language modeling head. Can be used for automatic speech recognition.",no,Me,$o=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,to,ve,xo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,so,F,ie,ao,Te,Co='The <a href="/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineForConditionalGeneration">MoonshineForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',ro,S,io,P,co,Z,de,lo,we,Jo="Generates sequences of token ids for models with a language modeling head.",po,Q,We,ce,Ne,ke,Xe;return A=new Ee({props:{title:"Moonshine",local:"moonshine",headingTag:"h1"}}),X=new Ve({props:{warning:!1,$$slots:{default:[No]},$$scope:{ctx:w}}}),V=new Wo({props:{id:"usage",options:["Pipeline","AutoModel"],$$slots:{default:[Eo]},$$scope:{ctx:w}}}),K=new Ee({props:{title:"MoonshineConfig",local:"transformers.MoonshineConfig",headingTag:"h2"}}),ee=new pe({props:{name:"class transformers.MoonshineConfig",anchor:"transformers.MoonshineConfig",parameters:[{name:"vocab_size",val:" = 32768"},{name:"hidden_size",val:" = 288"},{name:"intermediate_size",val:" = 1152"},{name:"encoder_num_hidden_layers",val:" = 6"},{name:"decoder_num_hidden_layers",val:" = 6"},{name:"encoder_num_attention_heads",val:" = 8"},{name:"decoder_num_attention_heads",val:" = 8"},{name:"encoder_num_key_value_heads",val:" = None"},{name:"decoder_num_key_value_heads",val:" = None"},{name:"pad_head_dim_to_multiple_of",val:" = None"},{name:"encoder_hidden_act",val:" = 'gelu'"},{name:"decoder_hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 512"},{name:"initializer_range",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 1"},{name:"use_cache",val:" = True"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"partial_rotary_factor",val:" = 0.9"},{name:"is_encoder_decoder",val:" = True"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MoonshineConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32768) &#x2014;
Vocabulary size of the Moonshine model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineModel">MoonshineModel</a>.`,name:"vocab_size"},{anchor:"transformers.MoonshineConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 288) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.MoonshineConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1152) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.MoonshineConfig.encoder_num_hidden_layers",description:`<strong>encoder_num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"encoder_num_hidden_layers"},{anchor:"transformers.MoonshineConfig.decoder_num_hidden_layers",description:`<strong>decoder_num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"decoder_num_hidden_layers"},{anchor:"transformers.MoonshineConfig.encoder_num_attention_heads",description:`<strong>encoder_num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_num_attention_heads"},{anchor:"transformers.MoonshineConfig.decoder_num_attention_heads",description:`<strong>decoder_num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_num_attention_heads"},{anchor:"transformers.MoonshineConfig.encoder_num_key_value_heads",description:`<strong>encoder_num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>encoder_num_key_value_heads=encoder_num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>encoder_num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"encoder_num_key_value_heads"},{anchor:"transformers.MoonshineConfig.decoder_num_key_value_heads",description:`<strong>decoder_num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>decoder_num_key_value_heads=decoder_num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>decoder_num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>decoder_num_attention_heads</code>.`,name:"decoder_num_key_value_heads"},{anchor:"transformers.MoonshineConfig.pad_head_dim_to_multiple_of",description:`<strong>pad_head_dim_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Pad head dimension in encoder and decoder to the next multiple of this value. Necessary for using certain
optimized attention implementations.`,name:"pad_head_dim_to_multiple_of"},{anchor:"transformers.MoonshineConfig.encoder_hidden_act",description:`<strong>encoder_hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder.`,name:"encoder_hidden_act"},{anchor:"transformers.MoonshineConfig.decoder_hidden_act",description:`<strong>decoder_hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"decoder_hidden_act"},{anchor:"transformers.MoonshineConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.MoonshineConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.MoonshineConfig.decoder_start_token_id",description:`<strong>decoder_start_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Corresponds to the &#x201D;&lt;|startoftranscript|&gt;&#x201D; token, which is automatically used when no <code>decoder_input_ids</code>
are provided to the <code>generate</code> function. It is used to guide the model\`s generation process depending on
the task.`,name:"decoder_start_token_id"},{anchor:"transformers.MoonshineConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.MoonshineConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.MoonshineConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.MoonshineConfig.partial_rotary_factor",description:`<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.9) &#x2014;
Percentage of the query and keys which will have rotary embedding.`,name:"partial_rotary_factor"},{anchor:"transformers.MoonshineConfig.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the model is used as an encoder/decoder or not.`,name:"is_encoder_decoder"},{anchor:"transformers.MoonshineConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.MoonshineConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.MoonshineConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Denotes beginning of sequences token id.`,name:"bos_token_id"},{anchor:"transformers.MoonshineConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Denotes end of sequences token id.`,name:"eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/configuration_moonshine.py#L25"}}),E=new ho({props:{anchor:"transformers.MoonshineConfig.example",$$slots:{default:[Ho]},$$scope:{ctx:w}}}),oe=new Ee({props:{title:"MoonshineModel",local:"transformers.MoonshineModel",headingTag:"h2"}}),ne=new pe({props:{name:"class transformers.MoonshineModel",anchor:"transformers.MoonshineModel",parameters:[{name:"config",val:": MoonshineConfig"}],parametersDescription:[{anchor:"transformers.MoonshineModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig">MoonshineConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L819"}}),te=new pe({props:{name:"forward",anchor:"transformers.MoonshineModel.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.EncoderDecoderCache, tuple[torch.FloatTensor], NoneType] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_position_ids",val:": typing.Optional[tuple[torch.LongTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.MoonshineModel.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, audio_length)</code>) &#x2014;
Float values of the raw speech waveform. Raw speech waveform can be
obtained by loading a <code>.flac</code> or <code>.wav</code> audio file into an array of type <code>list[float]</code>, a
<code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library (<code>pip install torchcodec</code>) or
the soundfile library (<code>pip install soundfile</code>). To prepare the array into
<code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoFeatureExtractor">AutoFeatureExtractor</a> should be used for padding
and conversion into a tensor of type <code>torch.FloatTensor</code>.`,name:"input_values"},{anchor:"transformers.MoonshineModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MoonshineModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.MoonshineModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.MoonshineModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MoonshineModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.EncoderDecoderCache, tuple[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MoonshineModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>tuple[torch.FloatTensor]</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MoonshineModel.forward.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings.
Used to calculate the position embeddings up to <code>config.decoder_config.max_position_embeddings</code>`,name:"decoder_position_ids"},{anchor:"transformers.MoonshineModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MoonshineModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L887",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig"
>MoonshineConfig</a>) and inputs.</p>
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
`}}),H=new Ve({props:{$$slots:{default:[Bo]},$$scope:{ctx:w}}}),B=new ho({props:{anchor:"transformers.MoonshineModel.forward.example",$$slots:{default:[Lo]},$$scope:{ctx:w}}}),se=new pe({props:{name:"_mask_input_features",anchor:"transformers.MoonshineModel._mask_input_features",parameters:[{name:"input_features",val:": FloatTensor"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L844"}}),ae=new Ee({props:{title:"MoonshineForConditionalGeneration",local:"transformers.MoonshineForConditionalGeneration",headingTag:"h2"}}),re=new pe({props:{name:"class transformers.MoonshineForConditionalGeneration",anchor:"transformers.MoonshineForConditionalGeneration",parameters:[{name:"config",val:": MoonshineConfig"}],parametersDescription:[{anchor:"transformers.MoonshineForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig">MoonshineConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L982"}}),ie=new pe({props:{name:"forward",anchor:"transformers.MoonshineForConditionalGeneration.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.EncoderDecoderCache, tuple[torch.FloatTensor], NoneType] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_position_ids",val:": typing.Optional[tuple[torch.LongTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.MoonshineForConditionalGeneration.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, audio_length)</code>) &#x2014;
Float values of the raw speech waveform. Raw speech waveform can be
obtained by loading a <code>.flac</code> or <code>.wav</code> audio file into an array of type <code>list[float]</code>, a
<code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library (<code>pip install torchcodec</code>) or
the soundfile library (<code>pip install soundfile</code>). To prepare the array into
<code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoFeatureExtractor">AutoFeatureExtractor</a> should be used for padding
and conversion into a tensor of type <code>torch.FloatTensor</code>.`,name:"input_values"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.EncoderDecoderCache, tuple[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>tuple[torch.FloatTensor]</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings.
Used to calculate the position embeddings up to <code>config.decoder_config.max_position_embeddings</code>`,name:"decoder_position_ids"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.MoonshineForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/moonshine/modeling_moonshine.py#L1008",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/moonshine#transformers.MoonshineConfig"
>MoonshineConfig</a>) and inputs.</p>
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
`}}),S=new Ve({props:{$$slots:{default:[So]},$$scope:{ctx:w}}}),P=new ho({props:{anchor:"transformers.MoonshineForConditionalGeneration.forward.example",$$slots:{default:[Po]},$$scope:{ctx:w}}}),de=new pe({props:{name:"generate",anchor:"transformers.MoonshineForConditionalGeneration.generate",parameters:[{name:"inputs",val:": typing.Optional[torch.Tensor] = None"},{name:"generation_config",val:": typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None"},{name:"logits_processor",val:": typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None"},{name:"stopping_criteria",val:": typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None"},{name:"prefix_allowed_tokens_fn",val:": typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None"},{name:"synced_gpus",val:": typing.Optional[bool] = None"},{name:"assistant_model",val:": typing.Optional[ForwardRef('PreTrainedModel')] = None"},{name:"streamer",val:": typing.Optional[ForwardRef('BaseStreamer')] = None"},{name:"negative_prompt_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"negative_prompt_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"use_model_defaults",val:": typing.Optional[bool] = None"},{name:"custom_generate",val:": typing.Union[str, typing.Callable, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MoonshineForConditionalGeneration.generate.inputs",description:`<strong>inputs</strong> (<code>torch.Tensor</code> of varying shape depending on the modality, <em>optional</em>) &#x2014;
The sequence used as a prompt for the generation or as model inputs to the encoder. If <code>None</code> the
method initializes it with <code>bos_token_id</code> and a batch size of 1. For decoder-only models <code>inputs</code>
should be in the format of <code>input_ids</code>. For encoder-decoder models <em>inputs</em> can represent any of
<code>input_ids</code>, <code>input_values</code>, <code>input_features</code>, or <code>pixel_values</code>.`,name:"inputs"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.generation_config",description:`<strong>generation_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>, <em>optional</em>) &#x2014;
The generation configuration to be used as base parametrization for the generation call. <code>**kwargs</code>
passed to generate matching the attributes of <code>generation_config</code> will override them. If
<code>generation_config</code> is not provided, the default will be used, which has the following loading
priority: 1) from the <code>generation_config.json</code> model file, if it exists; 2) from the model
configuration. Please note that unspecified parameters will inherit <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>&#x2019;s
default values, whose documentation should be checked to parameterize generation.`,name:"generation_config"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.logits_processor",description:`<strong>logits_processor</strong> (<code>LogitsProcessorList</code>, <em>optional</em>) &#x2014;
Custom logits processors that complement the default logits processors built from arguments and
generation config. If a logit processor is passed that is already created with the arguments or a
generation config an error is thrown. This feature is intended for advanced users.`,name:"logits_processor"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.stopping_criteria",description:`<strong>stopping_criteria</strong> (<code>StoppingCriteriaList</code>, <em>optional</em>) &#x2014;
Custom stopping criteria that complements the default stopping criteria built from arguments and a
generation config. If a stopping criteria is passed that is already created with the arguments or a
generation config an error is thrown. If your stopping criteria depends on the <code>scores</code> input, make
sure you pass <code>return_dict_in_generate=True, output_scores=True</code> to <code>generate</code>. This feature is
intended for advanced users.`,name:"stopping_criteria"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.prefix_allowed_tokens_fn",description:`<strong>prefix_allowed_tokens_fn</strong> (<code>Callable[[int, torch.Tensor], list[int]]</code>, <em>optional</em>) &#x2014;
If provided, this function constraints the beam search to allowed tokens only at each step. If not
provided no constraint is applied. This function takes 2 arguments: the batch ID <code>batch_id</code> and
<code>input_ids</code>. It has to return a list with the allowed tokens for the next generation step conditioned
on the batch ID <code>batch_id</code> and the previously generated tokens <code>inputs_ids</code>. This argument is useful
for constrained generation conditioned on the prefix, as described in <a href="https://huggingface.co/papers/2010.00904" rel="nofollow">Autoregressive Entity
Retrieval</a>.`,name:"prefix_allowed_tokens_fn"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.synced_gpus",description:`<strong>synced_gpus</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
to <code>True</code> if using <code>FullyShardedDataParallel</code> or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to <code>False</code>.`,name:"synced_gpus"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.assistant_model",description:`<strong>assistant_model</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
An assistant model that can be used to accelerate generation. The assistant model must have the exact
same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
is much faster than running generation with the model you&#x2019;re calling generate from. As such, the
assistant model should be much smaller.`,name:"assistant_model"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.streamer",description:`<strong>streamer</strong> (<code>BaseStreamer</code>, <em>optional</em>) &#x2014;
Streamer object that will be used to stream the generated sequences. Generated tokens are passed
through <code>streamer.put(token_ids)</code> and the streamer is responsible for any further processing.`,name:"streamer"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.negative_prompt_ids",description:`<strong>negative_prompt_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
The negative prompt needed for some processors such as CFG. The batch size must match the input batch
size. This is an experimental feature, subject to breaking API changes in future versions.`,name:"negative_prompt_ids"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.negative_prompt_attention_mask",description:`<strong>negative_prompt_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Attention_mask for <code>negative_prompt_ids</code>.`,name:"negative_prompt_attention_mask"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.use_model_defaults",description:`<strong>use_model_defaults</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
When it is <code>True</code>, unset parameters in <code>generation_config</code> will be set to the model-specific default
generation configuration (<code>model.generation_config</code>), as opposed to the global defaults
(<code>GenerationConfig()</code>). If unset, models saved starting from <code>v4.50</code> will consider this flag to be
<code>True</code>.`,name:"use_model_defaults"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.custom_generate",description:`<strong>custom_generate</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>) &#x2014;
One of the following:<ul>
<li><code>str</code> (Hugging Face Hub repository name): runs the custom <code>generate</code> function defined at
<code>custom_generate/generate.py</code> in that repository instead of the standard <code>generate</code> method. The
repository fully replaces the generation logic, and the return type may differ.</li>
<li><code>str</code> (local repository path): same as above but from a local path, <code>trust_remote_code</code> not required.</li>
<li><code>Callable</code>: <code>generate</code> will perform the usual input preparation steps, then call the provided callable to
run the decoding loop.
For more information, see <a href="../../generation_strategies#custom-generation-methods">the docs</a>.</li>
</ul>`,name:"custom_generate"},{anchor:"transformers.MoonshineForConditionalGeneration.generate.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Ad hoc parametrization of <code>generation_config</code> and/or additional model-specific kwargs that will be
forwarded to the <code>forward</code> function of the model. If the model is an encoder-decoder model, encoder
specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with <em>decoder_</em>.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L2140",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> (if <code>return_dict_in_generate=True</code>
or when <code>config.return_dict_in_generate=True</code>) or a <code>torch.LongTensor</code>.</p>
<p>If the model is <em>not</em> an encoder-decoder model (<code>model.config.is_encoder_decoder=False</code>), the possible
<a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> types are:</p>
<ul>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput"
>GenerateDecoderOnlyOutput</a>,</li>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput"
>GenerateBeamDecoderOnlyOutput</a></li>
</ul>
<p>If the model is an encoder-decoder model (<code>model.config.is_encoder_decoder=True</code>), the possible
<a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> types are:</p>
<ul>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput"
>GenerateEncoderDecoderOutput</a>,</li>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput"
>GenerateBeamEncoderDecoderOutput</a></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> or <code>torch.LongTensor</code></p>
`}}),Q=new Ve({props:{warning:!0,$$slots:{default:[Qo]},$$scope:{ctx:w}}}),ce=new Ro({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/moonshine.md"}}),{c(){o=m("meta"),l=d(),n=m("p"),a=d(),i=m("p"),i.innerHTML=t,T=d(),z=m("div"),z.innerHTML=mo,xe=d(),f(A.$$.fragment),Ce=d(),Y=m("p"),Y.innerHTML=uo,Je=d(),O=m("p"),O.innerHTML=fo,Ue=d(),f(X.$$.fragment),Fe=d(),D=m("p"),D.innerHTML=go,ze=d(),f(V.$$.fragment),je=d(),f(K.$$.fragment),Ge=d(),J=m("div"),f(ee.$$.fragment),He=d(),he=m("p"),he.innerHTML=_o,Be=d(),me=m("p"),me.innerHTML=yo,Le=d(),f(E.$$.fragment),Ie=d(),f(oe.$$.fragment),Ze=d(),k=m("div"),f(ne.$$.fragment),Se=d(),ue=m("p"),ue.textContent=bo,Pe=d(),fe=m("p"),fe.innerHTML=Mo,Qe=d(),ge=m("p"),ge.innerHTML=vo,Ae=d(),U=m("div"),f(te.$$.fragment),Ye=d(),_e=m("p"),_e.innerHTML=To,Oe=d(),f(H.$$.fragment),De=d(),f(B.$$.fragment),Ke=d(),L=m("div"),f(se.$$.fragment),eo=d(),ye=m("p"),ye.innerHTML=wo,qe=d(),f(ae.$$.fragment),Re=d(),$=m("div"),f(re.$$.fragment),oo=d(),be=m("p"),be.textContent=ko,no=d(),Me=m("p"),Me.innerHTML=$o,to=d(),ve=m("p"),ve.innerHTML=xo,so=d(),F=m("div"),f(ie.$$.fragment),ao=d(),Te=m("p"),Te.innerHTML=Co,ro=d(),f(S.$$.fragment),io=d(),f(P.$$.fragment),co=d(),Z=m("div"),f(de.$$.fragment),lo=d(),we=m("p"),we.textContent=Jo,po=d(),f(Q.$$.fragment),We=d(),f(ce.$$.fragment),Ne=d(),ke=m("p"),this.h()},l(e){const s=Zo("svelte-u9bgzb",document.head);o=u(s,"META",{name:!0,content:!0}),s.forEach(r),l=c(e),n=u(e,"P",{}),R(n).forEach(r),a=c(e),i=u(e,"P",{"data-svelte-h":!0}),v(i)!=="svelte-u5g6ul"&&(i.innerHTML=t),T=c(e),z=u(e,"DIV",{style:!0,"data-svelte-h":!0}),v(z)!=="svelte-1heeauf"&&(z.innerHTML=mo),xe=c(e),g(A.$$.fragment,e),Ce=c(e),Y=u(e,"P",{"data-svelte-h":!0}),v(Y)!=="svelte-16rfiku"&&(Y.innerHTML=uo),Je=c(e),O=u(e,"P",{"data-svelte-h":!0}),v(O)!=="svelte-m58aij"&&(O.innerHTML=fo),Ue=c(e),g(X.$$.fragment,e),Fe=c(e),D=u(e,"P",{"data-svelte-h":!0}),v(D)!=="svelte-15b6q9g"&&(D.innerHTML=go),ze=c(e),g(V.$$.fragment,e),je=c(e),g(K.$$.fragment,e),Ge=c(e),J=u(e,"DIV",{class:!0});var j=R(J);g(ee.$$.fragment,j),He=c(j),he=u(j,"P",{"data-svelte-h":!0}),v(he)!=="svelte-1si1oxg"&&(he.innerHTML=_o),Be=c(j),me=u(j,"P",{"data-svelte-h":!0}),v(me)!=="svelte-1ek1ss9"&&(me.innerHTML=yo),Le=c(j),g(E.$$.fragment,j),j.forEach(r),Ie=c(e),g(oe.$$.fragment,e),Ze=c(e),k=u(e,"DIV",{class:!0});var x=R(k);g(ne.$$.fragment,x),Se=c(x),ue=u(x,"P",{"data-svelte-h":!0}),v(ue)!=="svelte-1609zgi"&&(ue.textContent=bo),Pe=c(x),fe=u(x,"P",{"data-svelte-h":!0}),v(fe)!=="svelte-q52n56"&&(fe.innerHTML=Mo),Qe=c(x),ge=u(x,"P",{"data-svelte-h":!0}),v(ge)!=="svelte-hswkmf"&&(ge.innerHTML=vo),Ae=c(x),U=u(x,"DIV",{class:!0});var G=R(U);g(te.$$.fragment,G),Ye=c(G),_e=u(G,"P",{"data-svelte-h":!0}),v(_e)!=="svelte-d0yapd"&&(_e.innerHTML=To),Oe=c(G),g(H.$$.fragment,G),De=c(G),g(B.$$.fragment,G),G.forEach(r),Ke=c(x),L=u(x,"DIV",{class:!0});var le=R(L);g(se.$$.fragment,le),eo=c(le),ye=u(le,"P",{"data-svelte-h":!0}),v(ye)!=="svelte-i8aa35"&&(ye.innerHTML=wo),le.forEach(r),x.forEach(r),qe=c(e),g(ae.$$.fragment,e),Re=c(e),$=u(e,"DIV",{class:!0});var C=R($);g(re.$$.fragment,C),oo=c(C),be=u(C,"P",{"data-svelte-h":!0}),v(be)!=="svelte-1lctgbj"&&(be.textContent=ko),no=c(C),Me=u(C,"P",{"data-svelte-h":!0}),v(Me)!=="svelte-q52n56"&&(Me.innerHTML=$o),to=c(C),ve=u(C,"P",{"data-svelte-h":!0}),v(ve)!=="svelte-hswkmf"&&(ve.innerHTML=xo),so=c(C),F=u(C,"DIV",{class:!0});var I=R(F);g(ie.$$.fragment,I),ao=c(I),Te=u(I,"P",{"data-svelte-h":!0}),v(Te)!=="svelte-15vtk3z"&&(Te.innerHTML=Co),ro=c(I),g(S.$$.fragment,I),io=c(I),g(P.$$.fragment,I),I.forEach(r),co=c(C),Z=u(C,"DIV",{class:!0});var q=R(Z);g(de.$$.fragment,q),lo=c(q),we=u(q,"P",{"data-svelte-h":!0}),v(we)!=="svelte-s5ko3x"&&(we.textContent=Jo),po=c(q),g(Q.$$.fragment,q),q.forEach(r),C.forEach(r),We=c(e),g(ce.$$.fragment,e),Ne=c(e),ke=u(e,"P",{}),R(ke).forEach(r),this.h()},h(){W(o,"name","hf:doc:metadata"),W(o,"content",Yo),qo(z,"float","right"),W(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){h(document.head,o),p(e,l,s),p(e,n,s),p(e,a,s),p(e,i,s),p(e,T,s),p(e,z,s),p(e,xe,s),_(A,e,s),p(e,Ce,s),p(e,Y,s),p(e,Je,s),p(e,O,s),p(e,Ue,s),_(X,e,s),p(e,Fe,s),p(e,D,s),p(e,ze,s),_(V,e,s),p(e,je,s),_(K,e,s),p(e,Ge,s),p(e,J,s),_(ee,J,null),h(J,He),h(J,he),h(J,Be),h(J,me),h(J,Le),_(E,J,null),p(e,Ie,s),_(oe,e,s),p(e,Ze,s),p(e,k,s),_(ne,k,null),h(k,Se),h(k,ue),h(k,Pe),h(k,fe),h(k,Qe),h(k,ge),h(k,Ae),h(k,U),_(te,U,null),h(U,Ye),h(U,_e),h(U,Oe),_(H,U,null),h(U,De),_(B,U,null),h(k,Ke),h(k,L),_(se,L,null),h(L,eo),h(L,ye),p(e,qe,s),_(ae,e,s),p(e,Re,s),p(e,$,s),_(re,$,null),h($,oo),h($,be),h($,no),h($,Me),h($,to),h($,ve),h($,so),h($,F),_(ie,F,null),h(F,ao),h(F,Te),h(F,ro),_(S,F,null),h(F,io),_(P,F,null),h($,co),h($,Z),_(de,Z,null),h(Z,lo),h(Z,we),h(Z,po),_(Q,Z,null),p(e,We,s),_(ce,e,s),p(e,Ne,s),p(e,ke,s),Xe=!0},p(e,[s]){const j={};s&2&&(j.$$scope={dirty:s,ctx:e}),X.$set(j);const x={};s&2&&(x.$$scope={dirty:s,ctx:e}),V.$set(x);const G={};s&2&&(G.$$scope={dirty:s,ctx:e}),E.$set(G);const le={};s&2&&(le.$$scope={dirty:s,ctx:e}),H.$set(le);const C={};s&2&&(C.$$scope={dirty:s,ctx:e}),B.$set(C);const I={};s&2&&(I.$$scope={dirty:s,ctx:e}),S.$set(I);const q={};s&2&&(q.$$scope={dirty:s,ctx:e}),P.$set(q);const Uo={};s&2&&(Uo.$$scope={dirty:s,ctx:e}),Q.$set(Uo)},i(e){Xe||(y(A.$$.fragment,e),y(X.$$.fragment,e),y(V.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(E.$$.fragment,e),y(oe.$$.fragment,e),y(ne.$$.fragment,e),y(te.$$.fragment,e),y(H.$$.fragment,e),y(B.$$.fragment,e),y(se.$$.fragment,e),y(ae.$$.fragment,e),y(re.$$.fragment,e),y(ie.$$.fragment,e),y(S.$$.fragment,e),y(P.$$.fragment,e),y(de.$$.fragment,e),y(Q.$$.fragment,e),y(ce.$$.fragment,e),Xe=!0)},o(e){b(A.$$.fragment,e),b(X.$$.fragment,e),b(V.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(E.$$.fragment,e),b(oe.$$.fragment,e),b(ne.$$.fragment,e),b(te.$$.fragment,e),b(H.$$.fragment,e),b(B.$$.fragment,e),b(se.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(ie.$$.fragment,e),b(S.$$.fragment,e),b(P.$$.fragment,e),b(de.$$.fragment,e),b(Q.$$.fragment,e),b(ce.$$.fragment,e),Xe=!1},d(e){e&&(r(l),r(n),r(a),r(i),r(T),r(z),r(xe),r(Ce),r(Y),r(Je),r(O),r(Ue),r(Fe),r(D),r(ze),r(je),r(Ge),r(J),r(Ie),r(Ze),r(k),r(qe),r(Re),r($),r(We),r(Ne),r(ke)),r(o),M(A,e),M(X,e),M(V,e),M(K,e),M(ee),M(E),M(oe,e),M(ne),M(te),M(H),M(B),M(se),M(ae,e),M(re),M(ie),M(S),M(P),M(de),M(Q),M(ce,e)}}}const Yo='{"title":"Moonshine","local":"moonshine","sections":[{"title":"MoonshineConfig","local":"transformers.MoonshineConfig","sections":[],"depth":2},{"title":"MoonshineModel","local":"transformers.MoonshineModel","sections":[],"depth":2},{"title":"MoonshineForConditionalGeneration","local":"transformers.MoonshineForConditionalGeneration","sections":[],"depth":2}],"depth":1}';function Oo(w){return jo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class rn extends Go{constructor(o){super(),Io(this,o,Oo,Ao,zo,{})}}export{rn as component};
