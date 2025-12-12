import{s as ud,o as fd,n as C}from"../chunks/scheduler.18a86fab.js";import{S as gd,i as _d,g as p,s,r as u,A as bd,h as m,f as a,c as r,j as V,x as h,u as f,k,y as i,a as c,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as L}from"../chunks/Tip.77304350.js";import{D as x}from"../chunks/Docstring.a1ef7999.js";import{C as j}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as J}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{P as yd}from"../chunks/PipelineTag.7749150e.js";import{H as $,E as Td}from"../chunks/getInferenceSnippets.06c2775f.js";function Md(w){let t,T="Examples:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERhdGEyVmVjVGV4dENvbmZpZyUyQyUyMERhdGEyVmVjVGV4dE1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMERhdGEyVmVjVGV4dCUyMGZhY2Vib29rJTJGZGF0YTJ2ZWMtdGV4dC1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMERhdGEyVmVjVGV4dENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBmYWNlYm9vayUyRmRhdGEydmVjLXRleHQtYmFzZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwRGF0YTJWZWNUZXh0TW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Data2VecTextConfig, Data2VecTextModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Data2VecText facebook/data2vec-text-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Data2VecTextConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the facebook/data2vec-text-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function vd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERhdGEyVmVjQXVkaW9Db25maWclMkMlMjBEYXRhMlZlY0F1ZGlvTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwRGF0YTJWZWNBdWRpbyUyMGZhY2Vib29rJTJGZGF0YTJ2ZWMtYXVkaW8tYmFzZS05NjBoJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMERhdGEyVmVjQXVkaW9Db25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwZmFjZWJvb2slMkZkYXRhMnZlYy1hdWRpby1iYXNlLTk2MGglMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjQXVkaW9Nb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Data2VecAudioConfig, Data2VecAudioModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Data2VecAudio facebook/data2vec-audio-base-960h style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Data2VecAudioConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the facebook/data2vec-audio-base-960h style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecAudioModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function wd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERhdGEyVmVjVmlzaW9uQ29uZmlnJTJDJTIwRGF0YTJWZWNWaXNpb25Nb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBEYXRhMlZlY1Zpc2lvbiUyMGRhdGEydmVjX3Zpc2lvbi1iYXNlLXBhdGNoMTYtMjI0LWluMjJrJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMERhdGEyVmVjVmlzaW9uQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGRhdGEydmVjX3Zpc2lvbi1iYXNlLXBhdGNoMTYtMjI0LWluMjJrJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY1Zpc2lvbk1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Data2VecVisionConfig, Data2VecVisionModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Data2VecVision data2vec_vision-base-patch16-224-in22k style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Data2VecVisionConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the data2vec_vision-base-patch16-224-in22k style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecVisionModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function kd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Vd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function xd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9GZWF0dXJlRXh0cmFjdG9yJTJDJTIwRGF0YTJWZWNBdWRpb0ZvckF1ZGlvRnJhbWVDbGFzc2lmaWNhdGlvbiUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQWltcG9ydCUyMHRvcmNoJTBBJTBBZGF0YXNldCUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGbGlicmlzcGVlY2hfYXNyX2RlbW8lMjIlMkMlMjAlMjJjbGVhbiUyMiUyQyUyMHNwbGl0JTNEJTIydmFsaWRhdGlvbiUyMiklMEFkYXRhc2V0JTIwJTNEJTIwZGF0YXNldC5zb3J0KCUyMmlkJTIyKSUwQXNhbXBsaW5nX3JhdGUlMjAlM0QlMjBkYXRhc2V0LmZlYXR1cmVzJTVCJTIyYXVkaW8lMjIlNUQuc2FtcGxpbmdfcmF0ZSUwQSUwQWZlYXR1cmVfZXh0cmFjdG9yJTIwJTNEJTIwQXV0b0ZlYXR1cmVFeHRyYWN0b3IuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZGF0YTJ2ZWMtYXVkaW8tYmFzZS05NjBoJTIyKSUwQW1vZGVsJTIwJTNEJTIwRGF0YTJWZWNBdWRpb0ZvckF1ZGlvRnJhbWVDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy1hdWRpby1iYXNlLTk2MGglMjIpJTBBJTBBJTIzJTIwYXVkaW8lMjBmaWxlJTIwaXMlMjBkZWNvZGVkJTIwb24lMjB0aGUlMjBmbHklMEFpbnB1dHMlMjAlM0QlMjBmZWF0dXJlX2V4dHJhY3RvcihkYXRhc2V0JTVCMCU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwc2FtcGxpbmdfcmF0ZSUzRHNhbXBsaW5nX3JhdGUpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcm9iYWJpbGl0aWVzJTIwJTNEJTIwdG9yY2guc2lnbW9pZChsb2dpdHMlNUIwJTVEKSUwQSUyMyUyMGxhYmVscyUyMGlzJTIwYSUyMG9uZS1ob3QlMjBhcnJheSUyMG9mJTIwc2hhcGUlMjAobnVtX2ZyYW1lcyUyQyUyMG51bV9zcGVha2VycyklMEFsYWJlbHMlMjAlM0QlMjAocHJvYmFiaWxpdGllcyUyMCUzRSUyMDAuNSkubG9uZygpJTBBbGFiZWxzJTVCMCU1RC50b2xpc3QoKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoFeatureExtractor, Data2VecAudioForAudioFrameClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = AutoFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecAudioForAudioFrameClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, sampling_rate=sampling_rate)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>probabilities = torch.sigmoid(logits[<span class="hljs-number">0</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># labels is a one-hot array of shape (num_frames, num_speakers)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = (probabilities &gt; <span class="hljs-number">0.5</span>).long()
<span class="hljs-meta">&gt;&gt;&gt; </span>labels[<span class="hljs-number">0</span>].tolist()
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Cd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function $d(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBEYXRhMlZlY0F1ZGlvRm9yQ1RDJTBBZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTBBaW1wb3J0JTIwdG9yY2glMEElMEFkYXRhc2V0JTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZsaWJyaXNwZWVjaF9hc3JfZGVtbyUyMiUyQyUyMCUyMmNsZWFuJTIyJTJDJTIwc3BsaXQlM0QlMjJ2YWxpZGF0aW9uJTIyKSUwQWRhdGFzZXQlMjAlM0QlMjBkYXRhc2V0LnNvcnQoJTIyaWQlMjIpJTBBc2FtcGxpbmdfcmF0ZSUyMCUzRCUyMGRhdGFzZXQuZmVhdHVyZXMlNUIlMjJhdWRpbyUyMiU1RC5zYW1wbGluZ19yYXRlJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy1hdWRpby1iYXNlLTk2MGglMjIpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY0F1ZGlvRm9yQ1RDLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLWF1ZGlvLWJhc2UtOTYwaCUyMiklMEElMEElMjMlMjBhdWRpbyUyMGZpbGUlMjBpcyUyMGRlY29kZWQlMjBvbiUyMHRoZSUyMGZseSUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3NvcihkYXRhc2V0JTVCMCU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlMkMlMjBzYW1wbGluZ19yYXRlJTNEc2FtcGxpbmdfcmF0ZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEFwcmVkaWN0ZWRfaWRzJTIwJTNEJTIwdG9yY2guYXJnbWF4KGxvZ2l0cyUyQyUyMGRpbSUzRC0xKSUwQSUwQSUyMyUyMHRyYW5zY3JpYmUlMjBzcGVlY2glMEF0cmFuc2NyaXB0aW9uJTIwJTNEJTIwcHJvY2Vzc29yLmJhdGNoX2RlY29kZShwcmVkaWN0ZWRfaWRzKSUwQXRyYW5zY3JpcHRpb24lNUIwJTVEJTBBJTBBaW5wdXRzJTVCJTIybGFiZWxzJTIyJTVEJTIwJTNEJTIwcHJvY2Vzc29yKHRleHQlM0RkYXRhc2V0JTVCMCU1RCU1QiUyMnRleHQlMjIlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMEElMEElMjMlMjBjb21wdXRlJTIwbG9zcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, Data2VecAudioForCTC
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecAudioForCTC.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>], sampling_rate=sampling_rate, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_ids = torch.argmax(logits, dim=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># transcribe speech</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription = processor.batch_decode(predicted_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>transcription[<span class="hljs-number">0</span>]
...

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs[<span class="hljs-string">&quot;labels&quot;</span>] = processor(text=dataset[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;text&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># compute loss</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function jd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Jd(w){let t,T="Example of single-label classification:",l,d,M;return d=new j({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMERhdGEyVmVjQXVkaW9Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy1hdWRpby1iYXNlLTk2MGglMjIpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY0F1ZGlvRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy1hdWRpby1iYXNlLTk2MGglMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjQXVkaW9Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLWF1ZGlvLWJhc2UtOTYwaCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecAudioForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecAudioForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecAudioForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Fd(w){let t,T="Example of multi-label classification:",l,d,M;return d=new j({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMERhdGEyVmVjQXVkaW9Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy1hdWRpby1iYXNlLTk2MGglMjIpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY0F1ZGlvRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy1hdWRpby1iYXNlLTk2MGglMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjQXVkaW9Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJmYWNlYm9vayUyRmRhdGEydmVjLWF1ZGlvLWJhc2UtOTYwaCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecAudioForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecAudioForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecAudioForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Ud(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Dd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9GZWF0dXJlRXh0cmFjdG9yJTJDJTIwRGF0YTJWZWNBdWRpb0ZvclhWZWN0b3IlMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQWRhdGFzZXQlMjAlM0QlMjBsb2FkX2RhdGFzZXQoJTIyaGYtaW50ZXJuYWwtdGVzdGluZyUyRmxpYnJpc3BlZWNoX2Fzcl9kZW1vJTIyJTJDJTIwJTIyY2xlYW4lMjIlMkMlMjBzcGxpdCUzRCUyMnZhbGlkYXRpb24lMjIpJTBBZGF0YXNldCUyMCUzRCUyMGRhdGFzZXQuc29ydCglMjJpZCUyMiklMEFzYW1wbGluZ19yYXRlJTIwJTNEJTIwZGF0YXNldC5mZWF0dXJlcyU1QiUyMmF1ZGlvJTIyJTVELnNhbXBsaW5nX3JhdGUlMEElMEFmZWF0dXJlX2V4dHJhY3RvciUyMCUzRCUyMEF1dG9GZWF0dXJlRXh0cmFjdG9yLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLWF1ZGlvLWJhc2UtOTYwaCUyMiklMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjQXVkaW9Gb3JYVmVjdG9yLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLWF1ZGlvLWJhc2UtOTYwaCUyMiklMEElMEElMjMlMjBhdWRpbyUyMGZpbGUlMjBpcyUyMGRlY29kZWQlMjBvbiUyMHRoZSUyMGZseSUwQWlucHV0cyUyMCUzRCUyMGZlYXR1cmVfZXh0cmFjdG9yKCUwQSUyMCUyMCUyMCUyMCU1QmQlNUIlMjJhcnJheSUyMiU1RCUyMGZvciUyMGQlMjBpbiUyMGRhdGFzZXQlNUIlM0EyJTVEJTVCJTIyYXVkaW8lMjIlNUQlNUQlMkMlMjBzYW1wbGluZ19yYXRlJTNEc2FtcGxpbmdfcmF0ZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSUwQSklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwZW1iZWRkaW5ncyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5lbWJlZGRpbmdzJTBBJTBBZW1iZWRkaW5ncyUyMCUzRCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwubm9ybWFsaXplKGVtYmVkZGluZ3MlMkMlMjBkaW0lM0QtMSkuY3B1KCklMEElMEElMjMlMjB0aGUlMjByZXN1bHRpbmclMjBlbWJlZGRpbmdzJTIwY2FuJTIwYmUlMjB1c2VkJTIwZm9yJTIwY29zaW5lJTIwc2ltaWxhcml0eS1iYXNlZCUyMHJldHJpZXZhbCUwQWNvc2luZV9zaW0lMjAlM0QlMjB0b3JjaC5ubi5Db3NpbmVTaW1pbGFyaXR5KGRpbSUzRC0xKSUwQXNpbWlsYXJpdHklMjAlM0QlMjBjb3NpbmVfc2ltKGVtYmVkZGluZ3MlNUIwJTVEJTJDJTIwZW1iZWRkaW5ncyU1QjElNUQpJTBBdGhyZXNob2xkJTIwJTNEJTIwMC43JTIwJTIwJTIzJTIwdGhlJTIwb3B0aW1hbCUyMHRocmVzaG9sZCUyMGlzJTIwZGF0YXNldC1kZXBlbmRlbnQlMEFpZiUyMHNpbWlsYXJpdHklMjAlM0MlMjB0aHJlc2hvbGQlM0ElMEElMjAlMjAlMjAlMjBwcmludCglMjJTcGVha2VycyUyMGFyZSUyMG5vdCUyMHRoZSUyMHNhbWUhJTIyKSUwQXJvdW5kKHNpbWlsYXJpdHkuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoFeatureExtractor, Data2VecAudioForXVector
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_demo&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = dataset.sort(<span class="hljs-string">&quot;id&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = dataset.features[<span class="hljs-string">&quot;audio&quot;</span>].sampling_rate

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = AutoFeatureExtractor.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecAudioForXVector.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-audio-base-960h&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># audio file is decoded on the fly</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = feature_extractor(
<span class="hljs-meta">... </span>    [d[<span class="hljs-string">&quot;array&quot;</span>] <span class="hljs-keyword">for</span> d <span class="hljs-keyword">in</span> dataset[:<span class="hljs-number">2</span>][<span class="hljs-string">&quot;audio&quot;</span>]], sampling_rate=sampling_rate, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    embeddings = model(**inputs).embeddings

<span class="hljs-meta">&gt;&gt;&gt; </span>embeddings = torch.nn.functional.normalize(embeddings, dim=-<span class="hljs-number">1</span>).cpu()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the resulting embeddings can be used for cosine similarity-based retrieval</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>cosine_sim = torch.nn.CosineSimilarity(dim=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>similarity = cosine_sim(embeddings[<span class="hljs-number">0</span>], embeddings[<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>threshold = <span class="hljs-number">0.7</span>  <span class="hljs-comment"># the optimal threshold is dataset-dependent</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">if</span> similarity &lt; threshold:
<span class="hljs-meta">... </span>    <span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Speakers are not the same!&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(similarity.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Wd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Zd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function zd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEYXRhMlZlY1RleHRGb3JDYXVzYWxMTSUyQyUyMERhdGEyVmVjVGV4dENvbmZpZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy10ZXh0LWJhc2UlMjIpJTBBY29uZmlnJTIwJTNEJTIwRGF0YTJWZWNUZXh0Q29uZmlnLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLXRleHQtYmFzZSUyMiklMEFjb25maWcuaXNfZGVjb2RlciUyMCUzRCUyMFRydWUlMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjVGV4dEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLXRleHQtYmFzZSUyMiUyQyUyMGNvbmZpZyUzRGNvbmZpZyklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQXByZWRpY3Rpb25fbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecTextForCausalLM, Data2VecTextConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = Data2VecTextConfig.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Gd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Id(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEYXRhMlZlY1RleHRGb3JNYXNrZWRMTSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy10ZXh0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY1RleHRGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy10ZXh0LWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecTextForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForMaskedLM.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of &lt;mask&gt;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
...

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-&lt;mask&gt; tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(outputs.loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Rd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Bd(w){let t,T="Example of single-label classification:",l,d,M;return d=new j({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMERhdGEyVmVjVGV4dEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLXRleHQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjVGV4dEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZGF0YTJ2ZWMtdGV4dC1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY1RleHRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLXRleHQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecTextForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Nd(w){let t,T="Example of multi-label classification:",l,d,M;return d=new j({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMERhdGEyVmVjVGV4dEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLXRleHQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjVGV4dEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZGF0YTJ2ZWMtdGV4dC1iYXNlJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY1RleHRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJmYWNlYm9vayUyRmRhdGEydmVjLXRleHQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecTextForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function qd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Xd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEYXRhMlZlY1RleHRGb3JNdWx0aXBsZUNob2ljZSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy10ZXh0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY1RleHRGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy10ZXh0LWJhc2UlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecTextForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Ad(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Yd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEYXRhMlZlY1RleHRGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLXRleHQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjVGV4dEZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZGF0YTJ2ZWMtdGV4dC1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecTextForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForTokenClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;HuggingFace is a company based in Paris and New York&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_class_ids = logits.argmax(-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Note that tokens are classified rather then input words which means that</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># there might be more predicted token classes than words.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Multiple token classes might account for the same word</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_tokens_classes = [model.config.id2label[t.item()] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> predicted_token_class_ids[<span class="hljs-number">0</span>]]
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_tokens_classes
...

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Ld(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Sd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEYXRhMlZlY1RleHRGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy10ZXh0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY1RleHRGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy10ZXh0LWJhc2UlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Data2VecTextForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecTextForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-text-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predict_answer_tokens, skip_special_tokens=<span class="hljs-literal">True</span>)
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># target is &quot;nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index = torch.tensor([<span class="hljs-number">14</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>target_end_index = torch.tensor([<span class="hljs-number">15</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Hd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Ed(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Qd(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Pd(w){let t,T="Example:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMERhdGEyVmVjVmlzaW9uRm9ySW1hZ2VDbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTBBJTBBZGF0YXNldCUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJodWdnaW5nZmFjZSUyRmNhdHMtaW1hZ2UlMjIpJTBBaW1hZ2UlMjAlM0QlMjBkYXRhc2V0JTVCJTIydGVzdCUyMiU1RCU1QiUyMmltYWdlJTIyJTVEJTVCMCU1RCUwQSUwQWltYWdlX3Byb2Nlc3NvciUyMCUzRCUyMEF1dG9JbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy12aXNpb24tYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMERhdGEyVmVjVmlzaW9uRm9ySW1hZ2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZkYXRhMnZlYy12aXNpb24tYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IoaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwbW9kZWwlMjBwcmVkaWN0cyUyMG9uZSUyMG9mJTIwdGhlJTIwMTAwMCUyMEltYWdlTmV0JTIwY2xhc3NlcyUwQXByZWRpY3RlZF9sYWJlbCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpLml0ZW0oKSUwQXByaW50KG1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9sYWJlbCU1RCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, Data2VecVisionForImageClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;huggingface/cats-image&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>image = dataset[<span class="hljs-string">&quot;test&quot;</span>][<span class="hljs-string">&quot;image&quot;</span>][<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-vision-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecVisionForImageClassification.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-vision-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># model predicts one of the 1000 ImageNet classes</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_label = logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(model.config.id2label[predicted_label])
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function Od(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(l,d){c(l,t,d)},p:C,d(l){l&&a(t)}}}function Kd(w){let t,T="Examples:",l,d,M;return d=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMERhdGEyVmVjVmlzaW9uRm9yU2VtYW50aWNTZWdtZW50YXRpb24lMEFmcm9tJTIwUElMJTIwaW1wb3J0JTIwSW1hZ2UlMEFpbXBvcnQlMjByZXF1ZXN0cyUwQSUwQXVybCUyMCUzRCUyMCUyMmh0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEElMEFpbWFnZV9wcm9jZXNzb3IlMjAlM0QlMjBBdXRvSW1hZ2VQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZGF0YTJ2ZWMtdmlzaW9uLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBEYXRhMlZlY1Zpc2lvbkZvclNlbWFudGljU2VnbWVudGF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLXZpc2lvbi1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMGltYWdlX3Byb2Nlc3NvcihpbWFnZXMlM0RpbWFnZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUyMyUyMGxvZ2l0cyUyMGFyZSUyMG9mJTIwc2hhcGUlMjAoYmF0Y2hfc2l6ZSUyQyUyMG51bV9sYWJlbHMlMkMlMjBoZWlnaHQlMkMlMjB3aWR0aCklMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, Data2VecVisionForSemanticSegmentation
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-vision-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Data2VecVisionForSemanticSegmentation.from_pretrained(<span class="hljs-string">&quot;facebook/data2vec-vision-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># logits are of shape (batch_size, num_labels, height, width)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,l=s(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=T),l=r(o),f(d.$$.fragment,o)},m(o,v){c(o,t,v),c(o,l,v),g(d,o,v),M=!0},p:C,i(o){M||(_(d.$$.fragment,o),M=!0)},o(o){b(d.$$.fragment,o),M=!1},d(o){o&&(a(t),a(l)),y(d,o)}}}function ec(w){let t,T,l,d,M,o="<em>This model was released on 2022-02-07 and added to Hugging Face Transformers on 2022-03-01.</em>",v,kt,ha,Ae,Wi='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',ua,Vt,fa,xt,Zi=`The Data2Vec model was proposed in <a href="https://huggingface.co/papers/2202.03555" rel="nofollow">data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language</a> by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu and Michael Auli.
Data2Vec proposes a unified framework for self-supervised learning across different data modalities - text, audio and images.
Importantly, predicted targets for pre-training are contextualized latent representations of the inputs, rather than modality-specific, context-independent targets.`,ga,Ct,zi="The abstract from the paper is the following:",_a,$t,Gi=`<em>While the general idea of self-supervised learning is identical across modalities, the actual algorithms and
objectives differ widely because they were developed with a single modality in mind. To get us closer to general
self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech,
NLP or computer vision. The core idea is to predict latent representations of the full input data based on a
masked view of the input in a selfdistillation setup using a standard Transformer architecture.
Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which
are local in nature, data2vec predicts contextualized latent representations that contain information from
the entire input. Experiments on the major benchmarks of speech recognition, image classification, and
natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.
Models and code are available at <a href="http://www.github.com/pytorch/fairseq/tree/master/examples/data2vec" rel="nofollow">www.github.com/pytorch/fairseq/tree/master/examples/data2vec</a>.</em>`,ba,jt,Ii='This model was contributed by <a href="https://huggingface.co/edugp" rel="nofollow">edugp</a> and <a href="https://huggingface.co/patrickvonplaten" rel="nofollow">patrickvonplaten</a>.',ya,Jt,Ri=`The original code (for NLP and Speech) can be found <a href="https://github.com/pytorch/fairseq/tree/main/examples/data2vec" rel="nofollow">here</a>.
The original code for vision can be found <a href="https://github.com/facebookresearch/data2vec_vision/tree/main/beit" rel="nofollow">here</a>.`,Ta,Ft,Ma,Ut,Bi='<li>Data2VecAudio, Data2VecText, and Data2VecVision have all been trained using the same self-supervised learning method.</li> <li>For Data2VecAudio, preprocessing is identical to <a href="/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model">Wav2Vec2Model</a>, including feature extraction</li> <li>For Data2VecText, preprocessing is identical to <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel">RobertaModel</a>, including tokenization.</li> <li>For Data2VecVision, preprocessing is identical to <a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitModel">BeitModel</a>, including feature extraction.</li> <li>The <code>head_mask</code> argument is ignored when using all attention implementation other than “eager”. If you have a <code>head_mask</code> and want it to have effect, load the model with <code>XXXModel.from_pretrained(model_id, attn_implementation=&quot;eager&quot;)</code></li>',va,Dt,wa,Wt,Ni=`PyTorch includes a native scaled dot-product attention (SDPA) operator as part of <code>torch.nn.functional</code>. This function
encompasses several implementations that can be applied depending on the inputs and the hardware in use. See the
<a href="https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html" rel="nofollow">official documentation</a>
or the <a href="https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention" rel="nofollow">GPU Inference</a>
page for more information.`,ka,Zt,qi=`SDPA is used by default for <code>torch&gt;=2.1.1</code> when an implementation is available, but you may also set
<code>attn_implementation=&quot;sdpa&quot;</code> in <code>from_pretrained()</code> to explicitly request SDPA to be used.`,Va,zt,Xi="The SDPA implementation is currently available for the Data2VecAudio and Data2VecVision models.",xa,Gt,Ca,It,Ai="For the best speedups, we recommend loading the model in half-precision (e.g. <code>torch.float16</code> or <code>torch.bfloat16</code>).",$a,Rt,Yi=`For the Data2VecVision model, on a local benchmark (NVIDIA GeForce RTX 2060-8GB, PyTorch 2.5.1, OS Ubuntu 20.04)
with <code>float16</code> and <code>facebook/data2vec-vision-base</code> model, we saw the following improvements during training and
inference:`,ja,Bt,Ja,Nt,Li="<thead><tr><th>num_training_steps</th> <th>batch_size</th> <th>image_size</th> <th>is_cuda</th> <th>Time per batch (eager - s)</th> <th>Time per batch (sdpa - s)</th> <th>Speedup (%)</th> <th>Eager peak mem (MB)</th> <th>SDPA peak mem (MB)</th> <th>Mem saving (%)</th></tr></thead> <tbody><tr><td>50</td> <td>2</td> <td>(1048, 640)</td> <td>True</td> <td>0.996</td> <td>0.754</td> <td>32.147</td> <td>6722.198</td> <td>4264.653</td> <td>57.626</td></tr></tbody>",Fa,qt,Ua,Xt,Si='<thead><tr><th align="right">Image batch size</th> <th align="right">Eager (s/iter)</th> <th align="left">Eager CI, %</th> <th align="right">Eager memory (MB)</th> <th align="right">SDPA (s/iter)</th> <th align="left">SDPA CI, %</th> <th align="right">SDPA memory (MB)</th> <th align="right">SDPA speedup</th> <th align="right">SDPA memory saved</th></tr></thead> <tbody><tr><td align="right">1</td> <td align="right">0.011</td> <td align="left">±0.3%</td> <td align="right">3.76143e+08</td> <td align="right">0.01</td> <td align="left">±0.3%</td> <td align="right">3.74397e+08</td> <td align="right">1.101</td> <td align="right">0.466</td></tr> <tr><td align="right">4</td> <td align="right">0.014</td> <td align="left">±0.1%</td> <td align="right">4.02756e+08</td> <td align="right">0.012</td> <td align="left">±0.2%</td> <td align="right">3.91373e+08</td> <td align="right">1.219</td> <td align="right">2.909</td></tr> <tr><td align="right">16</td> <td align="right">0.046</td> <td align="left">±0.3%</td> <td align="right">4.96482e+08</td> <td align="right">0.035</td> <td align="left">±0.2%</td> <td align="right">4.51017e+08</td> <td align="right">1.314</td> <td align="right">10.081</td></tr> <tr><td align="right">32</td> <td align="right">0.088</td> <td align="left">±0.1%</td> <td align="right">6.23903e+08</td> <td align="right">0.067</td> <td align="left">±0.1%</td> <td align="right">5.32974e+08</td> <td align="right">1.33</td> <td align="right">17.061</td></tr></tbody>',Da,At,Wa,Yt,Hi="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Data2Vec.",Za,Lt,za,St,Ei='<li><a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionForImageClassification">Data2VecVisionForImageClassification</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification" rel="nofollow">example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb" rel="nofollow">notebook</a>.</li>',Ga,Ht,Qi="<strong>Data2VecText documentation resources</strong>",Ia,Et,Pi='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',Ra,Qt,Oi="<strong>Data2VecAudio documentation resources</strong>",Ba,Pt,Ki='<li><a href="../tasks/audio_classification">Audio classification task guide</a></li> <li><a href="../tasks/asr">Automatic speech recognition task guide</a></li>',Na,Ot,el="<strong>Data2VecVision documentation resources</strong>",qa,Kt,tl='<li><a href="../tasks/image_classification">Image classification</a></li> <li><a href="../tasks/semantic_segmentation">Semantic segmentation</a></li>',Xa,eo,ol="If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",Aa,to,Ya,E,oo,js,en,nl=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextModel">Data2VecTextModel</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextModel">Data2VecTextModel</a>. It
is used to instantiate a Data2VecText model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the Data2VecText
<a href="https://huggingface.co/facebook/data2vec-text-base" rel="nofollow">facebook/data2vec-text-base</a> architecture.`,Js,tn,al=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Fs,Ye,La,no,Sa,Q,ao,Us,on,sl=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioModel">Data2VecAudioModel</a>. It is used to instantiate
an Data2VecAudio model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Data2VecAudio
<a href="https://huggingface.co/facebook/data2vec-audio-base-960h" rel="nofollow">facebook/data2vec-audio-base-960h</a> architecture.`,Ds,nn,rl=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ws,Le,Ha,so,Ea,Ve,ro,Zs,an,il=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionModel">Data2VecVisionModel</a>. It is used to instantiate
an Data2VecVision model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Data2VecVision
<a href="https://huggingface.co/facebook/data2vec-vision-base" rel="nofollow">facebook/data2vec-vision-base</a> architecture.`,zs,Se,Qa,io,Pa,F,lo,Gs,sn,ll="The bare Data2Vec Audio Model outputting raw hidden-states without any specific head on top.",Is,rn,dl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rs,ln,cl=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bs,Re,co,Ns,dn,pl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioModel">Data2VecAudioModel</a> forward method, overrides the <code>__call__</code> special method.',qs,He,Oa,po,Ka,U,mo,Xs,cn,ml="The Data2Vec Audio Model with a frame classification head on top for tasks like Speaker Diarization.",As,pn,hl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ys,mn,ul=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ls,ue,ho,Ss,hn,fl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForAudioFrameClassification">Data2VecAudioForAudioFrameClassification</a> forward method, overrides the <code>__call__</code> special method.',Hs,Ee,Es,Qe,es,uo,ts,D,fo,Qs,un,gl="Data2VecAudio Model with a <code>language modeling</code> head on top for Connectionist Temporal Classification (CTC).",Ps,fn,_l=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Os,gn,bl=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ks,fe,go,er,_n,yl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForCTC">Data2VecAudioForCTC</a> forward method, overrides the <code>__call__</code> special method.',tr,Pe,or,Oe,os,_o,ns,W,bo,nr,bn,Tl=`Data2VecAudio Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
SUPERB Keyword Spotting.`,ar,yn,Ml=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,sr,Tn,vl=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,rr,S,yo,ir,Mn,wl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForSequenceClassification">Data2VecAudioForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',lr,Ke,dr,et,cr,tt,as,To,ss,Z,Mo,pr,vn,kl="Data2VecAudio Model with an XVector feature extraction head on top for tasks like Speaker Verification.",mr,wn,Vl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,hr,kn,xl=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ur,ge,vo,fr,Vn,Cl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForXVector">Data2VecAudioForXVector</a> forward method, overrides the <code>__call__</code> special method.',gr,ot,_r,nt,rs,wo,is,z,ko,br,xn,$l="The bare Data2Vec Text Text Model outputting raw hidden-states without any specific head on to.",yr,Cn,jl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Tr,$n,Jl=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Mr,Be,Vo,vr,jn,Fl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextModel">Data2VecTextModel</a> forward method, overrides the <code>__call__</code> special method.',wr,at,ls,xo,ds,G,Co,kr,Jn,Ul="Data2VecText Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Vr,Fn,Dl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xr,Un,Wl=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Cr,_e,$o,$r,Dn,Zl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForCausalLM">Data2VecTextForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',jr,st,Jr,rt,cs,jo,ps,I,Jo,Fr,Wn,zl="The Data2Vec Text Model with a <code>language modeling</code> head on top.”",Ur,Zn,Gl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Dr,zn,Il=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wr,be,Fo,Zr,Gn,Rl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMaskedLM">Data2VecTextForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',zr,it,Gr,lt,ms,Uo,hs,R,Do,Ir,In,Bl=`Data2VecText Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,Rr,Rn,Nl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Br,Bn,ql=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Nr,H,Wo,qr,Nn,Xl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForSequenceClassification">Data2VecTextForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Xr,dt,Ar,ct,Yr,pt,us,Zo,fs,B,zo,Lr,qn,Al=`The Data2Vec Text Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Sr,Xn,Yl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Hr,An,Ll=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Er,ye,Go,Qr,Yn,Sl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMultipleChoice">Data2VecTextForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Pr,mt,Or,ht,gs,Io,_s,N,Ro,Kr,Ln,Hl=`The Data2Vec Text transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,ei,Sn,El=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ti,Hn,Ql=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,oi,Te,Bo,ni,En,Pl='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForTokenClassification">Data2VecTextForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',ai,ut,si,ft,bs,No,ys,q,qo,ri,Qn,Ol=`The Data2Vec Text transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ii,Pn,Kl=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,li,On,ed=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,di,Me,Xo,ci,Kn,td='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForQuestionAnswering">Data2VecTextForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',pi,gt,mi,_t,Ts,Ao,Ms,X,Yo,hi,ea,od="The bare Data2Vec Vision Model outputting raw hidden-states without any specific head on top.",ui,ta,nd=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,fi,oa,ad=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,gi,ve,Lo,_i,na,sd='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionModel">Data2VecVisionModel</a> forward method, overrides the <code>__call__</code> special method.',bi,bt,yi,yt,vs,So,ws,A,Ho,Ti,aa,rd=`Data2VecVision Model transformer with an image classification head on top (a linear layer on top of the average of
the final hidden states of the patch tokens) e.g. for ImageNet.`,Mi,sa,id=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vi,ra,ld=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wi,we,Eo,ki,ia,dd='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionForImageClassification">Data2VecVisionForImageClassification</a> forward method, overrides the <code>__call__</code> special method.',Vi,Tt,xi,Mt,ks,Qo,Vs,Y,Po,Ci,la,cd="The Data2Vec Vision Model with a semantic segmentation head on top e.g. for ADE20K, CityScapes.",$i,da,pd=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ji,ca,md=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ji,ke,Oo,Fi,pa,hd='The <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionForSemanticSegmentation">Data2VecVisionForSemanticSegmentation</a> forward method, overrides the <code>__call__</code> special method.',Ui,vt,Di,wt,xs,Ko,Cs,ma,$s;return kt=new $({props:{title:"Data2Vec",local:"data2vec",headingTag:"h1"}}),Vt=new $({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ft=new $({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Dt=new $({props:{title:"Using Scaled Dot Product Attention (SDPA)",local:"using-scaled-dot-product-attention-sdpa",headingTag:"h3"}}),Gt=new j({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERhdGEyVmVjVmlzaW9uRm9ySW1hZ2VDbGFzc2lmaWNhdGlvbiUwQW1vZGVsJTIwJTNEJTIwRGF0YTJWZWNWaXNpb25Gb3JJbWFnZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRhdGEydmVjLXZpc2lvbi1iYXNlJTIyJTJDJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYpJTBBLi4u",highlighted:`from transformers import Data2VecVisionForImageClassification
model = <span class="hljs-module-access"><span class="hljs-module"><span class="hljs-identifier">Data2VecVisionForImageClassification</span>.</span></span>from<span class="hljs-constructor">_pretrained(<span class="hljs-string">&quot;facebook/data2vec-vision-base&quot;</span>, <span class="hljs-params">attn_implementation</span>=<span class="hljs-string">&quot;sdpa&quot;</span>, <span class="hljs-params">dtype</span>=<span class="hljs-params">torch</span>.<span class="hljs-params">float16</span>)</span>
...`,wrap:!1}}),Bt=new $({props:{title:"Training",local:"training",headingTag:"h4"}}),qt=new $({props:{title:"Inference",local:"inference",headingTag:"h4"}}),At=new $({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Lt=new yd({props:{pipeline:"image-classification"}}),to=new $({props:{title:"Data2VecTextConfig",local:"transformers.Data2VecTextConfig",headingTag:"h2"}}),oo=new x({props:{name:"class transformers.Data2VecTextConfig",anchor:"transformers.Data2VecTextConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Data2VecTextConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the DATA2VEC model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <code>Data2VecModel</code>.`,name:"vocab_size"},{anchor:"transformers.Data2VecTextConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.Data2VecTextConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Data2VecTextConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Data2VecTextConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.Data2VecTextConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.Data2VecTextConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.Data2VecTextConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.Data2VecTextConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.Data2VecTextConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <code>Data2VecModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.Data2VecTextConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Data2VecTextConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.Data2VecTextConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.Data2VecTextConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.Data2VecTextConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Data2VecTextConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/configuration_data2vec_text.py#L28"}}),Ye=new J({props:{anchor:"transformers.Data2VecTextConfig.example",$$slots:{default:[Md]},$$scope:{ctx:w}}}),no=new $({props:{title:"Data2VecAudioConfig",local:"transformers.Data2VecAudioConfig",headingTag:"h2"}}),ao=new x({props:{name:"class transformers.Data2VecAudioConfig",anchor:"transformers.Data2VecAudioConfig",parameters:[{name:"vocab_size",val:" = 32"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout",val:" = 0.1"},{name:"activation_dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.1"},{name:"feat_proj_dropout",val:" = 0.0"},{name:"final_dropout",val:" = 0.1"},{name:"layerdrop",val:" = 0.1"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"feat_extract_activation",val:" = 'gelu'"},{name:"conv_dim",val:" = (512, 512, 512, 512, 512, 512, 512)"},{name:"conv_stride",val:" = (5, 2, 2, 2, 2, 2, 2)"},{name:"conv_kernel",val:" = (10, 3, 3, 3, 3, 2, 2)"},{name:"conv_bias",val:" = False"},{name:"num_conv_pos_embedding_groups",val:" = 16"},{name:"conv_pos_kernel_size",val:" = 19"},{name:"num_conv_pos_embeddings",val:" = 5"},{name:"mask_time_prob",val:" = 0.05"},{name:"mask_time_length",val:" = 10"},{name:"mask_time_min_masks",val:" = 2"},{name:"mask_feature_prob",val:" = 0.0"},{name:"mask_feature_length",val:" = 10"},{name:"mask_feature_min_masks",val:" = 0"},{name:"ctc_loss_reduction",val:" = 'sum'"},{name:"ctc_zero_infinity",val:" = False"},{name:"use_weighted_layer_sum",val:" = False"},{name:"classifier_proj_size",val:" = 256"},{name:"tdnn_dim",val:" = (512, 512, 512, 512, 1500)"},{name:"tdnn_kernel",val:" = (5, 3, 3, 1, 1)"},{name:"tdnn_dilation",val:" = (1, 2, 3, 1, 1)"},{name:"xvector_output_dim",val:" = 512"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"add_adapter",val:" = False"},{name:"adapter_kernel_size",val:" = 3"},{name:"adapter_stride",val:" = 2"},{name:"num_adapter_layers",val:" = 3"},{name:"output_hidden_size",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Data2VecAudioConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Vocabulary size of the Data2VecAudio model. Defines the number of different tokens that can be represented
by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioModel">Data2VecAudioModel</a> or <code>TFData2VecAudioModel</code>. Vocabulary size
of the model. Defines the different tokens that can be represented by the <em>inputs_ids</em> passed to the
forward method of <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioModel">Data2VecAudioModel</a>.`,name:"vocab_size"},{anchor:"transformers.Data2VecAudioConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.Data2VecAudioConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Data2VecAudioConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Data2VecAudioConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.Data2VecAudioConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.Data2VecAudioConfig.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout"},{anchor:"transformers.Data2VecAudioConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.Data2VecAudioConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Data2VecAudioConfig.final_dropout",description:`<strong>final_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the final projection layer of <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForCTC">Data2VecAudioForCTC</a>.`,name:"final_dropout"},{anchor:"transformers.Data2VecAudioConfig.layerdrop",description:`<strong>layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The LayerDrop probability. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>) for more
details.`,name:"layerdrop"},{anchor:"transformers.Data2VecAudioConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Data2VecAudioConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.Data2VecAudioConfig.feat_proj_dropout",description:`<strong>feat_proj_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for output of the feature encoder.`,name:"feat_proj_dropout"},{anchor:"transformers.Data2VecAudioConfig.feat_extract_activation",description:"<strong>feat_extract_activation</strong> (<code>str, </code>optional<code>, defaults to </code>&#x201C;gelu&#x201D;<code>) -- The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string, </code>&#x201C;gelu&#x201D;<code>, </code>&#x201C;relu&#x201D;<code>, </code>&#x201C;selu&#x201D;<code>and</code>&#x201C;gelu_new&#x201D;` are supported.",name:"feat_extract_activation"},{anchor:"transformers.Data2VecAudioConfig.conv_dim",description:`<strong>conv_dim</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(512, 512, 512, 512, 512, 512, 512)</code>) &#x2014;
A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
feature encoder. The length of <em>conv_dim</em> defines the number of 1D convolutional layers.`,name:"conv_dim"},{anchor:"transformers.Data2VecAudioConfig.conv_stride",description:`<strong>conv_stride</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(5, 2, 2, 2, 2, 2, 2)</code>) &#x2014;
A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
of <em>conv_stride</em> defines the number of convolutional layers and has to match the length of <em>conv_dim</em>.`,name:"conv_stride"},{anchor:"transformers.Data2VecAudioConfig.conv_kernel",description:`<strong>conv_kernel</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(10, 3, 3, 3, 3, 3, 3)</code>) &#x2014;
A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
length of <em>conv_kernel</em> defines the number of convolutional layers and has to match the length of
<em>conv_dim</em>.`,name:"conv_kernel"},{anchor:"transformers.Data2VecAudioConfig.conv_bias",description:`<strong>conv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the 1D convolutional layers have a bias.`,name:"conv_bias"},{anchor:"transformers.Data2VecAudioConfig.num_conv_pos_embeddings",description:`<strong>num_conv_pos_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
embeddings layer.`,name:"num_conv_pos_embeddings"},{anchor:"transformers.Data2VecAudioConfig.num_conv_pos_embedding_groups",description:`<strong>num_conv_pos_embedding_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of groups of 1D convolutional positional embeddings layer.`,name:"num_conv_pos_embedding_groups"},{anchor:"transformers.Data2VecAudioConfig.mask_time_prob",description:`<strong>mask_time_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.05) &#x2014;
Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
procedure generates &#x201D;mask_time_prob<em>len(time_axis)/mask_time_length&#x201D; independent masks over the axis. If
reasoning from the probability of each feature vector to be chosen as the start of the vector span to be
masked, </em>mask_time_prob<em> should be \`prob_vector_start</em>mask_time_length\`. Note that overlap may decrease the`,name:"mask_time_prob"},{anchor:"transformers.Data2VecAudioConfig.mask_time_length",description:`<strong>mask_time_length</strong> (<code>int</code>, <em>optional</em>, defaults to 10) &#x2014;
Length of vector span along the time axis.`,name:"mask_time_length"},{anchor:"transformers.Data2VecAudioConfig.mask_time_min_masks",description:`<strong>mask_time_min_masks</strong> (<code>int</code>, <em>optional</em>, defaults to 2), &#x2014;
The minimum number of masks of length <code>mask_feature_length</code> generated along the time axis, each time step,
irrespectively of <code>mask_feature_prob</code>. Only relevant if &#x201D;mask_time_prob*len(time_axis)/mask_time_length &lt;
mask_time_min_masks&#x201D;`,name:"mask_time_min_masks"},{anchor:"transformers.Data2VecAudioConfig.mask_feature_prob",description:`<strong>mask_feature_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
masking procedure generates &#x201D;mask_feature_prob<em>len(feature_axis)/mask_time_length&#x201D; independent masks over
the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
span to be masked, </em>mask_feature_prob<em> should be \`prob_vector_start</em>mask_feature_length<code>. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if </code>apply_spec_augment is
True\`.`,name:"mask_feature_prob"},{anchor:"transformers.Data2VecAudioConfig.mask_feature_length",description:`<strong>mask_feature_length</strong> (<code>int</code>, <em>optional</em>, defaults to 10) &#x2014;
Length of vector span along the feature axis.`,name:"mask_feature_length"},{anchor:"transformers.Data2VecAudioConfig.mask_feature_min_masks",description:`<strong>mask_feature_min_masks</strong> (<code>int</code>, <em>optional</em>, defaults to 0), &#x2014;
The minimum number of masks of length <code>mask_feature_length</code> generated along the feature axis, each time
step, irrespectively of <code>mask_feature_prob</code>. Only relevant if
&#x201D;mask_feature_prob*len(feature_axis)/mask_feature_length &lt; mask_feature_min_masks&#x201D;`,name:"mask_feature_min_masks"},{anchor:"transformers.Data2VecAudioConfig.ctc_loss_reduction",description:`<strong>ctc_loss_reduction</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;sum&quot;</code>) &#x2014;
Specifies the reduction to apply to the output of <code>torch.nn.CTCLoss</code>. Only relevant when training an
instance of <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForCTC">Data2VecAudioForCTC</a>.`,name:"ctc_loss_reduction"},{anchor:"transformers.Data2VecAudioConfig.ctc_zero_infinity",description:`<strong>ctc_zero_infinity</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to zero infinite losses and the associated gradients of <code>torch.nn.CTCLoss</code>. Infinite losses mainly
occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
of <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForCTC">Data2VecAudioForCTC</a>.`,name:"ctc_zero_infinity"},{anchor:"transformers.Data2VecAudioConfig.use_weighted_layer_sum",description:`<strong>use_weighted_layer_sum</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
instance of <a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForSequenceClassification">Data2VecAudioForSequenceClassification</a>.`,name:"use_weighted_layer_sum"},{anchor:"transformers.Data2VecAudioConfig.classifier_proj_size",description:`<strong>classifier_proj_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the projection before token mean-pooling for classification.`,name:"classifier_proj_size"},{anchor:"transformers.Data2VecAudioConfig.tdnn_dim",description:`<strong>tdnn_dim</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(512, 512, 512, 512, 1500)</code>) &#x2014;
A tuple of integers defining the number of output channels of each 1D convolutional layer in the <em>TDNN</em>
module of the <em>XVector</em> model. The length of <em>tdnn_dim</em> defines the number of <em>TDNN</em> layers.`,name:"tdnn_dim"},{anchor:"transformers.Data2VecAudioConfig.tdnn_kernel",description:`<strong>tdnn_kernel</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(5, 3, 3, 1, 1)</code>) &#x2014;
A tuple of integers defining the kernel size of each 1D convolutional layer in the <em>TDNN</em> module of the
<em>XVector</em> model. The length of <em>tdnn_kernel</em> has to match the length of <em>tdnn_dim</em>.`,name:"tdnn_kernel"},{anchor:"transformers.Data2VecAudioConfig.tdnn_dilation",description:`<strong>tdnn_dilation</strong> (<code>tuple[int]</code> or <code>list[int]</code>, <em>optional</em>, defaults to <code>(1, 2, 3, 1, 1)</code>) &#x2014;
A tuple of integers defining the dilation factor of each 1D convolutional layer in <em>TDNN</em> module of the
<em>XVector</em> model. The length of <em>tdnn_dilation</em> has to match the length of <em>tdnn_dim</em>.`,name:"tdnn_dilation"},{anchor:"transformers.Data2VecAudioConfig.xvector_output_dim",description:`<strong>xvector_output_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the <em>XVector</em> embedding vectors.`,name:"xvector_output_dim"},{anchor:"transformers.Data2VecAudioConfig.add_adapter",description:`<strong>add_adapter</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether a convolutional network should be stacked on top of the Data2VecAudio Encoder. Can be very useful
for warm-starting Data2VecAudio for SpeechEncoderDecoder models.`,name:"add_adapter"},{anchor:"transformers.Data2VecAudioConfig.adapter_kernel_size",description:`<strong>adapter_kernel_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Kernel size of the convolutional layers in the adapter network. Only relevant if <code>add_adapter is True</code>.`,name:"adapter_kernel_size"},{anchor:"transformers.Data2VecAudioConfig.adapter_stride",description:`<strong>adapter_stride</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Stride of the convolutional layers in the adapter network. Only relevant if <code>add_adapter is True</code>.`,name:"adapter_stride"},{anchor:"transformers.Data2VecAudioConfig.num_adapter_layers",description:`<strong>num_adapter_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Number of convolutional layers that should be used in the adapter network. Only relevant if <code>add_adapter is True</code>.`,name:"num_adapter_layers"},{anchor:"transformers.Data2VecAudioConfig.output_hidden_size",description:`<strong>output_hidden_size</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Dimensionality of the encoder output layer. If not defined, this defaults to <em>hidden-size</em>. Only relevant
if <code>add_adapter is True</code>.`,name:"output_hidden_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/configuration_data2vec_audio.py#L26"}}),Le=new J({props:{anchor:"transformers.Data2VecAudioConfig.example",$$slots:{default:[vd]},$$scope:{ctx:w}}}),so=new $({props:{title:"Data2VecVisionConfig",local:"transformers.Data2VecVisionConfig",headingTag:"h2"}}),ro=new x({props:{name:"class transformers.Data2VecVisionConfig",anchor:"transformers.Data2VecVisionConfig",parameters:[{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.0"},{name:"attention_probs_dropout_prob",val:" = 0.0"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"image_size",val:" = 224"},{name:"patch_size",val:" = 16"},{name:"num_channels",val:" = 3"},{name:"use_mask_token",val:" = False"},{name:"use_absolute_position_embeddings",val:" = False"},{name:"use_relative_position_bias",val:" = False"},{name:"use_shared_relative_position_bias",val:" = False"},{name:"layer_scale_init_value",val:" = 0.1"},{name:"drop_path_rate",val:" = 0.1"},{name:"use_mean_pooling",val:" = True"},{name:"out_indices",val:" = [3, 5, 7, 11]"},{name:"pool_scales",val:" = [1, 2, 3, 6]"},{name:"use_auxiliary_head",val:" = True"},{name:"auxiliary_loss_weight",val:" = 0.4"},{name:"auxiliary_channels",val:" = 256"},{name:"auxiliary_num_convs",val:" = 1"},{name:"auxiliary_concat_input",val:" = False"},{name:"semantic_loss_ignore_index",val:" = 255"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Data2VecVisionConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.Data2VecVisionConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Data2VecVisionConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Data2VecVisionConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.Data2VecVisionConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.Data2VecVisionConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.Data2VecVisionConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.Data2VecVisionConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Data2VecVisionConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.Data2VecVisionConfig.image_size",description:`<strong>image_size</strong> (<code>int</code>, <em>optional</em>, defaults to 224) &#x2014;
The size (resolution) of each image.`,name:"image_size"},{anchor:"transformers.Data2VecVisionConfig.patch_size",description:`<strong>patch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
The size (resolution) of each patch.`,name:"patch_size"},{anchor:"transformers.Data2VecVisionConfig.num_channels",description:`<strong>num_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The number of input channels.`,name:"num_channels"},{anchor:"transformers.Data2VecVisionConfig.use_mask_token",description:`<strong>use_mask_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a mask token for masked image modeling.`,name:"use_mask_token"},{anchor:"transformers.Data2VecVisionConfig.use_absolute_position_embeddings",description:`<strong>use_absolute_position_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use BERT-style absolute position embeddings.`,name:"use_absolute_position_embeddings"},{anchor:"transformers.Data2VecVisionConfig.use_relative_position_bias",description:`<strong>use_relative_position_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use T5-style relative position embeddings in the self-attention layers.`,name:"use_relative_position_bias"},{anchor:"transformers.Data2VecVisionConfig.use_shared_relative_position_bias",description:`<strong>use_shared_relative_position_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use the same relative position embeddings across all self-attention layers of the Transformer.`,name:"use_shared_relative_position_bias"},{anchor:"transformers.Data2VecVisionConfig.layer_scale_init_value",description:`<strong>layer_scale_init_value</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.`,name:"layer_scale_init_value"},{anchor:"transformers.Data2VecVisionConfig.drop_path_rate",description:`<strong>drop_path_rate</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Stochastic depth rate per sample (when applied in the main path of residual layers).`,name:"drop_path_rate"},{anchor:"transformers.Data2VecVisionConfig.use_mean_pooling",description:`<strong>use_mean_pooling</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
CLS token, before applying the classification head.`,name:"use_mean_pooling"},{anchor:"transformers.Data2VecVisionConfig.out_indices",description:`<strong>out_indices</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[3, 5, 7, 11]</code>) &#x2014;
Indices of the feature maps to use for semantic segmentation.`,name:"out_indices"},{anchor:"transformers.Data2VecVisionConfig.pool_scales",description:`<strong>pool_scales</strong> (<code>tuple[int]</code>, <em>optional</em>, defaults to <code>[1, 2, 3, 6]</code>) &#x2014;
Pooling scales used in Pooling Pyramid Module applied on the last feature map.`,name:"pool_scales"},{anchor:"transformers.Data2VecVisionConfig.use_auxiliary_head",description:`<strong>use_auxiliary_head</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use an auxiliary head during training.`,name:"use_auxiliary_head"},{anchor:"transformers.Data2VecVisionConfig.auxiliary_loss_weight",description:`<strong>auxiliary_loss_weight</strong> (<code>float</code>, <em>optional</em>, defaults to 0.4) &#x2014;
Weight of the cross-entropy loss of the auxiliary head.`,name:"auxiliary_loss_weight"},{anchor:"transformers.Data2VecVisionConfig.auxiliary_channels",description:`<strong>auxiliary_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Number of channels to use in the auxiliary head.`,name:"auxiliary_channels"},{anchor:"transformers.Data2VecVisionConfig.auxiliary_num_convs",description:`<strong>auxiliary_num_convs</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Number of convolutional layers to use in the auxiliary head.`,name:"auxiliary_num_convs"},{anchor:"transformers.Data2VecVisionConfig.auxiliary_concat_input",description:`<strong>auxiliary_concat_input</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to concatenate the output of the auxiliary head with the input before the classification layer.`,name:"auxiliary_concat_input"},{anchor:"transformers.Data2VecVisionConfig.semantic_loss_ignore_index",description:`<strong>semantic_loss_ignore_index</strong> (<code>int</code>, <em>optional</em>, defaults to 255) &#x2014;
The index that is ignored by the loss function of the semantic segmentation model.`,name:"semantic_loss_ignore_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/configuration_data2vec_vision.py#L30"}}),Se=new J({props:{anchor:"transformers.Data2VecVisionConfig.example",$$slots:{default:[wd]},$$scope:{ctx:w}}}),io=new $({props:{title:"Data2VecAudioModel",local:"transformers.Data2VecAudioModel",headingTag:"h2"}}),lo=new x({props:{name:"class transformers.Data2VecAudioModel",anchor:"transformers.Data2VecAudioModel",parameters:[{name:"config",val:": Data2VecAudioConfig"}],parametersDescription:[{anchor:"transformers.Data2VecAudioModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig">Data2VecAudioConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L701"}}),co=new x({props:{name:"forward",anchor:"transformers.Data2VecAudioModel.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mask_time_indices",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecAudioModel.forward.input_values",description:`<strong>input_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <code>.flac</code> or <code>.wav</code> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor">AutoProcessor</a> should be used for padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <code>processor_class.__call__</code> for details.`,name:"input_values"},{anchor:"transformers.Data2VecAudioModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecAudioModel.forward.mask_time_indices",description:`<strong>mask_time_indices</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
masked extracted features in <em>config.proj_codevector_dim</em> space.`,name:"mask_time_indices"},{anchor:"transformers.Data2VecAudioModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecAudioModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecAudioModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L772",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput"
>transformers.modeling_outputs.Wav2Vec2BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig"
>Data2VecAudioConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>extract_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, conv_dim[-1])</code>) — Sequence of extracted feature vectors of the last convolutional layer of the model.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput"
>transformers.modeling_outputs.Wav2Vec2BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),He=new L({props:{$$slots:{default:[kd]},$$scope:{ctx:w}}}),po=new $({props:{title:"Data2VecAudioForAudioFrameClassification",local:"transformers.Data2VecAudioForAudioFrameClassification",headingTag:"h2"}}),mo=new x({props:{name:"class transformers.Data2VecAudioForAudioFrameClassification",anchor:"transformers.Data2VecAudioForAudioFrameClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecAudioForAudioFrameClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForAudioFrameClassification">Data2VecAudioForAudioFrameClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L1080"}}),ho=new x({props:{name:"forward",anchor:"transformers.Data2VecAudioForAudioFrameClassification.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecAudioForAudioFrameClassification.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <code>.flac</code> or <code>.wav</code> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor">AutoProcessor</a> should be used for padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <code>Data2VecAudioProcessor.__call__</code> for details.`,name:"input_values"},{anchor:"transformers.Data2VecAudioForAudioFrameClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecAudioForAudioFrameClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.Data2VecAudioForAudioFrameClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecAudioForAudioFrameClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecAudioForAudioFrameClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L1124",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig"
>Data2VecAudioConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided)  — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) — Classification scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ee=new L({props:{$$slots:{default:[Vd]},$$scope:{ctx:w}}}),Qe=new J({props:{anchor:"transformers.Data2VecAudioForAudioFrameClassification.forward.example",$$slots:{default:[xd]},$$scope:{ctx:w}}}),uo=new $({props:{title:"Data2VecAudioForCTC",local:"transformers.Data2VecAudioForCTC",headingTag:"h2"}}),fo=new x({props:{name:"class transformers.Data2VecAudioForCTC",anchor:"transformers.Data2VecAudioForCTC",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecAudioForCTC.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForCTC">Data2VecAudioForCTC</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L839"}}),go=new x({props:{name:"forward",anchor:"transformers.Data2VecAudioForCTC.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.Data2VecAudioForCTC.forward.input_values",description:`<strong>input_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <code>.flac</code> or <code>.wav</code> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor">AutoProcessor</a> should be used for padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <code>processor_class.__call__</code> for details.`,name:"input_values"},{anchor:"transformers.Data2VecAudioForCTC.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecAudioForCTC.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecAudioForCTC.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecAudioForCTC.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Data2VecAudioForCTC.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_length)</code>, <em>optional</em>) &#x2014;
Labels for connectionist temporal classification. Note that <code>target_length</code> has to be smaller or equal to
the sequence length of the output logits. Indices are selected in <code>[-100, 0, ..., config.vocab_size - 1]</code>.
All labels set to <code>-100</code> are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L886",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput"
>transformers.modeling_outputs.CausalLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig"
>Data2VecAudioConfig</a>) and inputs.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput"
>transformers.modeling_outputs.CausalLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Pe=new L({props:{$$slots:{default:[Cd]},$$scope:{ctx:w}}}),Oe=new J({props:{anchor:"transformers.Data2VecAudioForCTC.forward.example",$$slots:{default:[$d]},$$scope:{ctx:w}}}),_o=new $({props:{title:"Data2VecAudioForSequenceClassification",local:"transformers.Data2VecAudioForSequenceClassification",headingTag:"h2"}}),bo=new x({props:{name:"class transformers.Data2VecAudioForSequenceClassification",anchor:"transformers.Data2VecAudioForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecAudioForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForSequenceClassification">Data2VecAudioForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L964"}}),yo=new x({props:{name:"forward",anchor:"transformers.Data2VecAudioForSequenceClassification.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.Data2VecAudioForSequenceClassification.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <code>.flac</code> or <code>.wav</code> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor">AutoProcessor</a> should be used for padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <code>Data2VecAudioProcessor.__call__</code> for details.`,name:"input_values"},{anchor:"transformers.Data2VecAudioForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecAudioForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecAudioForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecAudioForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Data2VecAudioForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L1009",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig"
>Data2VecAudioConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ke=new L({props:{$$slots:{default:[jd]},$$scope:{ctx:w}}}),et=new J({props:{anchor:"transformers.Data2VecAudioForSequenceClassification.forward.example",$$slots:{default:[Jd]},$$scope:{ctx:w}}}),tt=new J({props:{anchor:"transformers.Data2VecAudioForSequenceClassification.forward.example-2",$$slots:{default:[Fd]},$$scope:{ctx:w}}}),To=new $({props:{title:"Data2VecAudioForXVector",local:"transformers.Data2VecAudioForXVector",headingTag:"h2"}}),Mo=new x({props:{name:"class transformers.Data2VecAudioForXVector",anchor:"transformers.Data2VecAudioForXVector",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecAudioForXVector.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioForXVector">Data2VecAudioForXVector</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L1245"}}),vo=new x({props:{name:"forward",anchor:"transformers.Data2VecAudioForXVector.forward",parameters:[{name:"input_values",val:": typing.Optional[torch.Tensor]"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.Data2VecAudioForXVector.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <code>.flac</code> or <code>.wav</code> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor">AutoProcessor</a> should be used for padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <code>Data2VecAudioProcessor.__call__</code> for details.`,name:"input_values"},{anchor:"transformers.Data2VecAudioForXVector.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecAudioForXVector.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecAudioForXVector.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecAudioForXVector.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Data2VecAudioForXVector.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_audio.py#L1307",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput"
>transformers.modeling_outputs.XVectorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecAudioConfig"
>Data2VecAudioConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.xvector_output_dim)</code>) — Classification hidden states before AMSoftmax.</p>
</li>
<li>
<p><strong>embeddings</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.xvector_output_dim)</code>) — Utterance embeddings used for vector similarity-based retrieval.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput"
>transformers.modeling_outputs.XVectorOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ot=new L({props:{$$slots:{default:[Ud]},$$scope:{ctx:w}}}),nt=new J({props:{anchor:"transformers.Data2VecAudioForXVector.forward.example",$$slots:{default:[Dd]},$$scope:{ctx:w}}}),wo=new $({props:{title:"Data2VecTextModel",local:"transformers.Data2VecTextModel",headingTag:"h2"}}),ko=new x({props:{name:"class transformers.Data2VecTextModel",anchor:"transformers.Data2VecTextModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.Data2VecTextModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextModel">Data2VecTextModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.Data2VecTextModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L588"}}),Vo=new x({props:{name:"forward",anchor:"transformers.Data2VecTextModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.Data2VecTextModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Data2VecTextModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecTextModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Data2VecTextModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Data2VecTextModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecTextModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Data2VecTextModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.Data2VecTextModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.Data2VecTextModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Data2VecTextModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Data2VecTextModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecTextModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecTextModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Data2VecTextModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L634",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig"
>Data2VecTextConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, hidden_size)</code>) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
the classification token after processing through a linear layer and a tanh activation function. The linear
layer weights are trained from the next sentence prediction (classification) objective during pretraining.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),at=new L({props:{$$slots:{default:[Wd]},$$scope:{ctx:w}}}),xo=new $({props:{title:"Data2VecTextForCausalLM",local:"transformers.Data2VecTextForCausalLM",headingTag:"h2"}}),Co=new x({props:{name:"class transformers.Data2VecTextForCausalLM",anchor:"transformers.Data2VecTextForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecTextForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForCausalLM">Data2VecTextForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L758"}}),$o=new x({props:{name:"forward",anchor:"transformers.Data2VecTextForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Data2VecTextForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Data2VecTextForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecTextForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Data2VecTextForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Data2VecTextForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecTextForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Data2VecTextForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.Data2VecTextForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.Data2VecTextForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.Data2VecTextForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Data2VecTextForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Data2VecTextForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecTextForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecTextForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Data2VecTextForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L779",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig"
>Data2VecTextConfig</a>) and inputs.</p>
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
`}}),st=new L({props:{$$slots:{default:[Zd]},$$scope:{ctx:w}}}),rt=new J({props:{anchor:"transformers.Data2VecTextForCausalLM.forward.example",$$slots:{default:[zd]},$$scope:{ctx:w}}}),jo=new $({props:{title:"Data2VecTextForMaskedLM",local:"transformers.Data2VecTextForMaskedLM",headingTag:"h2"}}),Jo=new x({props:{name:"class transformers.Data2VecTextForMaskedLM",anchor:"transformers.Data2VecTextForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecTextForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMaskedLM">Data2VecTextForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L869"}}),Fo=new x({props:{name:"forward",anchor:"transformers.Data2VecTextForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecTextForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecTextForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L893",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig"
>Data2VecTextConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Masked language modeling (MLM) loss.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),it=new L({props:{$$slots:{default:[Gd]},$$scope:{ctx:w}}}),lt=new J({props:{anchor:"transformers.Data2VecTextForMaskedLM.forward.example",$$slots:{default:[Id]},$$scope:{ctx:w}}}),Uo=new $({props:{title:"Data2VecTextForSequenceClassification",local:"transformers.Data2VecTextForSequenceClassification",headingTag:"h2"}}),Do=new x({props:{name:"class transformers.Data2VecTextForSequenceClassification",anchor:"transformers.Data2VecTextForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecTextForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForSequenceClassification">Data2VecTextForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L990"}}),Wo=new x({props:{name:"forward",anchor:"transformers.Data2VecTextForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecTextForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecTextForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L1002",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig"
>Data2VecTextConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),dt=new L({props:{$$slots:{default:[Rd]},$$scope:{ctx:w}}}),ct=new J({props:{anchor:"transformers.Data2VecTextForSequenceClassification.forward.example",$$slots:{default:[Bd]},$$scope:{ctx:w}}}),pt=new J({props:{anchor:"transformers.Data2VecTextForSequenceClassification.forward.example-2",$$slots:{default:[Nd]},$$scope:{ctx:w}}}),Zo=new $({props:{title:"Data2VecTextForMultipleChoice",local:"transformers.Data2VecTextForMultipleChoice",headingTag:"h2"}}),zo=new x({props:{name:"class transformers.Data2VecTextForMultipleChoice",anchor:"transformers.Data2VecTextForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecTextForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForMultipleChoice">Data2VecTextForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L1076"}}),Go=new x({props:{name:"forward",anchor:"transformers.Data2VecTextForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecTextForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecTextForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L1087",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig"
>Data2VecTextConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) — <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),mt=new L({props:{$$slots:{default:[qd]},$$scope:{ctx:w}}}),ht=new J({props:{anchor:"transformers.Data2VecTextForMultipleChoice.forward.example",$$slots:{default:[Xd]},$$scope:{ctx:w}}}),Io=new $({props:{title:"Data2VecTextForTokenClassification",local:"transformers.Data2VecTextForTokenClassification",headingTag:"h2"}}),Ro=new x({props:{name:"class transformers.Data2VecTextForTokenClassification",anchor:"transformers.Data2VecTextForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecTextForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForTokenClassification">Data2VecTextForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L1181"}}),Bo=new x({props:{name:"forward",anchor:"transformers.Data2VecTextForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecTextForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecTextForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L1196",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig"
>Data2VecTextConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided)  — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) — Classification scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ut=new L({props:{$$slots:{default:[Ad]},$$scope:{ctx:w}}}),ft=new J({props:{anchor:"transformers.Data2VecTextForTokenClassification.forward.example",$$slots:{default:[Yd]},$$scope:{ctx:w}}}),No=new $({props:{title:"Data2VecTextForQuestionAnswering",local:"transformers.Data2VecTextForQuestionAnswering",headingTag:"h2"}}),qo=new x({props:{name:"class transformers.Data2VecTextForQuestionAnswering",anchor:"transformers.Data2VecTextForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Data2VecTextForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextForQuestionAnswering">Data2VecTextForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L1276"}}),Xo=new x({props:{name:"forward",anchor:"transformers.Data2VecTextForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_text.py#L1287",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecTextConfig"
>Data2VecTextConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.</p>
</li>
<li>
<p><strong>start_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) — Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) — Span-end scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),gt=new L({props:{$$slots:{default:[Ld]},$$scope:{ctx:w}}}),_t=new J({props:{anchor:"transformers.Data2VecTextForQuestionAnswering.forward.example",$$slots:{default:[Sd]},$$scope:{ctx:w}}}),Ao=new $({props:{title:"Data2VecVisionModel",local:"transformers.Data2VecVisionModel",headingTag:"h2"}}),Yo=new x({props:{name:"class transformers.Data2VecVisionModel",anchor:"transformers.Data2VecVisionModel",parameters:[{name:"config",val:": Data2VecVisionConfig"},{name:"add_pooling_layer",val:": bool = False"}],parametersDescription:[{anchor:"transformers.Data2VecVisionModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig">Data2VecVisionConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.Data2VecVisionModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_vision.py#L778"}}),Lo=new x({props:{name:"forward",anchor:"transformers.Data2VecVisionModel.forward",parameters:[{name:"pixel_values",val:": Tensor"},{name:"bool_masked_pos",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"interpolate_pos_encoding",val:": bool = False"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecVisionModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessor">BeitImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitFeatureExtractor.__call__">BeitImageProcessor.<strong>call</strong>()</a> for details (<code>processor_class</code> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessor">BeitImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.Data2VecVisionModel.forward.bool_masked_pos",description:`<strong>bool_masked_pos</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, num_patches)</code>, <em>optional</em>) &#x2014;
Boolean masked positions. Indicates which patches are masked (1) and which aren&#x2019;t (0).`,name:"bool_masked_pos"},{anchor:"transformers.Data2VecVisionModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecVisionModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecVisionModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecVisionModel.forward.interpolate_pos_encoding",description:`<strong>interpolate_pos_encoding</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to interpolate the pre-trained position encodings.`,name:"interpolate_pos_encoding"},{anchor:"transformers.Data2VecVisionModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_vision.py#L809",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.data2vec.modeling_data2vec_vision.Data2VecVisionModelOutputWithPooling</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig"
>Data2VecVisionConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, hidden_size)</code>) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a
Linear layer and a Tanh activation function.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.data2vec.modeling_data2vec_vision.Data2VecVisionModelOutputWithPooling</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),bt=new L({props:{$$slots:{default:[Hd]},$$scope:{ctx:w}}}),yt=new J({props:{anchor:"transformers.Data2VecVisionModel.forward.example",$$slots:{default:[Ed]},$$scope:{ctx:w}}}),So=new $({props:{title:"Data2VecVisionForImageClassification",local:"transformers.Data2VecVisionForImageClassification",headingTag:"h2"}}),Ho=new x({props:{name:"class transformers.Data2VecVisionForImageClassification",anchor:"transformers.Data2VecVisionForImageClassification",parameters:[{name:"config",val:": Data2VecVisionConfig"}],parametersDescription:[{anchor:"transformers.Data2VecVisionForImageClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig">Data2VecVisionConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_vision.py#L892"}}),Eo=new x({props:{name:"forward",anchor:"transformers.Data2VecVisionForImageClassification.forward",parameters:[{name:"pixel_values",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"interpolate_pos_encoding",val:": bool = False"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecVisionForImageClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessor">BeitImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitFeatureExtractor.__call__">BeitImageProcessor.<strong>call</strong>()</a> for details (<code>processor_class</code> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessor">BeitImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.Data2VecVisionForImageClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecVisionForImageClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the image classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.Data2VecVisionForImageClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecVisionForImageClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecVisionForImageClassification.forward.interpolate_pos_encoding",description:`<strong>interpolate_pos_encoding</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to interpolate the pre-trained position encodings.`,name:"interpolate_pos_encoding"},{anchor:"transformers.Data2VecVisionForImageClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_vision.py#L905",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput"
>transformers.modeling_outputs.ImageClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig"
>Data2VecVisionConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each stage) of shape <code>(batch_size, sequence_length, hidden_size)</code>. Hidden-states
(also called feature maps) of the model at the output of each stage.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, patch_size, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput"
>transformers.modeling_outputs.ImageClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Tt=new L({props:{$$slots:{default:[Qd]},$$scope:{ctx:w}}}),Mt=new J({props:{anchor:"transformers.Data2VecVisionForImageClassification.forward.example",$$slots:{default:[Pd]},$$scope:{ctx:w}}}),Qo=new $({props:{title:"Data2VecVisionForSemanticSegmentation",local:"transformers.Data2VecVisionForSemanticSegmentation",headingTag:"h2"}}),Po=new x({props:{name:"class transformers.Data2VecVisionForSemanticSegmentation",anchor:"transformers.Data2VecVisionForSemanticSegmentation",parameters:[{name:"config",val:": Data2VecVisionConfig"}],parametersDescription:[{anchor:"transformers.Data2VecVisionForSemanticSegmentation.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig">Data2VecVisionConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_vision.py#L1218"}}),Oo=new x({props:{name:"forward",anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward",parameters:[{name:"pixel_values",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"interpolate_pos_encoding",val:": bool = False"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessor">BeitImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitFeatureExtractor.__call__">BeitImageProcessor.<strong>call</strong>()</a> for details (<code>processor_class</code> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/beit#transformers.BeitImageProcessor">BeitImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Ground truth semantic segmentation maps for computing the loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels &gt; 1</code>, a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward.interpolate_pos_encoding",description:`<strong>interpolate_pos_encoding</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to interpolate the pre-trained position encodings.`,name:"interpolate_pos_encoding"},{anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/data2vec/modeling_data2vec_vision.py#L1270",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput"
>transformers.modeling_outputs.SemanticSegmenterOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/data2vec#transformers.Data2VecVisionConfig"
>Data2VecVisionConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels, logits_height, logits_width)</code>) — Classification scores for each pixel.</p>
<Tip warning={true}>
<p>The logits returned do not necessarily have the same size as the <code>pixel_values</code> passed as inputs. This is
to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
original image size as post-processing. You should always check your logits shape and resize as needed.</p>
</Tip>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, patch_size, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, patch_size, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput"
>transformers.modeling_outputs.SemanticSegmenterOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),vt=new L({props:{$$slots:{default:[Od]},$$scope:{ctx:w}}}),wt=new J({props:{anchor:"transformers.Data2VecVisionForSemanticSegmentation.forward.example",$$slots:{default:[Kd]},$$scope:{ctx:w}}}),Ko=new Td({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/data2vec.md"}}),{c(){t=p("meta"),T=s(),l=p("p"),d=s(),M=p("p"),M.innerHTML=o,v=s(),u(kt.$$.fragment),ha=s(),Ae=p("div"),Ae.innerHTML=Wi,ua=s(),u(Vt.$$.fragment),fa=s(),xt=p("p"),xt.innerHTML=Zi,ga=s(),Ct=p("p"),Ct.textContent=zi,_a=s(),$t=p("p"),$t.innerHTML=Gi,ba=s(),jt=p("p"),jt.innerHTML=Ii,ya=s(),Jt=p("p"),Jt.innerHTML=Ri,Ta=s(),u(Ft.$$.fragment),Ma=s(),Ut=p("ul"),Ut.innerHTML=Bi,va=s(),u(Dt.$$.fragment),wa=s(),Wt=p("p"),Wt.innerHTML=Ni,ka=s(),Zt=p("p"),Zt.innerHTML=qi,Va=s(),zt=p("p"),zt.textContent=Xi,xa=s(),u(Gt.$$.fragment),Ca=s(),It=p("p"),It.innerHTML=Ai,$a=s(),Rt=p("p"),Rt.innerHTML=Yi,ja=s(),u(Bt.$$.fragment),Ja=s(),Nt=p("table"),Nt.innerHTML=Li,Fa=s(),u(qt.$$.fragment),Ua=s(),Xt=p("table"),Xt.innerHTML=Si,Da=s(),u(At.$$.fragment),Wa=s(),Yt=p("p"),Yt.textContent=Hi,Za=s(),u(Lt.$$.fragment),za=s(),St=p("ul"),St.innerHTML=Ei,Ga=s(),Ht=p("p"),Ht.innerHTML=Qi,Ia=s(),Et=p("ul"),Et.innerHTML=Pi,Ra=s(),Qt=p("p"),Qt.innerHTML=Oi,Ba=s(),Pt=p("ul"),Pt.innerHTML=Ki,Na=s(),Ot=p("p"),Ot.innerHTML=el,qa=s(),Kt=p("ul"),Kt.innerHTML=tl,Xa=s(),eo=p("p"),eo.textContent=ol,Aa=s(),u(to.$$.fragment),Ya=s(),E=p("div"),u(oo.$$.fragment),js=s(),en=p("p"),en.innerHTML=nl,Js=s(),tn=p("p"),tn.innerHTML=al,Fs=s(),u(Ye.$$.fragment),La=s(),u(no.$$.fragment),Sa=s(),Q=p("div"),u(ao.$$.fragment),Us=s(),on=p("p"),on.innerHTML=sl,Ds=s(),nn=p("p"),nn.innerHTML=rl,Ws=s(),u(Le.$$.fragment),Ha=s(),u(so.$$.fragment),Ea=s(),Ve=p("div"),u(ro.$$.fragment),Zs=s(),an=p("p"),an.innerHTML=il,zs=s(),u(Se.$$.fragment),Qa=s(),u(io.$$.fragment),Pa=s(),F=p("div"),u(lo.$$.fragment),Gs=s(),sn=p("p"),sn.textContent=ll,Is=s(),rn=p("p"),rn.innerHTML=dl,Rs=s(),ln=p("p"),ln.innerHTML=cl,Bs=s(),Re=p("div"),u(co.$$.fragment),Ns=s(),dn=p("p"),dn.innerHTML=pl,qs=s(),u(He.$$.fragment),Oa=s(),u(po.$$.fragment),Ka=s(),U=p("div"),u(mo.$$.fragment),Xs=s(),cn=p("p"),cn.textContent=ml,As=s(),pn=p("p"),pn.innerHTML=hl,Ys=s(),mn=p("p"),mn.innerHTML=ul,Ls=s(),ue=p("div"),u(ho.$$.fragment),Ss=s(),hn=p("p"),hn.innerHTML=fl,Hs=s(),u(Ee.$$.fragment),Es=s(),u(Qe.$$.fragment),es=s(),u(uo.$$.fragment),ts=s(),D=p("div"),u(fo.$$.fragment),Qs=s(),un=p("p"),un.innerHTML=gl,Ps=s(),fn=p("p"),fn.innerHTML=_l,Os=s(),gn=p("p"),gn.innerHTML=bl,Ks=s(),fe=p("div"),u(go.$$.fragment),er=s(),_n=p("p"),_n.innerHTML=yl,tr=s(),u(Pe.$$.fragment),or=s(),u(Oe.$$.fragment),os=s(),u(_o.$$.fragment),ns=s(),W=p("div"),u(bo.$$.fragment),nr=s(),bn=p("p"),bn.textContent=Tl,ar=s(),yn=p("p"),yn.innerHTML=Ml,sr=s(),Tn=p("p"),Tn.innerHTML=vl,rr=s(),S=p("div"),u(yo.$$.fragment),ir=s(),Mn=p("p"),Mn.innerHTML=wl,lr=s(),u(Ke.$$.fragment),dr=s(),u(et.$$.fragment),cr=s(),u(tt.$$.fragment),as=s(),u(To.$$.fragment),ss=s(),Z=p("div"),u(Mo.$$.fragment),pr=s(),vn=p("p"),vn.textContent=kl,mr=s(),wn=p("p"),wn.innerHTML=Vl,hr=s(),kn=p("p"),kn.innerHTML=xl,ur=s(),ge=p("div"),u(vo.$$.fragment),fr=s(),Vn=p("p"),Vn.innerHTML=Cl,gr=s(),u(ot.$$.fragment),_r=s(),u(nt.$$.fragment),rs=s(),u(wo.$$.fragment),is=s(),z=p("div"),u(ko.$$.fragment),br=s(),xn=p("p"),xn.textContent=$l,yr=s(),Cn=p("p"),Cn.innerHTML=jl,Tr=s(),$n=p("p"),$n.innerHTML=Jl,Mr=s(),Be=p("div"),u(Vo.$$.fragment),vr=s(),jn=p("p"),jn.innerHTML=Fl,wr=s(),u(at.$$.fragment),ls=s(),u(xo.$$.fragment),ds=s(),G=p("div"),u(Co.$$.fragment),kr=s(),Jn=p("p"),Jn.innerHTML=Ul,Vr=s(),Fn=p("p"),Fn.innerHTML=Dl,xr=s(),Un=p("p"),Un.innerHTML=Wl,Cr=s(),_e=p("div"),u($o.$$.fragment),$r=s(),Dn=p("p"),Dn.innerHTML=Zl,jr=s(),u(st.$$.fragment),Jr=s(),u(rt.$$.fragment),cs=s(),u(jo.$$.fragment),ps=s(),I=p("div"),u(Jo.$$.fragment),Fr=s(),Wn=p("p"),Wn.innerHTML=zl,Ur=s(),Zn=p("p"),Zn.innerHTML=Gl,Dr=s(),zn=p("p"),zn.innerHTML=Il,Wr=s(),be=p("div"),u(Fo.$$.fragment),Zr=s(),Gn=p("p"),Gn.innerHTML=Rl,zr=s(),u(it.$$.fragment),Gr=s(),u(lt.$$.fragment),ms=s(),u(Uo.$$.fragment),hs=s(),R=p("div"),u(Do.$$.fragment),Ir=s(),In=p("p"),In.textContent=Bl,Rr=s(),Rn=p("p"),Rn.innerHTML=Nl,Br=s(),Bn=p("p"),Bn.innerHTML=ql,Nr=s(),H=p("div"),u(Wo.$$.fragment),qr=s(),Nn=p("p"),Nn.innerHTML=Xl,Xr=s(),u(dt.$$.fragment),Ar=s(),u(ct.$$.fragment),Yr=s(),u(pt.$$.fragment),us=s(),u(Zo.$$.fragment),fs=s(),B=p("div"),u(zo.$$.fragment),Lr=s(),qn=p("p"),qn.textContent=Al,Sr=s(),Xn=p("p"),Xn.innerHTML=Yl,Hr=s(),An=p("p"),An.innerHTML=Ll,Er=s(),ye=p("div"),u(Go.$$.fragment),Qr=s(),Yn=p("p"),Yn.innerHTML=Sl,Pr=s(),u(mt.$$.fragment),Or=s(),u(ht.$$.fragment),gs=s(),u(Io.$$.fragment),_s=s(),N=p("div"),u(Ro.$$.fragment),Kr=s(),Ln=p("p"),Ln.textContent=Hl,ei=s(),Sn=p("p"),Sn.innerHTML=El,ti=s(),Hn=p("p"),Hn.innerHTML=Ql,oi=s(),Te=p("div"),u(Bo.$$.fragment),ni=s(),En=p("p"),En.innerHTML=Pl,ai=s(),u(ut.$$.fragment),si=s(),u(ft.$$.fragment),bs=s(),u(No.$$.fragment),ys=s(),q=p("div"),u(qo.$$.fragment),ri=s(),Qn=p("p"),Qn.innerHTML=Ol,ii=s(),Pn=p("p"),Pn.innerHTML=Kl,li=s(),On=p("p"),On.innerHTML=ed,di=s(),Me=p("div"),u(Xo.$$.fragment),ci=s(),Kn=p("p"),Kn.innerHTML=td,pi=s(),u(gt.$$.fragment),mi=s(),u(_t.$$.fragment),Ts=s(),u(Ao.$$.fragment),Ms=s(),X=p("div"),u(Yo.$$.fragment),hi=s(),ea=p("p"),ea.textContent=od,ui=s(),ta=p("p"),ta.innerHTML=nd,fi=s(),oa=p("p"),oa.innerHTML=ad,gi=s(),ve=p("div"),u(Lo.$$.fragment),_i=s(),na=p("p"),na.innerHTML=sd,bi=s(),u(bt.$$.fragment),yi=s(),u(yt.$$.fragment),vs=s(),u(So.$$.fragment),ws=s(),A=p("div"),u(Ho.$$.fragment),Ti=s(),aa=p("p"),aa.textContent=rd,Mi=s(),sa=p("p"),sa.innerHTML=id,vi=s(),ra=p("p"),ra.innerHTML=ld,wi=s(),we=p("div"),u(Eo.$$.fragment),ki=s(),ia=p("p"),ia.innerHTML=dd,Vi=s(),u(Tt.$$.fragment),xi=s(),u(Mt.$$.fragment),ks=s(),u(Qo.$$.fragment),Vs=s(),Y=p("div"),u(Po.$$.fragment),Ci=s(),la=p("p"),la.textContent=cd,$i=s(),da=p("p"),da.innerHTML=pd,ji=s(),ca=p("p"),ca.innerHTML=md,Ji=s(),ke=p("div"),u(Oo.$$.fragment),Fi=s(),pa=p("p"),pa.innerHTML=hd,Ui=s(),u(vt.$$.fragment),Di=s(),u(wt.$$.fragment),xs=s(),u(Ko.$$.fragment),Cs=s(),ma=p("p"),this.h()},l(e){const n=bd("svelte-u9bgzb",document.head);t=m(n,"META",{name:!0,content:!0}),n.forEach(a),T=r(e),l=m(e,"P",{}),V(l).forEach(a),d=r(e),M=m(e,"P",{"data-svelte-h":!0}),h(M)!=="svelte-nnbvx5"&&(M.innerHTML=o),v=r(e),f(kt.$$.fragment,e),ha=r(e),Ae=m(e,"DIV",{class:!0,"data-svelte-h":!0}),h(Ae)!=="svelte-b95w5j"&&(Ae.innerHTML=Wi),ua=r(e),f(Vt.$$.fragment,e),fa=r(e),xt=m(e,"P",{"data-svelte-h":!0}),h(xt)!=="svelte-w0t1fa"&&(xt.innerHTML=Zi),ga=r(e),Ct=m(e,"P",{"data-svelte-h":!0}),h(Ct)!=="svelte-vfdo9a"&&(Ct.textContent=zi),_a=r(e),$t=m(e,"P",{"data-svelte-h":!0}),h($t)!=="svelte-p7b135"&&($t.innerHTML=Gi),ba=r(e),jt=m(e,"P",{"data-svelte-h":!0}),h(jt)!=="svelte-vamg3z"&&(jt.innerHTML=Ii),ya=r(e),Jt=m(e,"P",{"data-svelte-h":!0}),h(Jt)!=="svelte-hfd38w"&&(Jt.innerHTML=Ri),Ta=r(e),f(Ft.$$.fragment,e),Ma=r(e),Ut=m(e,"UL",{"data-svelte-h":!0}),h(Ut)!=="svelte-1t0xgkk"&&(Ut.innerHTML=Bi),va=r(e),f(Dt.$$.fragment,e),wa=r(e),Wt=m(e,"P",{"data-svelte-h":!0}),h(Wt)!=="svelte-1cid2pe"&&(Wt.innerHTML=Ni),ka=r(e),Zt=m(e,"P",{"data-svelte-h":!0}),h(Zt)!=="svelte-1x11lxg"&&(Zt.innerHTML=qi),Va=r(e),zt=m(e,"P",{"data-svelte-h":!0}),h(zt)!=="svelte-e6eddf"&&(zt.textContent=Xi),xa=r(e),f(Gt.$$.fragment,e),Ca=r(e),It=m(e,"P",{"data-svelte-h":!0}),h(It)!=="svelte-djb2w0"&&(It.innerHTML=Ai),$a=r(e),Rt=m(e,"P",{"data-svelte-h":!0}),h(Rt)!=="svelte-ncqwsi"&&(Rt.innerHTML=Yi),ja=r(e),f(Bt.$$.fragment,e),Ja=r(e),Nt=m(e,"TABLE",{"data-svelte-h":!0}),h(Nt)!=="svelte-1c1is8e"&&(Nt.innerHTML=Li),Fa=r(e),f(qt.$$.fragment,e),Ua=r(e),Xt=m(e,"TABLE",{"data-svelte-h":!0}),h(Xt)!=="svelte-t2ewyd"&&(Xt.innerHTML=Si),Da=r(e),f(At.$$.fragment,e),Wa=r(e),Yt=m(e,"P",{"data-svelte-h":!0}),h(Yt)!=="svelte-1k2qpaj"&&(Yt.textContent=Hi),Za=r(e),f(Lt.$$.fragment,e),za=r(e),St=m(e,"UL",{"data-svelte-h":!0}),h(St)!=="svelte-1o9ejcx"&&(St.innerHTML=Ei),Ga=r(e),Ht=m(e,"P",{"data-svelte-h":!0}),h(Ht)!=="svelte-1q8dm51"&&(Ht.innerHTML=Qi),Ia=r(e),Et=m(e,"UL",{"data-svelte-h":!0}),h(Et)!=="svelte-p1b16m"&&(Et.innerHTML=Pi),Ra=r(e),Qt=m(e,"P",{"data-svelte-h":!0}),h(Qt)!=="svelte-3b41eg"&&(Qt.innerHTML=Oi),Ba=r(e),Pt=m(e,"UL",{"data-svelte-h":!0}),h(Pt)!=="svelte-11qmliz"&&(Pt.innerHTML=Ki),Na=r(e),Ot=m(e,"P",{"data-svelte-h":!0}),h(Ot)!=="svelte-ytlszg"&&(Ot.innerHTML=el),qa=r(e),Kt=m(e,"UL",{"data-svelte-h":!0}),h(Kt)!=="svelte-1kxvcoe"&&(Kt.innerHTML=tl),Xa=r(e),eo=m(e,"P",{"data-svelte-h":!0}),h(eo)!=="svelte-1xesile"&&(eo.textContent=ol),Aa=r(e),f(to.$$.fragment,e),Ya=r(e),E=m(e,"DIV",{class:!0});var xe=V(E);f(oo.$$.fragment,xe),js=r(xe),en=m(xe,"P",{"data-svelte-h":!0}),h(en)!=="svelte-r1navl"&&(en.innerHTML=nl),Js=r(xe),tn=m(xe,"P",{"data-svelte-h":!0}),h(tn)!=="svelte-1ek1ss9"&&(tn.innerHTML=al),Fs=r(xe),f(Ye.$$.fragment,xe),xe.forEach(a),La=r(e),f(no.$$.fragment,e),Sa=r(e),Q=m(e,"DIV",{class:!0});var Ce=V(Q);f(ao.$$.fragment,Ce),Us=r(Ce),on=m(Ce,"P",{"data-svelte-h":!0}),h(on)!=="svelte-w0zy8z"&&(on.innerHTML=sl),Ds=r(Ce),nn=m(Ce,"P",{"data-svelte-h":!0}),h(nn)!=="svelte-1ek1ss9"&&(nn.innerHTML=rl),Ws=r(Ce),f(Le.$$.fragment,Ce),Ce.forEach(a),Ha=r(e),f(so.$$.fragment,e),Ea=r(e),Ve=m(e,"DIV",{class:!0});var Ne=V(Ve);f(ro.$$.fragment,Ne),Zs=r(Ne),an=m(Ne,"P",{"data-svelte-h":!0}),h(an)!=="svelte-1oogo8h"&&(an.innerHTML=il),zs=r(Ne),f(Se.$$.fragment,Ne),Ne.forEach(a),Qa=r(e),f(io.$$.fragment,e),Pa=r(e),F=m(e,"DIV",{class:!0});var P=V(F);f(lo.$$.fragment,P),Gs=r(P),sn=m(P,"P",{"data-svelte-h":!0}),h(sn)!=="svelte-1pter3e"&&(sn.textContent=ll),Is=r(P),rn=m(P,"P",{"data-svelte-h":!0}),h(rn)!=="svelte-q52n56"&&(rn.innerHTML=dl),Rs=r(P),ln=m(P,"P",{"data-svelte-h":!0}),h(ln)!=="svelte-hswkmf"&&(ln.innerHTML=cl),Bs=r(P),Re=m(P,"DIV",{class:!0});var qe=V(Re);f(co.$$.fragment,qe),Ns=r(qe),dn=m(qe,"P",{"data-svelte-h":!0}),h(dn)!=="svelte-a68in1"&&(dn.innerHTML=pl),qs=r(qe),f(He.$$.fragment,qe),qe.forEach(a),P.forEach(a),Oa=r(e),f(po.$$.fragment,e),Ka=r(e),U=m(e,"DIV",{class:!0});var O=V(U);f(mo.$$.fragment,O),Xs=r(O),cn=m(O,"P",{"data-svelte-h":!0}),h(cn)!=="svelte-f5mg43"&&(cn.textContent=ml),As=r(O),pn=m(O,"P",{"data-svelte-h":!0}),h(pn)!=="svelte-q52n56"&&(pn.innerHTML=hl),Ys=r(O),mn=m(O,"P",{"data-svelte-h":!0}),h(mn)!=="svelte-hswkmf"&&(mn.innerHTML=ul),Ls=r(O),ue=m(O,"DIV",{class:!0});var $e=V(ue);f(ho.$$.fragment,$e),Ss=r($e),hn=m($e,"P",{"data-svelte-h":!0}),h(hn)!=="svelte-q2fier"&&(hn.innerHTML=fl),Hs=r($e),f(Ee.$$.fragment,$e),Es=r($e),f(Qe.$$.fragment,$e),$e.forEach(a),O.forEach(a),es=r(e),f(uo.$$.fragment,e),ts=r(e),D=m(e,"DIV",{class:!0});var K=V(D);f(fo.$$.fragment,K),Qs=r(K),un=m(K,"P",{"data-svelte-h":!0}),h(un)!=="svelte-1uea8pq"&&(un.innerHTML=gl),Ps=r(K),fn=m(K,"P",{"data-svelte-h":!0}),h(fn)!=="svelte-q52n56"&&(fn.innerHTML=_l),Os=r(K),gn=m(K,"P",{"data-svelte-h":!0}),h(gn)!=="svelte-hswkmf"&&(gn.innerHTML=bl),Ks=r(K),fe=m(K,"DIV",{class:!0});var je=V(fe);f(go.$$.fragment,je),er=r(je),_n=m(je,"P",{"data-svelte-h":!0}),h(_n)!=="svelte-41kxj7"&&(_n.innerHTML=yl),tr=r(je),f(Pe.$$.fragment,je),or=r(je),f(Oe.$$.fragment,je),je.forEach(a),K.forEach(a),os=r(e),f(_o.$$.fragment,e),ns=r(e),W=m(e,"DIV",{class:!0});var ee=V(W);f(bo.$$.fragment,ee),nr=r(ee),bn=m(ee,"P",{"data-svelte-h":!0}),h(bn)!=="svelte-go4okw"&&(bn.textContent=Tl),ar=r(ee),yn=m(ee,"P",{"data-svelte-h":!0}),h(yn)!=="svelte-q52n56"&&(yn.innerHTML=Ml),sr=r(ee),Tn=m(ee,"P",{"data-svelte-h":!0}),h(Tn)!=="svelte-hswkmf"&&(Tn.innerHTML=vl),rr=r(ee),S=m(ee,"DIV",{class:!0});var te=V(S);f(yo.$$.fragment,te),ir=r(te),Mn=m(te,"P",{"data-svelte-h":!0}),h(Mn)!=="svelte-g1rntv"&&(Mn.innerHTML=wl),lr=r(te),f(Ke.$$.fragment,te),dr=r(te),f(et.$$.fragment,te),cr=r(te),f(tt.$$.fragment,te),te.forEach(a),ee.forEach(a),as=r(e),f(To.$$.fragment,e),ss=r(e),Z=m(e,"DIV",{class:!0});var oe=V(Z);f(Mo.$$.fragment,oe),pr=r(oe),vn=m(oe,"P",{"data-svelte-h":!0}),h(vn)!=="svelte-1x720uc"&&(vn.textContent=kl),mr=r(oe),wn=m(oe,"P",{"data-svelte-h":!0}),h(wn)!=="svelte-q52n56"&&(wn.innerHTML=Vl),hr=r(oe),kn=m(oe,"P",{"data-svelte-h":!0}),h(kn)!=="svelte-hswkmf"&&(kn.innerHTML=xl),ur=r(oe),ge=m(oe,"DIV",{class:!0});var Je=V(ge);f(vo.$$.fragment,Je),fr=r(Je),Vn=m(Je,"P",{"data-svelte-h":!0}),h(Vn)!=="svelte-kw71zr"&&(Vn.innerHTML=Cl),gr=r(Je),f(ot.$$.fragment,Je),_r=r(Je),f(nt.$$.fragment,Je),Je.forEach(a),oe.forEach(a),rs=r(e),f(wo.$$.fragment,e),is=r(e),z=m(e,"DIV",{class:!0});var ne=V(z);f(ko.$$.fragment,ne),br=r(ne),xn=m(ne,"P",{"data-svelte-h":!0}),h(xn)!=="svelte-1k6dp6m"&&(xn.textContent=$l),yr=r(ne),Cn=m(ne,"P",{"data-svelte-h":!0}),h(Cn)!=="svelte-q52n56"&&(Cn.innerHTML=jl),Tr=r(ne),$n=m(ne,"P",{"data-svelte-h":!0}),h($n)!=="svelte-hswkmf"&&($n.innerHTML=Jl),Mr=r(ne),Be=m(ne,"DIV",{class:!0});var Xe=V(Be);f(Vo.$$.fragment,Xe),vr=r(Xe),jn=m(Xe,"P",{"data-svelte-h":!0}),h(jn)!=="svelte-13klftv"&&(jn.innerHTML=Fl),wr=r(Xe),f(at.$$.fragment,Xe),Xe.forEach(a),ne.forEach(a),ls=r(e),f(xo.$$.fragment,e),ds=r(e),G=m(e,"DIV",{class:!0});var ae=V(G);f(Co.$$.fragment,ae),kr=r(ae),Jn=m(ae,"P",{"data-svelte-h":!0}),h(Jn)!=="svelte-1yzozvu"&&(Jn.innerHTML=Ul),Vr=r(ae),Fn=m(ae,"P",{"data-svelte-h":!0}),h(Fn)!=="svelte-q52n56"&&(Fn.innerHTML=Dl),xr=r(ae),Un=m(ae,"P",{"data-svelte-h":!0}),h(Un)!=="svelte-hswkmf"&&(Un.innerHTML=Wl),Cr=r(ae),_e=m(ae,"DIV",{class:!0});var Fe=V(_e);f($o.$$.fragment,Fe),$r=r(Fe),Dn=m(Fe,"P",{"data-svelte-h":!0}),h(Dn)!=="svelte-1rn310b"&&(Dn.innerHTML=Zl),jr=r(Fe),f(st.$$.fragment,Fe),Jr=r(Fe),f(rt.$$.fragment,Fe),Fe.forEach(a),ae.forEach(a),cs=r(e),f(jo.$$.fragment,e),ps=r(e),I=m(e,"DIV",{class:!0});var se=V(I);f(Jo.$$.fragment,se),Fr=r(se),Wn=m(se,"P",{"data-svelte-h":!0}),h(Wn)!=="svelte-ygbe0d"&&(Wn.innerHTML=zl),Ur=r(se),Zn=m(se,"P",{"data-svelte-h":!0}),h(Zn)!=="svelte-q52n56"&&(Zn.innerHTML=Gl),Dr=r(se),zn=m(se,"P",{"data-svelte-h":!0}),h(zn)!=="svelte-hswkmf"&&(zn.innerHTML=Il),Wr=r(se),be=m(se,"DIV",{class:!0});var Ue=V(be);f(Fo.$$.fragment,Ue),Zr=r(Ue),Gn=m(Ue,"P",{"data-svelte-h":!0}),h(Gn)!=="svelte-f8mh8j"&&(Gn.innerHTML=Rl),zr=r(Ue),f(it.$$.fragment,Ue),Gr=r(Ue),f(lt.$$.fragment,Ue),Ue.forEach(a),se.forEach(a),ms=r(e),f(Uo.$$.fragment,e),hs=r(e),R=m(e,"DIV",{class:!0});var re=V(R);f(Do.$$.fragment,re),Ir=r(re),In=m(re,"P",{"data-svelte-h":!0}),h(In)!=="svelte-5exrgz"&&(In.textContent=Bl),Rr=r(re),Rn=m(re,"P",{"data-svelte-h":!0}),h(Rn)!=="svelte-q52n56"&&(Rn.innerHTML=Nl),Br=r(re),Bn=m(re,"P",{"data-svelte-h":!0}),h(Bn)!=="svelte-hswkmf"&&(Bn.innerHTML=ql),Nr=r(re),H=m(re,"DIV",{class:!0});var ie=V(H);f(Wo.$$.fragment,ie),qr=r(ie),Nn=m(ie,"P",{"data-svelte-h":!0}),h(Nn)!=="svelte-90fxz7"&&(Nn.innerHTML=Xl),Xr=r(ie),f(dt.$$.fragment,ie),Ar=r(ie),f(ct.$$.fragment,ie),Yr=r(ie),f(pt.$$.fragment,ie),ie.forEach(a),re.forEach(a),us=r(e),f(Zo.$$.fragment,e),fs=r(e),B=m(e,"DIV",{class:!0});var le=V(B);f(zo.$$.fragment,le),Lr=r(le),qn=m(le,"P",{"data-svelte-h":!0}),h(qn)!=="svelte-80uqni"&&(qn.textContent=Al),Sr=r(le),Xn=m(le,"P",{"data-svelte-h":!0}),h(Xn)!=="svelte-q52n56"&&(Xn.innerHTML=Yl),Hr=r(le),An=m(le,"P",{"data-svelte-h":!0}),h(An)!=="svelte-hswkmf"&&(An.innerHTML=Ll),Er=r(le),ye=m(le,"DIV",{class:!0});var De=V(ye);f(Go.$$.fragment,De),Qr=r(De),Yn=m(De,"P",{"data-svelte-h":!0}),h(Yn)!=="svelte-npehuz"&&(Yn.innerHTML=Sl),Pr=r(De),f(mt.$$.fragment,De),Or=r(De),f(ht.$$.fragment,De),De.forEach(a),le.forEach(a),gs=r(e),f(Io.$$.fragment,e),_s=r(e),N=m(e,"DIV",{class:!0});var de=V(N);f(Ro.$$.fragment,de),Kr=r(de),Ln=m(de,"P",{"data-svelte-h":!0}),h(Ln)!=="svelte-139jurl"&&(Ln.textContent=Hl),ei=r(de),Sn=m(de,"P",{"data-svelte-h":!0}),h(Sn)!=="svelte-q52n56"&&(Sn.innerHTML=El),ti=r(de),Hn=m(de,"P",{"data-svelte-h":!0}),h(Hn)!=="svelte-hswkmf"&&(Hn.innerHTML=Ql),oi=r(de),Te=m(de,"DIV",{class:!0});var We=V(Te);f(Bo.$$.fragment,We),ni=r(We),En=m(We,"P",{"data-svelte-h":!0}),h(En)!=="svelte-1n6ev11"&&(En.innerHTML=Pl),ai=r(We),f(ut.$$.fragment,We),si=r(We),f(ft.$$.fragment,We),We.forEach(a),de.forEach(a),bs=r(e),f(No.$$.fragment,e),ys=r(e),q=m(e,"DIV",{class:!0});var ce=V(q);f(qo.$$.fragment,ce),ri=r(ce),Qn=m(ce,"P",{"data-svelte-h":!0}),h(Qn)!=="svelte-4mmyqi"&&(Qn.innerHTML=Ol),ii=r(ce),Pn=m(ce,"P",{"data-svelte-h":!0}),h(Pn)!=="svelte-q52n56"&&(Pn.innerHTML=Kl),li=r(ce),On=m(ce,"P",{"data-svelte-h":!0}),h(On)!=="svelte-hswkmf"&&(On.innerHTML=ed),di=r(ce),Me=m(ce,"DIV",{class:!0});var Ze=V(Me);f(Xo.$$.fragment,Ze),ci=r(Ze),Kn=m(Ze,"P",{"data-svelte-h":!0}),h(Kn)!=="svelte-19ylslf"&&(Kn.innerHTML=td),pi=r(Ze),f(gt.$$.fragment,Ze),mi=r(Ze),f(_t.$$.fragment,Ze),Ze.forEach(a),ce.forEach(a),Ts=r(e),f(Ao.$$.fragment,e),Ms=r(e),X=m(e,"DIV",{class:!0});var pe=V(X);f(Yo.$$.fragment,pe),hi=r(pe),ea=m(pe,"P",{"data-svelte-h":!0}),h(ea)!=="svelte-cuh4se"&&(ea.textContent=od),ui=r(pe),ta=m(pe,"P",{"data-svelte-h":!0}),h(ta)!=="svelte-q52n56"&&(ta.innerHTML=nd),fi=r(pe),oa=m(pe,"P",{"data-svelte-h":!0}),h(oa)!=="svelte-hswkmf"&&(oa.innerHTML=ad),gi=r(pe),ve=m(pe,"DIV",{class:!0});var ze=V(ve);f(Lo.$$.fragment,ze),_i=r(ze),na=m(ze,"P",{"data-svelte-h":!0}),h(na)!=="svelte-1ckg54f"&&(na.innerHTML=sd),bi=r(ze),f(bt.$$.fragment,ze),yi=r(ze),f(yt.$$.fragment,ze),ze.forEach(a),pe.forEach(a),vs=r(e),f(So.$$.fragment,e),ws=r(e),A=m(e,"DIV",{class:!0});var me=V(A);f(Ho.$$.fragment,me),Ti=r(me),aa=m(me,"P",{"data-svelte-h":!0}),h(aa)!=="svelte-jbss5k"&&(aa.textContent=rd),Mi=r(me),sa=m(me,"P",{"data-svelte-h":!0}),h(sa)!=="svelte-q52n56"&&(sa.innerHTML=id),vi=r(me),ra=m(me,"P",{"data-svelte-h":!0}),h(ra)!=="svelte-hswkmf"&&(ra.innerHTML=ld),wi=r(me),we=m(me,"DIV",{class:!0});var Ge=V(we);f(Eo.$$.fragment,Ge),ki=r(Ge),ia=m(Ge,"P",{"data-svelte-h":!0}),h(ia)!=="svelte-x8btxv"&&(ia.innerHTML=dd),Vi=r(Ge),f(Tt.$$.fragment,Ge),xi=r(Ge),f(Mt.$$.fragment,Ge),Ge.forEach(a),me.forEach(a),ks=r(e),f(Qo.$$.fragment,e),Vs=r(e),Y=m(e,"DIV",{class:!0});var he=V(Y);f(Po.$$.fragment,he),Ci=r(he),la=m(he,"P",{"data-svelte-h":!0}),h(la)!=="svelte-1jeckm7"&&(la.textContent=cd),$i=r(he),da=m(he,"P",{"data-svelte-h":!0}),h(da)!=="svelte-q52n56"&&(da.innerHTML=pd),ji=r(he),ca=m(he,"P",{"data-svelte-h":!0}),h(ca)!=="svelte-hswkmf"&&(ca.innerHTML=md),Ji=r(he),ke=m(he,"DIV",{class:!0});var Ie=V(ke);f(Oo.$$.fragment,Ie),Fi=r(Ie),pa=m(Ie,"P",{"data-svelte-h":!0}),h(pa)!=="svelte-jqjqtf"&&(pa.innerHTML=hd),Ui=r(Ie),f(vt.$$.fragment,Ie),Di=r(Ie),f(wt.$$.fragment,Ie),Ie.forEach(a),he.forEach(a),xs=r(e),f(Ko.$$.fragment,e),Cs=r(e),ma=m(e,"P",{}),V(ma).forEach(a),this.h()},h(){k(t,"name","hf:doc:metadata"),k(t,"content",tc),k(Ae,"class","flex flex-wrap space-x-1"),k(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(_e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(we,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){i(document.head,t),c(e,T,n),c(e,l,n),c(e,d,n),c(e,M,n),c(e,v,n),g(kt,e,n),c(e,ha,n),c(e,Ae,n),c(e,ua,n),g(Vt,e,n),c(e,fa,n),c(e,xt,n),c(e,ga,n),c(e,Ct,n),c(e,_a,n),c(e,$t,n),c(e,ba,n),c(e,jt,n),c(e,ya,n),c(e,Jt,n),c(e,Ta,n),g(Ft,e,n),c(e,Ma,n),c(e,Ut,n),c(e,va,n),g(Dt,e,n),c(e,wa,n),c(e,Wt,n),c(e,ka,n),c(e,Zt,n),c(e,Va,n),c(e,zt,n),c(e,xa,n),g(Gt,e,n),c(e,Ca,n),c(e,It,n),c(e,$a,n),c(e,Rt,n),c(e,ja,n),g(Bt,e,n),c(e,Ja,n),c(e,Nt,n),c(e,Fa,n),g(qt,e,n),c(e,Ua,n),c(e,Xt,n),c(e,Da,n),g(At,e,n),c(e,Wa,n),c(e,Yt,n),c(e,Za,n),g(Lt,e,n),c(e,za,n),c(e,St,n),c(e,Ga,n),c(e,Ht,n),c(e,Ia,n),c(e,Et,n),c(e,Ra,n),c(e,Qt,n),c(e,Ba,n),c(e,Pt,n),c(e,Na,n),c(e,Ot,n),c(e,qa,n),c(e,Kt,n),c(e,Xa,n),c(e,eo,n),c(e,Aa,n),g(to,e,n),c(e,Ya,n),c(e,E,n),g(oo,E,null),i(E,js),i(E,en),i(E,Js),i(E,tn),i(E,Fs),g(Ye,E,null),c(e,La,n),g(no,e,n),c(e,Sa,n),c(e,Q,n),g(ao,Q,null),i(Q,Us),i(Q,on),i(Q,Ds),i(Q,nn),i(Q,Ws),g(Le,Q,null),c(e,Ha,n),g(so,e,n),c(e,Ea,n),c(e,Ve,n),g(ro,Ve,null),i(Ve,Zs),i(Ve,an),i(Ve,zs),g(Se,Ve,null),c(e,Qa,n),g(io,e,n),c(e,Pa,n),c(e,F,n),g(lo,F,null),i(F,Gs),i(F,sn),i(F,Is),i(F,rn),i(F,Rs),i(F,ln),i(F,Bs),i(F,Re),g(co,Re,null),i(Re,Ns),i(Re,dn),i(Re,qs),g(He,Re,null),c(e,Oa,n),g(po,e,n),c(e,Ka,n),c(e,U,n),g(mo,U,null),i(U,Xs),i(U,cn),i(U,As),i(U,pn),i(U,Ys),i(U,mn),i(U,Ls),i(U,ue),g(ho,ue,null),i(ue,Ss),i(ue,hn),i(ue,Hs),g(Ee,ue,null),i(ue,Es),g(Qe,ue,null),c(e,es,n),g(uo,e,n),c(e,ts,n),c(e,D,n),g(fo,D,null),i(D,Qs),i(D,un),i(D,Ps),i(D,fn),i(D,Os),i(D,gn),i(D,Ks),i(D,fe),g(go,fe,null),i(fe,er),i(fe,_n),i(fe,tr),g(Pe,fe,null),i(fe,or),g(Oe,fe,null),c(e,os,n),g(_o,e,n),c(e,ns,n),c(e,W,n),g(bo,W,null),i(W,nr),i(W,bn),i(W,ar),i(W,yn),i(W,sr),i(W,Tn),i(W,rr),i(W,S),g(yo,S,null),i(S,ir),i(S,Mn),i(S,lr),g(Ke,S,null),i(S,dr),g(et,S,null),i(S,cr),g(tt,S,null),c(e,as,n),g(To,e,n),c(e,ss,n),c(e,Z,n),g(Mo,Z,null),i(Z,pr),i(Z,vn),i(Z,mr),i(Z,wn),i(Z,hr),i(Z,kn),i(Z,ur),i(Z,ge),g(vo,ge,null),i(ge,fr),i(ge,Vn),i(ge,gr),g(ot,ge,null),i(ge,_r),g(nt,ge,null),c(e,rs,n),g(wo,e,n),c(e,is,n),c(e,z,n),g(ko,z,null),i(z,br),i(z,xn),i(z,yr),i(z,Cn),i(z,Tr),i(z,$n),i(z,Mr),i(z,Be),g(Vo,Be,null),i(Be,vr),i(Be,jn),i(Be,wr),g(at,Be,null),c(e,ls,n),g(xo,e,n),c(e,ds,n),c(e,G,n),g(Co,G,null),i(G,kr),i(G,Jn),i(G,Vr),i(G,Fn),i(G,xr),i(G,Un),i(G,Cr),i(G,_e),g($o,_e,null),i(_e,$r),i(_e,Dn),i(_e,jr),g(st,_e,null),i(_e,Jr),g(rt,_e,null),c(e,cs,n),g(jo,e,n),c(e,ps,n),c(e,I,n),g(Jo,I,null),i(I,Fr),i(I,Wn),i(I,Ur),i(I,Zn),i(I,Dr),i(I,zn),i(I,Wr),i(I,be),g(Fo,be,null),i(be,Zr),i(be,Gn),i(be,zr),g(it,be,null),i(be,Gr),g(lt,be,null),c(e,ms,n),g(Uo,e,n),c(e,hs,n),c(e,R,n),g(Do,R,null),i(R,Ir),i(R,In),i(R,Rr),i(R,Rn),i(R,Br),i(R,Bn),i(R,Nr),i(R,H),g(Wo,H,null),i(H,qr),i(H,Nn),i(H,Xr),g(dt,H,null),i(H,Ar),g(ct,H,null),i(H,Yr),g(pt,H,null),c(e,us,n),g(Zo,e,n),c(e,fs,n),c(e,B,n),g(zo,B,null),i(B,Lr),i(B,qn),i(B,Sr),i(B,Xn),i(B,Hr),i(B,An),i(B,Er),i(B,ye),g(Go,ye,null),i(ye,Qr),i(ye,Yn),i(ye,Pr),g(mt,ye,null),i(ye,Or),g(ht,ye,null),c(e,gs,n),g(Io,e,n),c(e,_s,n),c(e,N,n),g(Ro,N,null),i(N,Kr),i(N,Ln),i(N,ei),i(N,Sn),i(N,ti),i(N,Hn),i(N,oi),i(N,Te),g(Bo,Te,null),i(Te,ni),i(Te,En),i(Te,ai),g(ut,Te,null),i(Te,si),g(ft,Te,null),c(e,bs,n),g(No,e,n),c(e,ys,n),c(e,q,n),g(qo,q,null),i(q,ri),i(q,Qn),i(q,ii),i(q,Pn),i(q,li),i(q,On),i(q,di),i(q,Me),g(Xo,Me,null),i(Me,ci),i(Me,Kn),i(Me,pi),g(gt,Me,null),i(Me,mi),g(_t,Me,null),c(e,Ts,n),g(Ao,e,n),c(e,Ms,n),c(e,X,n),g(Yo,X,null),i(X,hi),i(X,ea),i(X,ui),i(X,ta),i(X,fi),i(X,oa),i(X,gi),i(X,ve),g(Lo,ve,null),i(ve,_i),i(ve,na),i(ve,bi),g(bt,ve,null),i(ve,yi),g(yt,ve,null),c(e,vs,n),g(So,e,n),c(e,ws,n),c(e,A,n),g(Ho,A,null),i(A,Ti),i(A,aa),i(A,Mi),i(A,sa),i(A,vi),i(A,ra),i(A,wi),i(A,we),g(Eo,we,null),i(we,ki),i(we,ia),i(we,Vi),g(Tt,we,null),i(we,xi),g(Mt,we,null),c(e,ks,n),g(Qo,e,n),c(e,Vs,n),c(e,Y,n),g(Po,Y,null),i(Y,Ci),i(Y,la),i(Y,$i),i(Y,da),i(Y,ji),i(Y,ca),i(Y,Ji),i(Y,ke),g(Oo,ke,null),i(ke,Fi),i(ke,pa),i(ke,Ui),g(vt,ke,null),i(ke,Di),g(wt,ke,null),c(e,xs,n),g(Ko,e,n),c(e,Cs,n),c(e,ma,n),$s=!0},p(e,[n]){const xe={};n&2&&(xe.$$scope={dirty:n,ctx:e}),Ye.$set(xe);const Ce={};n&2&&(Ce.$$scope={dirty:n,ctx:e}),Le.$set(Ce);const Ne={};n&2&&(Ne.$$scope={dirty:n,ctx:e}),Se.$set(Ne);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),He.$set(P);const qe={};n&2&&(qe.$$scope={dirty:n,ctx:e}),Ee.$set(qe);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),Qe.$set(O);const $e={};n&2&&($e.$$scope={dirty:n,ctx:e}),Pe.$set($e);const K={};n&2&&(K.$$scope={dirty:n,ctx:e}),Oe.$set(K);const je={};n&2&&(je.$$scope={dirty:n,ctx:e}),Ke.$set(je);const ee={};n&2&&(ee.$$scope={dirty:n,ctx:e}),et.$set(ee);const te={};n&2&&(te.$$scope={dirty:n,ctx:e}),tt.$set(te);const oe={};n&2&&(oe.$$scope={dirty:n,ctx:e}),ot.$set(oe);const Je={};n&2&&(Je.$$scope={dirty:n,ctx:e}),nt.$set(Je);const ne={};n&2&&(ne.$$scope={dirty:n,ctx:e}),at.$set(ne);const Xe={};n&2&&(Xe.$$scope={dirty:n,ctx:e}),st.$set(Xe);const ae={};n&2&&(ae.$$scope={dirty:n,ctx:e}),rt.$set(ae);const Fe={};n&2&&(Fe.$$scope={dirty:n,ctx:e}),it.$set(Fe);const se={};n&2&&(se.$$scope={dirty:n,ctx:e}),lt.$set(se);const Ue={};n&2&&(Ue.$$scope={dirty:n,ctx:e}),dt.$set(Ue);const re={};n&2&&(re.$$scope={dirty:n,ctx:e}),ct.$set(re);const ie={};n&2&&(ie.$$scope={dirty:n,ctx:e}),pt.$set(ie);const le={};n&2&&(le.$$scope={dirty:n,ctx:e}),mt.$set(le);const De={};n&2&&(De.$$scope={dirty:n,ctx:e}),ht.$set(De);const de={};n&2&&(de.$$scope={dirty:n,ctx:e}),ut.$set(de);const We={};n&2&&(We.$$scope={dirty:n,ctx:e}),ft.$set(We);const ce={};n&2&&(ce.$$scope={dirty:n,ctx:e}),gt.$set(ce);const Ze={};n&2&&(Ze.$$scope={dirty:n,ctx:e}),_t.$set(Ze);const pe={};n&2&&(pe.$$scope={dirty:n,ctx:e}),bt.$set(pe);const ze={};n&2&&(ze.$$scope={dirty:n,ctx:e}),yt.$set(ze);const me={};n&2&&(me.$$scope={dirty:n,ctx:e}),Tt.$set(me);const Ge={};n&2&&(Ge.$$scope={dirty:n,ctx:e}),Mt.$set(Ge);const he={};n&2&&(he.$$scope={dirty:n,ctx:e}),vt.$set(he);const Ie={};n&2&&(Ie.$$scope={dirty:n,ctx:e}),wt.$set(Ie)},i(e){$s||(_(kt.$$.fragment,e),_(Vt.$$.fragment,e),_(Ft.$$.fragment,e),_(Dt.$$.fragment,e),_(Gt.$$.fragment,e),_(Bt.$$.fragment,e),_(qt.$$.fragment,e),_(At.$$.fragment,e),_(Lt.$$.fragment,e),_(to.$$.fragment,e),_(oo.$$.fragment,e),_(Ye.$$.fragment,e),_(no.$$.fragment,e),_(ao.$$.fragment,e),_(Le.$$.fragment,e),_(so.$$.fragment,e),_(ro.$$.fragment,e),_(Se.$$.fragment,e),_(io.$$.fragment,e),_(lo.$$.fragment,e),_(co.$$.fragment,e),_(He.$$.fragment,e),_(po.$$.fragment,e),_(mo.$$.fragment,e),_(ho.$$.fragment,e),_(Ee.$$.fragment,e),_(Qe.$$.fragment,e),_(uo.$$.fragment,e),_(fo.$$.fragment,e),_(go.$$.fragment,e),_(Pe.$$.fragment,e),_(Oe.$$.fragment,e),_(_o.$$.fragment,e),_(bo.$$.fragment,e),_(yo.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(To.$$.fragment,e),_(Mo.$$.fragment,e),_(vo.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(wo.$$.fragment,e),_(ko.$$.fragment,e),_(Vo.$$.fragment,e),_(at.$$.fragment,e),_(xo.$$.fragment,e),_(Co.$$.fragment,e),_($o.$$.fragment,e),_(st.$$.fragment,e),_(rt.$$.fragment,e),_(jo.$$.fragment,e),_(Jo.$$.fragment,e),_(Fo.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(Uo.$$.fragment,e),_(Do.$$.fragment,e),_(Wo.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(Zo.$$.fragment,e),_(zo.$$.fragment,e),_(Go.$$.fragment,e),_(mt.$$.fragment,e),_(ht.$$.fragment,e),_(Io.$$.fragment,e),_(Ro.$$.fragment,e),_(Bo.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(No.$$.fragment,e),_(qo.$$.fragment,e),_(Xo.$$.fragment,e),_(gt.$$.fragment,e),_(_t.$$.fragment,e),_(Ao.$$.fragment,e),_(Yo.$$.fragment,e),_(Lo.$$.fragment,e),_(bt.$$.fragment,e),_(yt.$$.fragment,e),_(So.$$.fragment,e),_(Ho.$$.fragment,e),_(Eo.$$.fragment,e),_(Tt.$$.fragment,e),_(Mt.$$.fragment,e),_(Qo.$$.fragment,e),_(Po.$$.fragment,e),_(Oo.$$.fragment,e),_(vt.$$.fragment,e),_(wt.$$.fragment,e),_(Ko.$$.fragment,e),$s=!0)},o(e){b(kt.$$.fragment,e),b(Vt.$$.fragment,e),b(Ft.$$.fragment,e),b(Dt.$$.fragment,e),b(Gt.$$.fragment,e),b(Bt.$$.fragment,e),b(qt.$$.fragment,e),b(At.$$.fragment,e),b(Lt.$$.fragment,e),b(to.$$.fragment,e),b(oo.$$.fragment,e),b(Ye.$$.fragment,e),b(no.$$.fragment,e),b(ao.$$.fragment,e),b(Le.$$.fragment,e),b(so.$$.fragment,e),b(ro.$$.fragment,e),b(Se.$$.fragment,e),b(io.$$.fragment,e),b(lo.$$.fragment,e),b(co.$$.fragment,e),b(He.$$.fragment,e),b(po.$$.fragment,e),b(mo.$$.fragment,e),b(ho.$$.fragment,e),b(Ee.$$.fragment,e),b(Qe.$$.fragment,e),b(uo.$$.fragment,e),b(fo.$$.fragment,e),b(go.$$.fragment,e),b(Pe.$$.fragment,e),b(Oe.$$.fragment,e),b(_o.$$.fragment,e),b(bo.$$.fragment,e),b(yo.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(To.$$.fragment,e),b(Mo.$$.fragment,e),b(vo.$$.fragment,e),b(ot.$$.fragment,e),b(nt.$$.fragment,e),b(wo.$$.fragment,e),b(ko.$$.fragment,e),b(Vo.$$.fragment,e),b(at.$$.fragment,e),b(xo.$$.fragment,e),b(Co.$$.fragment,e),b($o.$$.fragment,e),b(st.$$.fragment,e),b(rt.$$.fragment,e),b(jo.$$.fragment,e),b(Jo.$$.fragment,e),b(Fo.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(Uo.$$.fragment,e),b(Do.$$.fragment,e),b(Wo.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(Zo.$$.fragment,e),b(zo.$$.fragment,e),b(Go.$$.fragment,e),b(mt.$$.fragment,e),b(ht.$$.fragment,e),b(Io.$$.fragment,e),b(Ro.$$.fragment,e),b(Bo.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(No.$$.fragment,e),b(qo.$$.fragment,e),b(Xo.$$.fragment,e),b(gt.$$.fragment,e),b(_t.$$.fragment,e),b(Ao.$$.fragment,e),b(Yo.$$.fragment,e),b(Lo.$$.fragment,e),b(bt.$$.fragment,e),b(yt.$$.fragment,e),b(So.$$.fragment,e),b(Ho.$$.fragment,e),b(Eo.$$.fragment,e),b(Tt.$$.fragment,e),b(Mt.$$.fragment,e),b(Qo.$$.fragment,e),b(Po.$$.fragment,e),b(Oo.$$.fragment,e),b(vt.$$.fragment,e),b(wt.$$.fragment,e),b(Ko.$$.fragment,e),$s=!1},d(e){e&&(a(T),a(l),a(d),a(M),a(v),a(ha),a(Ae),a(ua),a(fa),a(xt),a(ga),a(Ct),a(_a),a($t),a(ba),a(jt),a(ya),a(Jt),a(Ta),a(Ma),a(Ut),a(va),a(wa),a(Wt),a(ka),a(Zt),a(Va),a(zt),a(xa),a(Ca),a(It),a($a),a(Rt),a(ja),a(Ja),a(Nt),a(Fa),a(Ua),a(Xt),a(Da),a(Wa),a(Yt),a(Za),a(za),a(St),a(Ga),a(Ht),a(Ia),a(Et),a(Ra),a(Qt),a(Ba),a(Pt),a(Na),a(Ot),a(qa),a(Kt),a(Xa),a(eo),a(Aa),a(Ya),a(E),a(La),a(Sa),a(Q),a(Ha),a(Ea),a(Ve),a(Qa),a(Pa),a(F),a(Oa),a(Ka),a(U),a(es),a(ts),a(D),a(os),a(ns),a(W),a(as),a(ss),a(Z),a(rs),a(is),a(z),a(ls),a(ds),a(G),a(cs),a(ps),a(I),a(ms),a(hs),a(R),a(us),a(fs),a(B),a(gs),a(_s),a(N),a(bs),a(ys),a(q),a(Ts),a(Ms),a(X),a(vs),a(ws),a(A),a(ks),a(Vs),a(Y),a(xs),a(Cs),a(ma)),a(t),y(kt,e),y(Vt,e),y(Ft,e),y(Dt,e),y(Gt,e),y(Bt,e),y(qt,e),y(At,e),y(Lt,e),y(to,e),y(oo),y(Ye),y(no,e),y(ao),y(Le),y(so,e),y(ro),y(Se),y(io,e),y(lo),y(co),y(He),y(po,e),y(mo),y(ho),y(Ee),y(Qe),y(uo,e),y(fo),y(go),y(Pe),y(Oe),y(_o,e),y(bo),y(yo),y(Ke),y(et),y(tt),y(To,e),y(Mo),y(vo),y(ot),y(nt),y(wo,e),y(ko),y(Vo),y(at),y(xo,e),y(Co),y($o),y(st),y(rt),y(jo,e),y(Jo),y(Fo),y(it),y(lt),y(Uo,e),y(Do),y(Wo),y(dt),y(ct),y(pt),y(Zo,e),y(zo),y(Go),y(mt),y(ht),y(Io,e),y(Ro),y(Bo),y(ut),y(ft),y(No,e),y(qo),y(Xo),y(gt),y(_t),y(Ao,e),y(Yo),y(Lo),y(bt),y(yt),y(So,e),y(Ho),y(Eo),y(Tt),y(Mt),y(Qo,e),y(Po),y(Oo),y(vt),y(wt),y(Ko,e)}}}const tc='{"title":"Data2Vec","local":"data2vec","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[{"title":"Using Scaled Dot Product Attention (SDPA)","local":"using-scaled-dot-product-attention-sdpa","sections":[{"title":"Training","local":"training","sections":[],"depth":4},{"title":"Inference","local":"inference","sections":[],"depth":4}],"depth":3}],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"Data2VecTextConfig","local":"transformers.Data2VecTextConfig","sections":[],"depth":2},{"title":"Data2VecAudioConfig","local":"transformers.Data2VecAudioConfig","sections":[],"depth":2},{"title":"Data2VecVisionConfig","local":"transformers.Data2VecVisionConfig","sections":[],"depth":2},{"title":"Data2VecAudioModel","local":"transformers.Data2VecAudioModel","sections":[],"depth":2},{"title":"Data2VecAudioForAudioFrameClassification","local":"transformers.Data2VecAudioForAudioFrameClassification","sections":[],"depth":2},{"title":"Data2VecAudioForCTC","local":"transformers.Data2VecAudioForCTC","sections":[],"depth":2},{"title":"Data2VecAudioForSequenceClassification","local":"transformers.Data2VecAudioForSequenceClassification","sections":[],"depth":2},{"title":"Data2VecAudioForXVector","local":"transformers.Data2VecAudioForXVector","sections":[],"depth":2},{"title":"Data2VecTextModel","local":"transformers.Data2VecTextModel","sections":[],"depth":2},{"title":"Data2VecTextForCausalLM","local":"transformers.Data2VecTextForCausalLM","sections":[],"depth":2},{"title":"Data2VecTextForMaskedLM","local":"transformers.Data2VecTextForMaskedLM","sections":[],"depth":2},{"title":"Data2VecTextForSequenceClassification","local":"transformers.Data2VecTextForSequenceClassification","sections":[],"depth":2},{"title":"Data2VecTextForMultipleChoice","local":"transformers.Data2VecTextForMultipleChoice","sections":[],"depth":2},{"title":"Data2VecTextForTokenClassification","local":"transformers.Data2VecTextForTokenClassification","sections":[],"depth":2},{"title":"Data2VecTextForQuestionAnswering","local":"transformers.Data2VecTextForQuestionAnswering","sections":[],"depth":2},{"title":"Data2VecVisionModel","local":"transformers.Data2VecVisionModel","sections":[],"depth":2},{"title":"Data2VecVisionForImageClassification","local":"transformers.Data2VecVisionForImageClassification","sections":[],"depth":2},{"title":"Data2VecVisionForSemanticSegmentation","local":"transformers.Data2VecVisionForSemanticSegmentation","sections":[],"depth":2}],"depth":1}';function oc(w){return fd(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class pc extends gd{constructor(t){super(),_d(this,t,oc,ec,ud,{})}}export{pc as component};
