import{s as go,o as _o,n as j}from"../chunks/scheduler.18a86fab.js";import{S as yo,i as bo,g as m,s as a,r as f,A as To,h as u,f as s,c as r,j as J,x as v,u as g,k as I,l as vo,y as p,a as i,v as _,d as y,t as b,w as T}from"../chunks/index.98837b22.js";import{T as Ee}from"../chunks/Tip.77304350.js";import{D as V}from"../chunks/Docstring.a1ef7999.js";import{C as ze}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as fo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Pe,E as wo}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Mo,a as Et}from"../chunks/HfOption.6641485e.js";function ko(M){let t,d="Click on the Phi models in the right sidebar for more examples of how to apply Phi to different language tasks.";return{c(){t=m("p"),t.textContent=d},l(o){t=u(o,"P",{"data-svelte-h":!0}),v(t)!=="svelte-1nkysjl"&&(t.textContent=d)},m(o,l){i(o,t,l)},p:j,d(o){o&&s(t)}}}function $o(M){let t,d;return t=new ze({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMjBtb2RlbCUzRCUyMm1pY3Jvc29mdCUyRnBoaS0xLjUlMjIlMkMlMjBkZXZpY2UlM0QwJTJDJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiklMEFwaXBlbGluZSglMjJwaXBlbGluZSgnJydkZWYlMjBwcmludF9wcmltZShuKSUzQSUyMCUyMiUyMiUyMiUyMFByaW50JTIwYWxsJTIwcHJpbWVzJTIwYmV0d2VlbiUyMDElMjBhbmQlMjBuJTIyJTIyJTIyJycnKSUyMiklMEE=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(task=<span class="hljs-string">&quot;text-generation&quot;</span>, model=<span class="hljs-string">&quot;microsoft/phi-1.5&quot;</span>, device=<span class="hljs-number">0</span>, dtype=torch.bfloat16)
pipeline(<span class="hljs-string">&quot;pipeline(&#x27;&#x27;&#x27;def print_prime(n): &quot;</span><span class="hljs-string">&quot;&quot;</span> Print <span class="hljs-built_in">all</span> primes between <span class="hljs-number">1</span> <span class="hljs-keyword">and</span> n<span class="hljs-string">&quot;&quot;&quot;&#x27;&#x27;&#x27;)&quot;)
</span>`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,l){_(t,o,l),d=!0},p:j,i(o){d||(y(t.$$.fragment,o),d=!0)},o(o){b(t.$$.fragment,o),d=!1},d(o){T(t,o)}}}function Co(M){let t,d;return t=new ze({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcGhpLTElMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcGhpLTElMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcignJydkZWYlMjBwcmludF9wcmltZShuKSUzQSUwQSUyMCUyMCUyMCUyMiUyMiUyMiUwQSUyMCUyMCUyMFByaW50JTIwYWxsJTIwcHJpbWVzJTIwYmV0d2VlbiUyMDElMjBhbmQlMjBuJTBBJTIwJTIwJTIwJTIyJTIyJTIyJycnJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/phi-1&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;microsoft/phi-1&quot;</span>, dtype=torch.float16, device_map=<span class="hljs-string">&quot;auto&quot;</span>, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>)

input_ids = tokenizer(<span class="hljs-string">&#x27;&#x27;&#x27;def print_prime(n):
   &quot;&quot;&quot;
   Print all primes between 1 and n
   &quot;&quot;&quot;&#x27;&#x27;&#x27;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,l){_(t,o,l),d=!0},p:j,i(o){d||(y(t.$$.fragment,o),d=!0)},o(o){b(t.$$.fragment,o),d=!1},d(o){T(t,o)}}}function xo(M){let t,d;return t=new ze({props:{code:"ZWNobyUyMC1lJTIwJTIyJycnZGVmJTIwcHJpbnRfcHJpbWUobiklM0ElMjAlMjIlMjIlMjIlMjBQcmludCUyMGFsbCUyMHByaW1lcyUyMGJldHdlZW4lMjAxJTIwYW5kJTIwbiUyMiUyMiUyMicnJyUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQtY2xhc3NpZmljYXRpb24lMjAtLW1vZGVsJTIwbWljcm9zb2Z0JTJGcGhpLTEuNSUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;&#x27;&#x27;&#x27;def print_prime(n): &quot;</span><span class="hljs-string">&quot;&quot;</span> Print all primes between 1 and n<span class="hljs-string">&quot;&quot;</span><span class="hljs-string">&quot;&#x27;&#x27;&#x27;&quot;</span> | transformers run --task text-classification --model microsoft/phi-1.5 --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,l){_(t,o,l),d=!0},p:j,i(o){d||(y(t.$$.fragment,o),d=!0)},o(o){b(t.$$.fragment,o),d=!1},d(o){T(t,o)}}}function Jo(M){let t,d,o,l,w,c;return t=new Et({props:{id:"usage",option:"Pipeline",$$slots:{default:[$o]},$$scope:{ctx:M}}}),o=new Et({props:{id:"usage",option:"AutoModel",$$slots:{default:[Co]},$$scope:{ctx:M}}}),w=new Et({props:{id:"usage",option:"transformers CLI",$$slots:{default:[xo]},$$scope:{ctx:M}}}),{c(){f(t.$$.fragment),d=a(),f(o.$$.fragment),l=a(),f(w.$$.fragment)},l(h){g(t.$$.fragment,h),d=r(h),g(o.$$.fragment,h),l=r(h),g(w.$$.fragment,h)},m(h,k){_(t,h,k),i(h,d,k),_(o,h,k),i(h,l,k),_(w,h,k),c=!0},p(h,k){const Oe={};k&2&&(Oe.$$scope={dirty:k,ctx:h}),t.$set(Oe);const te={};k&2&&(te.$$scope={dirty:k,ctx:h}),o.$set(te);const W={};k&2&&(W.$$scope={dirty:k,ctx:h}),w.$set(W)},i(h){c||(y(t.$$.fragment,h),y(o.$$.fragment,h),y(w.$$.fragment,h),c=!0)},o(h){b(t.$$.fragment,h),b(o.$$.fragment,h),b(w.$$.fragment,h),c=!1},d(h){h&&(s(d),s(l)),T(t,h),T(o,h),T(w,h)}}}function Po(M){let t,d="Example:",o,l,w;return l=new ze({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBoaU1vZGVsJTJDJTIwUGhpQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMFBoaS0xJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFBoaUNvbmZpZy5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcGhpLTElMjIpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMFBoaU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PhiModel, PhiConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Phi-1 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = PhiConfig.from_pretrained(<span class="hljs-string">&quot;microsoft/phi-1&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PhiModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=m("p"),t.textContent=d,o=a(),f(l.$$.fragment)},l(c){t=u(c,"P",{"data-svelte-h":!0}),v(t)!=="svelte-11lpom8"&&(t.textContent=d),o=r(c),g(l.$$.fragment,c)},m(c,h){i(c,t,h),i(c,o,h),_(l,c,h),w=!0},p:j,i(c){w||(y(l.$$.fragment,c),w=!0)},o(c){b(l.$$.fragment,c),w=!1},d(c){c&&(s(t),s(o)),T(l,c)}}}function zo(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=d},l(o){t=u(o,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(o,l){i(o,t,l)},p:j,d(o){o&&s(t)}}}function Uo(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=d},l(o){t=u(o,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(o,l){i(o,t,l)},p:j,d(o){o&&s(t)}}}function Io(M){let t,d="Example:",o,l,w;return l=new ze({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQaGlGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwUGhpRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtcGhpJTJGUGhpLTItN2ItaGYlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1waGklMkZQaGktMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PhiForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = PhiForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-phi/Phi-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-phi/Phi-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=m("p"),t.textContent=d,o=a(),f(l.$$.fragment)},l(c){t=u(c,"P",{"data-svelte-h":!0}),v(t)!=="svelte-11lpom8"&&(t.textContent=d),o=r(c),g(l.$$.fragment,c)},m(c,h){i(c,t,h),i(c,o,h),_(l,c,h),w=!0},p:j,i(c){w||(y(l.$$.fragment,c),w=!0)},o(c){b(l.$$.fragment,c),w=!1},d(c){c&&(s(t),s(o)),T(l,c)}}}function Fo(M){let t,d=`Most generation-controlling parameters are set in <code>generation_config</code> which, if not passed, will be set to the
model’s default generation configuration. You can override any <code>generation_config</code> by passing the corresponding
parameters to generate(), e.g. <code>.generate(inputs, num_beams=4, do_sample=True)</code>.`,o,l,w=`For an overview of generation strategies and code examples, check out the <a href="../generation_strategies">following
guide</a>.`;return{c(){t=m("p"),t.innerHTML=d,o=a(),l=m("p"),l.innerHTML=w},l(c){t=u(c,"P",{"data-svelte-h":!0}),v(t)!=="svelte-1c5u34l"&&(t.innerHTML=d),o=r(c),l=u(c,"P",{"data-svelte-h":!0}),v(l)!=="svelte-fvlq1g"&&(l.innerHTML=w)},m(c,h){i(c,t,h),i(c,o,h),i(c,l,h)},p:j,d(c){c&&(s(t),s(o),s(l))}}}function qo(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=d},l(o){t=u(o,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(o,l){i(o,t,l)},p:j,d(o){o&&s(t)}}}function jo(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=d},l(o){t=u(o,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(o,l){i(o,t,l)},p:j,d(o){o&&s(t)}}}function Wo(M){let t,d,o,l,w,c="<em>This model was released on 2023-06-20 and added to Hugging Face Transformers on 2023-11-10.</em>",h,k,Oe='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',te,W,Ae,oe,Ot='<a href="https://huggingface.co/papers/2306.11644" rel="nofollow">Phi</a> is a 1.3B parameter transformer model optimized for Python code generation. It focuses on “textbook-quality” training data of code examples, exercises and synthetic Python problems rather than scaling the model size or compute.',Ke,ne,Dt='You can find all the original Phi checkpoints under the <a href="https://huggingface.co/collections/microsoft/phi-1-6626e29134744e94e222d572" rel="nofollow">Phi-1</a> collection.',et,N,tt,se,Yt='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> and from the command line.',ot,Q,nt,ae,At='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',st,re,Kt='The example below uses <a href="https://huggingface.co/docs/transformers/en/quantization/bitsandbytes" rel="nofollow">bitsandbytes</a> to only quantize the weights to 4-bits.',at,ie,rt,le,it,Ue,de,Ie,eo='If you’re using Transformers &lt; 4.37.0.dev, set <code>trust_remote_code=True</code> in <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained">from_pretrained()</a>. Otherwise, make sure you update Transformers to the latest stable version.',wt,ce,lt,pe,dt,P,he,Mt,Fe,to=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiModel">PhiModel</a>. It is used to instantiate an Phi
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Phi
<a href="https://huggingface.co/microsoft/phi-1" rel="nofollow">microsoft/phi-1</a>.`,kt,qe,oo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,$t,E,ct,me,pt,C,ue,Ct,je,no="The bare Phi Model outputting raw hidden-states without any specific head on top.",xt,We,so=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Jt,Le,ao=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Pt,L,fe,zt,Ge,ro='The <a href="/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiModel">PhiModel</a> forward method, overrides the <code>__call__</code> special method.',Ut,O,ht,ge,mt,$,_e,It,Ze,io="The Phi Model for causal language modeling.",Ft,Be,lo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,qt,Ve,co=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,jt,U,ye,Wt,Re,po='The <a href="/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiForCausalLM">PhiForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Lt,D,Gt,Y,Zt,G,be,Bt,Xe,ho="Generates sequences of token ids for models with a language modeling head.",Vt,A,ut,Te,ft,R,ve,Rt,Z,we,Xt,He,mo="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Ht,K,gt,Me,_t,X,ke,St,B,$e,Nt,Se,uo="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Qt,ee,yt,Ce,bt,De,Tt;return W=new Pe({props:{title:"Phi",local:"phi",headingTag:"h1"}}),N=new Ee({props:{warning:!1,$$slots:{default:[ko]},$$scope:{ctx:M}}}),Q=new Mo({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Jo]},$$scope:{ctx:M}}}),ie=new ze({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBYm5iX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyhsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTIwYm5iXzRiaXRfY29tcHV0ZV9kdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRnBoaS0xJTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRnBoaS0xJTIyJTJDJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUyQyUyMHF1YW50aXphdGlvbl9jb25maWclM0RibmJfY29uZmlnKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcignJydkZWYlMjBwcmludF9wcmltZShuKSUzQSUwQSUyMCUyMCUyMCUyMiUyMiUyMiUwQSUyMCUyMCUyMFByaW50JTIwYWxsJTIwcHJpbWVzJTIwYmV0d2VlbiUyMDElMjBhbmQlMjBuJTBBJTIwJTIwJTIwJTIyJTIyJTIyJycnJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(load_in_4bit=<span class="hljs-literal">True</span>, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>, bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/phi-1&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;microsoft/phi-1&quot;</span>, dtype=torch.float16, device_map=<span class="hljs-string">&quot;auto&quot;</span>, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, quantization_config=bnb_config)

input_ids = tokenizer(<span class="hljs-string">&#x27;&#x27;&#x27;def print_prime(n):
   &quot;&quot;&quot;
   Print all primes between 1 and n
   &quot;&quot;&quot;&#x27;&#x27;&#x27;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),le=new Pe({props:{title:"Notes",local:"notes",headingTag:"h2"}}),ce=new ze({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcGhpLTElMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybWljcm9zb2Z0JTJGcGhpLTElMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMHRydXN0X3JlbW90ZV9jb2RlJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcignJydkZWYlMjBwcmludF9wcmltZShuKSUzQSUwQSUyMCUyMCUyMCUyMiUyMiUyMiUwQSUyMCUyMCUyMFByaW50JTIwYWxsJTIwcHJpbWVzJTIwYmV0d2VlbiUyMDElMjBhbmQlMjBuJTBBJTIwJTIwJTIwJTIyJTIyJTIyJycnJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/phi-1&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;microsoft/phi-1&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    trust_remote_code=<span class="hljs-literal">True</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>)

input_ids = tokenizer(<span class="hljs-string">&#x27;&#x27;&#x27;def print_prime(n):
   &quot;&quot;&quot;
   Print all primes between 1 and n
   &quot;&quot;&quot;&#x27;&#x27;&#x27;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),pe=new Pe({props:{title:"PhiConfig",local:"transformers.PhiConfig",headingTag:"h2"}}),he=new V({props:{name:"class transformers.PhiConfig",anchor:"transformers.PhiConfig",parameters:[{name:"vocab_size",val:" = 51200"},{name:"hidden_size",val:" = 2048"},{name:"intermediate_size",val:" = 8192"},{name:"num_hidden_layers",val:" = 24"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"resid_pdrop",val:" = 0.0"},{name:"embd_pdrop",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"hidden_act",val:" = 'gelu_new'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"partial_rotary_factor",val:" = 0.5"},{name:"qk_layernorm",val:" = False"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PhiConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 51200) &#x2014;
Vocabulary size of the Phi model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiModel">PhiModel</a>.`,name:"vocab_size"},{anchor:"transformers.PhiConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.PhiConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.PhiConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.PhiConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.PhiConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.PhiConfig.resid_pdrop",description:`<strong>resid_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Dropout probability for mlp outputs.`,name:"resid_pdrop"},{anchor:"transformers.PhiConfig.embd_pdrop",description:`<strong>embd_pdrop</strong> (<code>int</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the embeddings.`,name:"embd_pdrop"},{anchor:"transformers.PhiConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio after computing the attention scores.`,name:"attention_dropout"},{anchor:"transformers.PhiConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu_new&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.PhiConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Phi-1 and Phi-1.5 supports up to 2048
tokens.`,name:"max_position_embeddings"},{anchor:"transformers.PhiConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.PhiConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.PhiConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>. Whether to tie weight embeddings or not.`,name:"use_cache"},{anchor:"transformers.PhiConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.PhiConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.PhiConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.PhiConfig.partial_rotary_factor",description:`<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Percentage of the query and keys which will have rotary embedding.`,name:"partial_rotary_factor"},{anchor:"transformers.PhiConfig.qk_layernorm",description:`<strong>qk_layernorm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to normalize the Queries and Keys after projecting the hidden states.`,name:"qk_layernorm"},{anchor:"transformers.PhiConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Denotes beginning of sequences token id.`,name:"bos_token_id"},{anchor:"transformers.PhiConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Denotes end of sequences token id.`,name:"eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi/configuration_phi.py#L26"}}),E=new fo({props:{anchor:"transformers.PhiConfig.example",$$slots:{default:[Po]},$$scope:{ctx:M}}}),me=new Pe({props:{title:"PhiModel",local:"transformers.PhiModel",headingTag:"h2"}}),ue=new V({props:{name:"class transformers.PhiModel",anchor:"transformers.PhiModel",parameters:[{name:"config",val:": PhiConfig"}],parametersDescription:[{anchor:"transformers.PhiModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiConfig">PhiConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi/modeling_phi.py#L315"}}),fe=new V({props:{name:"forward",anchor:"transformers.PhiModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.PhiModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PhiModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PhiModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PhiModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PhiModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PhiModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PhiModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PhiModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PhiModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi/modeling_phi.py#L333",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiConfig"
>PhiConfig</a>) and inputs.</p>
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
`}}),O=new Ee({props:{$$slots:{default:[zo]},$$scope:{ctx:M}}}),ge=new Pe({props:{title:"PhiForCausalLM",local:"transformers.PhiForCausalLM",headingTag:"h2"}}),_e=new V({props:{name:"class transformers.PhiForCausalLM",anchor:"transformers.PhiForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.PhiForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiForCausalLM">PhiForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi/modeling_phi.py#L433"}}),ye=new V({props:{name:"forward",anchor:"transformers.PhiForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.PhiForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PhiForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PhiForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PhiForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PhiForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PhiForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PhiForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PhiForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.PhiForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi/modeling_phi.py#L447",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/phi#transformers.PhiConfig"
>PhiConfig</a>) and inputs.</p>
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
`}}),D=new Ee({props:{$$slots:{default:[Uo]},$$scope:{ctx:M}}}),Y=new fo({props:{anchor:"transformers.PhiForCausalLM.forward.example",$$slots:{default:[Io]},$$scope:{ctx:M}}}),be=new V({props:{name:"generate",anchor:"transformers.PhiForCausalLM.generate",parameters:[{name:"inputs",val:": typing.Optional[torch.Tensor] = None"},{name:"generation_config",val:": typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None"},{name:"logits_processor",val:": typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None"},{name:"stopping_criteria",val:": typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None"},{name:"prefix_allowed_tokens_fn",val:": typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None"},{name:"synced_gpus",val:": typing.Optional[bool] = None"},{name:"assistant_model",val:": typing.Optional[ForwardRef('PreTrainedModel')] = None"},{name:"streamer",val:": typing.Optional[ForwardRef('BaseStreamer')] = None"},{name:"negative_prompt_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"negative_prompt_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"use_model_defaults",val:": typing.Optional[bool] = None"},{name:"custom_generate",val:": typing.Union[str, typing.Callable, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PhiForCausalLM.generate.inputs",description:`<strong>inputs</strong> (<code>torch.Tensor</code> of varying shape depending on the modality, <em>optional</em>) &#x2014;
The sequence used as a prompt for the generation or as model inputs to the encoder. If <code>None</code> the
method initializes it with <code>bos_token_id</code> and a batch size of 1. For decoder-only models <code>inputs</code>
should be in the format of <code>input_ids</code>. For encoder-decoder models <em>inputs</em> can represent any of
<code>input_ids</code>, <code>input_values</code>, <code>input_features</code>, or <code>pixel_values</code>.`,name:"inputs"},{anchor:"transformers.PhiForCausalLM.generate.generation_config",description:`<strong>generation_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>, <em>optional</em>) &#x2014;
The generation configuration to be used as base parametrization for the generation call. <code>**kwargs</code>
passed to generate matching the attributes of <code>generation_config</code> will override them. If
<code>generation_config</code> is not provided, the default will be used, which has the following loading
priority: 1) from the <code>generation_config.json</code> model file, if it exists; 2) from the model
configuration. Please note that unspecified parameters will inherit <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>&#x2019;s
default values, whose documentation should be checked to parameterize generation.`,name:"generation_config"},{anchor:"transformers.PhiForCausalLM.generate.logits_processor",description:`<strong>logits_processor</strong> (<code>LogitsProcessorList</code>, <em>optional</em>) &#x2014;
Custom logits processors that complement the default logits processors built from arguments and
generation config. If a logit processor is passed that is already created with the arguments or a
generation config an error is thrown. This feature is intended for advanced users.`,name:"logits_processor"},{anchor:"transformers.PhiForCausalLM.generate.stopping_criteria",description:`<strong>stopping_criteria</strong> (<code>StoppingCriteriaList</code>, <em>optional</em>) &#x2014;
Custom stopping criteria that complements the default stopping criteria built from arguments and a
generation config. If a stopping criteria is passed that is already created with the arguments or a
generation config an error is thrown. If your stopping criteria depends on the <code>scores</code> input, make
sure you pass <code>return_dict_in_generate=True, output_scores=True</code> to <code>generate</code>. This feature is
intended for advanced users.`,name:"stopping_criteria"},{anchor:"transformers.PhiForCausalLM.generate.prefix_allowed_tokens_fn",description:`<strong>prefix_allowed_tokens_fn</strong> (<code>Callable[[int, torch.Tensor], list[int]]</code>, <em>optional</em>) &#x2014;
If provided, this function constraints the beam search to allowed tokens only at each step. If not
provided no constraint is applied. This function takes 2 arguments: the batch ID <code>batch_id</code> and
<code>input_ids</code>. It has to return a list with the allowed tokens for the next generation step conditioned
on the batch ID <code>batch_id</code> and the previously generated tokens <code>inputs_ids</code>. This argument is useful
for constrained generation conditioned on the prefix, as described in <a href="https://huggingface.co/papers/2010.00904" rel="nofollow">Autoregressive Entity
Retrieval</a>.`,name:"prefix_allowed_tokens_fn"},{anchor:"transformers.PhiForCausalLM.generate.synced_gpus",description:`<strong>synced_gpus</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
to <code>True</code> if using <code>FullyShardedDataParallel</code> or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to <code>False</code>.`,name:"synced_gpus"},{anchor:"transformers.PhiForCausalLM.generate.assistant_model",description:`<strong>assistant_model</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
An assistant model that can be used to accelerate generation. The assistant model must have the exact
same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
is much faster than running generation with the model you&#x2019;re calling generate from. As such, the
assistant model should be much smaller.`,name:"assistant_model"},{anchor:"transformers.PhiForCausalLM.generate.streamer",description:`<strong>streamer</strong> (<code>BaseStreamer</code>, <em>optional</em>) &#x2014;
Streamer object that will be used to stream the generated sequences. Generated tokens are passed
through <code>streamer.put(token_ids)</code> and the streamer is responsible for any further processing.`,name:"streamer"},{anchor:"transformers.PhiForCausalLM.generate.negative_prompt_ids",description:`<strong>negative_prompt_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
The negative prompt needed for some processors such as CFG. The batch size must match the input batch
size. This is an experimental feature, subject to breaking API changes in future versions.`,name:"negative_prompt_ids"},{anchor:"transformers.PhiForCausalLM.generate.negative_prompt_attention_mask",description:`<strong>negative_prompt_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Attention_mask for <code>negative_prompt_ids</code>.`,name:"negative_prompt_attention_mask"},{anchor:"transformers.PhiForCausalLM.generate.use_model_defaults",description:`<strong>use_model_defaults</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
When it is <code>True</code>, unset parameters in <code>generation_config</code> will be set to the model-specific default
generation configuration (<code>model.generation_config</code>), as opposed to the global defaults
(<code>GenerationConfig()</code>). If unset, models saved starting from <code>v4.50</code> will consider this flag to be
<code>True</code>.`,name:"use_model_defaults"},{anchor:"transformers.PhiForCausalLM.generate.custom_generate",description:`<strong>custom_generate</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>) &#x2014;
One of the following:<ul>
<li><code>str</code> (Hugging Face Hub repository name): runs the custom <code>generate</code> function defined at
<code>custom_generate/generate.py</code> in that repository instead of the standard <code>generate</code> method. The
repository fully replaces the generation logic, and the return type may differ.</li>
<li><code>str</code> (local repository path): same as above but from a local path, <code>trust_remote_code</code> not required.</li>
<li><code>Callable</code>: <code>generate</code> will perform the usual input preparation steps, then call the provided callable to
run the decoding loop.
For more information, see <a href="../../generation_strategies#custom-generation-methods">the docs</a>.</li>
</ul>`,name:"custom_generate"},{anchor:"transformers.PhiForCausalLM.generate.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
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
`}}),A=new Ee({props:{warning:!0,$$slots:{default:[Fo]},$$scope:{ctx:M}}}),Te=new Pe({props:{title:"PhiForSequenceClassification",local:"transformers.PhiForSequenceClassification",headingTag:"h2"}}),ve=new V({props:{name:"class transformers.PhiForSequenceClassification",anchor:"transformers.PhiForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi/modeling_phi.py#L508"}}),we=new V({props:{name:"forward",anchor:"transformers.PhiForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.PhiForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PhiForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PhiForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PhiForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PhiForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PhiForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PhiForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L111",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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


<p><code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),K=new Ee({props:{$$slots:{default:[qo]},$$scope:{ctx:M}}}),Me=new Pe({props:{title:"PhiForTokenClassification",local:"transformers.PhiForTokenClassification",headingTag:"h2"}}),ke=new V({props:{name:"class transformers.PhiForTokenClassification",anchor:"transformers.PhiForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi/modeling_phi.py#L512"}}),$e=new V({props:{name:"forward",anchor:"transformers.PhiForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PhiForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PhiForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PhiForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PhiForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PhiForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PhiForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PhiForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L254",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
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
`}}),ee=new Ee({props:{$$slots:{default:[jo]},$$scope:{ctx:M}}}),Ce=new wo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/phi.md"}}),{c(){t=m("meta"),d=a(),o=m("p"),l=a(),w=m("p"),w.innerHTML=c,h=a(),k=m("div"),k.innerHTML=Oe,te=a(),f(W.$$.fragment),Ae=a(),oe=m("p"),oe.innerHTML=Ot,Ke=a(),ne=m("p"),ne.innerHTML=Dt,et=a(),f(N.$$.fragment),tt=a(),se=m("p"),se.innerHTML=Yt,ot=a(),f(Q.$$.fragment),nt=a(),ae=m("p"),ae.innerHTML=At,st=a(),re=m("p"),re.innerHTML=Kt,at=a(),f(ie.$$.fragment),rt=a(),f(le.$$.fragment),it=a(),Ue=m("ul"),de=m("li"),Ie=m("p"),Ie.innerHTML=eo,wt=a(),f(ce.$$.fragment),lt=a(),f(pe.$$.fragment),dt=a(),P=m("div"),f(he.$$.fragment),Mt=a(),Fe=m("p"),Fe.innerHTML=to,kt=a(),qe=m("p"),qe.innerHTML=oo,$t=a(),f(E.$$.fragment),ct=a(),f(me.$$.fragment),pt=a(),C=m("div"),f(ue.$$.fragment),Ct=a(),je=m("p"),je.textContent=no,xt=a(),We=m("p"),We.innerHTML=so,Jt=a(),Le=m("p"),Le.innerHTML=ao,Pt=a(),L=m("div"),f(fe.$$.fragment),zt=a(),Ge=m("p"),Ge.innerHTML=ro,Ut=a(),f(O.$$.fragment),ht=a(),f(ge.$$.fragment),mt=a(),$=m("div"),f(_e.$$.fragment),It=a(),Ze=m("p"),Ze.textContent=io,Ft=a(),Be=m("p"),Be.innerHTML=lo,qt=a(),Ve=m("p"),Ve.innerHTML=co,jt=a(),U=m("div"),f(ye.$$.fragment),Wt=a(),Re=m("p"),Re.innerHTML=po,Lt=a(),f(D.$$.fragment),Gt=a(),f(Y.$$.fragment),Zt=a(),G=m("div"),f(be.$$.fragment),Bt=a(),Xe=m("p"),Xe.textContent=ho,Vt=a(),f(A.$$.fragment),ut=a(),f(Te.$$.fragment),ft=a(),R=m("div"),f(ve.$$.fragment),Rt=a(),Z=m("div"),f(we.$$.fragment),Xt=a(),He=m("p"),He.innerHTML=mo,Ht=a(),f(K.$$.fragment),gt=a(),f(Me.$$.fragment),_t=a(),X=m("div"),f(ke.$$.fragment),St=a(),B=m("div"),f($e.$$.fragment),Nt=a(),Se=m("p"),Se.innerHTML=uo,Qt=a(),f(ee.$$.fragment),yt=a(),f(Ce.$$.fragment),bt=a(),De=m("p"),this.h()},l(e){const n=To("svelte-u9bgzb",document.head);t=u(n,"META",{name:!0,content:!0}),n.forEach(s),d=r(e),o=u(e,"P",{}),J(o).forEach(s),l=r(e),w=u(e,"P",{"data-svelte-h":!0}),v(w)!=="svelte-1vc8841"&&(w.innerHTML=c),h=r(e),k=u(e,"DIV",{style:!0,"data-svelte-h":!0}),v(k)!=="svelte-11gpmgv"&&(k.innerHTML=Oe),te=r(e),g(W.$$.fragment,e),Ae=r(e),oe=u(e,"P",{"data-svelte-h":!0}),v(oe)!=="svelte-o7bff2"&&(oe.innerHTML=Ot),Ke=r(e),ne=u(e,"P",{"data-svelte-h":!0}),v(ne)!=="svelte-ydwxi7"&&(ne.innerHTML=Dt),et=r(e),g(N.$$.fragment,e),tt=r(e),se=u(e,"P",{"data-svelte-h":!0}),v(se)!=="svelte-s90gxn"&&(se.innerHTML=Yt),ot=r(e),g(Q.$$.fragment,e),nt=r(e),ae=u(e,"P",{"data-svelte-h":!0}),v(ae)!=="svelte-nf5ooi"&&(ae.innerHTML=At),st=r(e),re=u(e,"P",{"data-svelte-h":!0}),v(re)!=="svelte-8j3ubg"&&(re.innerHTML=Kt),at=r(e),g(ie.$$.fragment,e),rt=r(e),g(le.$$.fragment,e),it=r(e),Ue=u(e,"UL",{});var Ye=J(Ue);de=u(Ye,"LI",{});var xe=J(de);Ie=u(xe,"P",{"data-svelte-h":!0}),v(Ie)!=="svelte-gl849f"&&(Ie.innerHTML=eo),wt=r(xe),g(ce.$$.fragment,xe),xe.forEach(s),Ye.forEach(s),lt=r(e),g(pe.$$.fragment,e),dt=r(e),P=u(e,"DIV",{class:!0});var F=J(P);g(he.$$.fragment,F),Mt=r(F),Fe=u(F,"P",{"data-svelte-h":!0}),v(Fe)!=="svelte-1vf2dlh"&&(Fe.innerHTML=to),kt=r(F),qe=u(F,"P",{"data-svelte-h":!0}),v(qe)!=="svelte-1ek1ss9"&&(qe.innerHTML=oo),$t=r(F),g(E.$$.fragment,F),F.forEach(s),ct=r(e),g(me.$$.fragment,e),pt=r(e),C=u(e,"DIV",{class:!0});var z=J(C);g(ue.$$.fragment,z),Ct=r(z),je=u(z,"P",{"data-svelte-h":!0}),v(je)!=="svelte-1apf5a7"&&(je.textContent=no),xt=r(z),We=u(z,"P",{"data-svelte-h":!0}),v(We)!=="svelte-q52n56"&&(We.innerHTML=so),Jt=r(z),Le=u(z,"P",{"data-svelte-h":!0}),v(Le)!=="svelte-hswkmf"&&(Le.innerHTML=ao),Pt=r(z),L=u(z,"DIV",{class:!0});var H=J(L);g(fe.$$.fragment,H),zt=r(H),Ge=u(H,"P",{"data-svelte-h":!0}),v(Ge)!=="svelte-40tagy"&&(Ge.innerHTML=ro),Ut=r(H),g(O.$$.fragment,H),H.forEach(s),z.forEach(s),ht=r(e),g(ge.$$.fragment,e),mt=r(e),$=u(e,"DIV",{class:!0});var x=J($);g(_e.$$.fragment,x),It=r(x),Ze=u(x,"P",{"data-svelte-h":!0}),v(Ze)!=="svelte-1c4comc"&&(Ze.textContent=io),Ft=r(x),Be=u(x,"P",{"data-svelte-h":!0}),v(Be)!=="svelte-q52n56"&&(Be.innerHTML=lo),qt=r(x),Ve=u(x,"P",{"data-svelte-h":!0}),v(Ve)!=="svelte-hswkmf"&&(Ve.innerHTML=co),jt=r(x),U=u(x,"DIV",{class:!0});var q=J(U);g(ye.$$.fragment,q),Wt=r(q),Re=u(q,"P",{"data-svelte-h":!0}),v(Re)!=="svelte-1gbje7e"&&(Re.innerHTML=po),Lt=r(q),g(D.$$.fragment,q),Gt=r(q),g(Y.$$.fragment,q),q.forEach(s),Zt=r(x),G=u(x,"DIV",{class:!0});var S=J(G);g(be.$$.fragment,S),Bt=r(S),Xe=u(S,"P",{"data-svelte-h":!0}),v(Xe)!=="svelte-s5ko3x"&&(Xe.textContent=ho),Vt=r(S),g(A.$$.fragment,S),S.forEach(s),x.forEach(s),ut=r(e),g(Te.$$.fragment,e),ft=r(e),R=u(e,"DIV",{class:!0});var Je=J(R);g(ve.$$.fragment,Je),Rt=r(Je),Z=u(Je,"DIV",{class:!0});var Ne=J(Z);g(we.$$.fragment,Ne),Xt=r(Ne),He=u(Ne,"P",{"data-svelte-h":!0}),v(He)!=="svelte-1sal4ui"&&(He.innerHTML=mo),Ht=r(Ne),g(K.$$.fragment,Ne),Ne.forEach(s),Je.forEach(s),gt=r(e),g(Me.$$.fragment,e),_t=r(e),X=u(e,"DIV",{class:!0});var vt=J(X);g(ke.$$.fragment,vt),St=r(vt),B=u(vt,"DIV",{class:!0});var Qe=J(B);g($e.$$.fragment,Qe),Nt=r(Qe),Se=u(Qe,"P",{"data-svelte-h":!0}),v(Se)!=="svelte-1py4aay"&&(Se.innerHTML=uo),Qt=r(Qe),g(ee.$$.fragment,Qe),Qe.forEach(s),vt.forEach(s),yt=r(e),g(Ce.$$.fragment,e),bt=r(e),De=u(e,"P",{}),J(De).forEach(s),this.h()},h(){I(t,"name","hf:doc:metadata"),I(t,"content",Lo),vo(k,"float","right"),I(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){p(document.head,t),i(e,d,n),i(e,o,n),i(e,l,n),i(e,w,n),i(e,h,n),i(e,k,n),i(e,te,n),_(W,e,n),i(e,Ae,n),i(e,oe,n),i(e,Ke,n),i(e,ne,n),i(e,et,n),_(N,e,n),i(e,tt,n),i(e,se,n),i(e,ot,n),_(Q,e,n),i(e,nt,n),i(e,ae,n),i(e,st,n),i(e,re,n),i(e,at,n),_(ie,e,n),i(e,rt,n),_(le,e,n),i(e,it,n),i(e,Ue,n),p(Ue,de),p(de,Ie),p(de,wt),_(ce,de,null),i(e,lt,n),_(pe,e,n),i(e,dt,n),i(e,P,n),_(he,P,null),p(P,Mt),p(P,Fe),p(P,kt),p(P,qe),p(P,$t),_(E,P,null),i(e,ct,n),_(me,e,n),i(e,pt,n),i(e,C,n),_(ue,C,null),p(C,Ct),p(C,je),p(C,xt),p(C,We),p(C,Jt),p(C,Le),p(C,Pt),p(C,L),_(fe,L,null),p(L,zt),p(L,Ge),p(L,Ut),_(O,L,null),i(e,ht,n),_(ge,e,n),i(e,mt,n),i(e,$,n),_(_e,$,null),p($,It),p($,Ze),p($,Ft),p($,Be),p($,qt),p($,Ve),p($,jt),p($,U),_(ye,U,null),p(U,Wt),p(U,Re),p(U,Lt),_(D,U,null),p(U,Gt),_(Y,U,null),p($,Zt),p($,G),_(be,G,null),p(G,Bt),p(G,Xe),p(G,Vt),_(A,G,null),i(e,ut,n),_(Te,e,n),i(e,ft,n),i(e,R,n),_(ve,R,null),p(R,Rt),p(R,Z),_(we,Z,null),p(Z,Xt),p(Z,He),p(Z,Ht),_(K,Z,null),i(e,gt,n),_(Me,e,n),i(e,_t,n),i(e,X,n),_(ke,X,null),p(X,St),p(X,B),_($e,B,null),p(B,Nt),p(B,Se),p(B,Qt),_(ee,B,null),i(e,yt,n),_(Ce,e,n),i(e,bt,n),i(e,De,n),Tt=!0},p(e,[n]){const Ye={};n&2&&(Ye.$$scope={dirty:n,ctx:e}),N.$set(Ye);const xe={};n&2&&(xe.$$scope={dirty:n,ctx:e}),Q.$set(xe);const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),E.$set(F);const z={};n&2&&(z.$$scope={dirty:n,ctx:e}),O.$set(z);const H={};n&2&&(H.$$scope={dirty:n,ctx:e}),D.$set(H);const x={};n&2&&(x.$$scope={dirty:n,ctx:e}),Y.$set(x);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),A.$set(q);const S={};n&2&&(S.$$scope={dirty:n,ctx:e}),K.$set(S);const Je={};n&2&&(Je.$$scope={dirty:n,ctx:e}),ee.$set(Je)},i(e){Tt||(y(W.$$.fragment,e),y(N.$$.fragment,e),y(Q.$$.fragment,e),y(ie.$$.fragment,e),y(le.$$.fragment,e),y(ce.$$.fragment,e),y(pe.$$.fragment,e),y(he.$$.fragment,e),y(E.$$.fragment,e),y(me.$$.fragment,e),y(ue.$$.fragment,e),y(fe.$$.fragment,e),y(O.$$.fragment,e),y(ge.$$.fragment,e),y(_e.$$.fragment,e),y(ye.$$.fragment,e),y(D.$$.fragment,e),y(Y.$$.fragment,e),y(be.$$.fragment,e),y(A.$$.fragment,e),y(Te.$$.fragment,e),y(ve.$$.fragment,e),y(we.$$.fragment,e),y(K.$$.fragment,e),y(Me.$$.fragment,e),y(ke.$$.fragment,e),y($e.$$.fragment,e),y(ee.$$.fragment,e),y(Ce.$$.fragment,e),Tt=!0)},o(e){b(W.$$.fragment,e),b(N.$$.fragment,e),b(Q.$$.fragment,e),b(ie.$$.fragment,e),b(le.$$.fragment,e),b(ce.$$.fragment,e),b(pe.$$.fragment,e),b(he.$$.fragment,e),b(E.$$.fragment,e),b(me.$$.fragment,e),b(ue.$$.fragment,e),b(fe.$$.fragment,e),b(O.$$.fragment,e),b(ge.$$.fragment,e),b(_e.$$.fragment,e),b(ye.$$.fragment,e),b(D.$$.fragment,e),b(Y.$$.fragment,e),b(be.$$.fragment,e),b(A.$$.fragment,e),b(Te.$$.fragment,e),b(ve.$$.fragment,e),b(we.$$.fragment,e),b(K.$$.fragment,e),b(Me.$$.fragment,e),b(ke.$$.fragment,e),b($e.$$.fragment,e),b(ee.$$.fragment,e),b(Ce.$$.fragment,e),Tt=!1},d(e){e&&(s(d),s(o),s(l),s(w),s(h),s(k),s(te),s(Ae),s(oe),s(Ke),s(ne),s(et),s(tt),s(se),s(ot),s(nt),s(ae),s(st),s(re),s(at),s(rt),s(it),s(Ue),s(lt),s(dt),s(P),s(ct),s(pt),s(C),s(ht),s(mt),s($),s(ut),s(ft),s(R),s(gt),s(_t),s(X),s(yt),s(bt),s(De)),s(t),T(W,e),T(N,e),T(Q,e),T(ie,e),T(le,e),T(ce),T(pe,e),T(he),T(E),T(me,e),T(ue),T(fe),T(O),T(ge,e),T(_e),T(ye),T(D),T(Y),T(be),T(A),T(Te,e),T(ve),T(we),T(K),T(Me,e),T(ke),T($e),T(ee),T(Ce,e)}}}const Lo='{"title":"Phi","local":"phi","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"PhiConfig","local":"transformers.PhiConfig","sections":[],"depth":2},{"title":"PhiModel","local":"transformers.PhiModel","sections":[],"depth":2},{"title":"PhiForCausalLM","local":"transformers.PhiForCausalLM","sections":[],"depth":2},{"title":"PhiForSequenceClassification","local":"transformers.PhiForSequenceClassification","sections":[],"depth":2},{"title":"PhiForTokenClassification","local":"transformers.PhiForTokenClassification","sections":[],"depth":2}],"depth":1}';function Go(M){return _o(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Qo extends yo{constructor(t){super(),bo(this,t,Go,Wo,go,{})}}export{Qo as component};
