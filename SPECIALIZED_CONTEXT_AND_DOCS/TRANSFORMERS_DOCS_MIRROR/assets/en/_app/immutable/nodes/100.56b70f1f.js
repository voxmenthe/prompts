import{s as Do,o as Yo,n as Le}from"../chunks/scheduler.18a86fab.js";import{S as Ao,i as Qo,g as i,s as n,r as h,A as Ko,h as c,f as o,c as s,j as C,x as k,u,k as $,y as r,a as d,v as f,d as _,t as g,w as b}from"../chunks/index.98837b22.js";import{T as yo}from"../chunks/Tip.77304350.js";import{D as z}from"../chunks/Docstring.a1ef7999.js";import{C as Gt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ut}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as W,E as en}from"../chunks/getInferenceSnippets.06c2775f.js";function tn(B){let a,T="Example:",m,p,y;return p=new Gt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJsZW5kZXJib3RTbWFsbENvbmZpZyUyQyUyMEJsZW5kZXJib3RTbWFsbE1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEJsZW5kZXJib3RTbWFsbCUyMGZhY2Vib29rJTJGYmxlbmRlcmJvdF9zbWFsbC05ME0lMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwQmxlbmRlcmJvdFNtYWxsQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGZhY2Vib29rJTJGYmxlbmRlcmJvdF9zbWFsbC05ME0lMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEJsZW5kZXJib3RTbWFsbE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotSmallConfig, BlenderbotSmallModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a BlenderbotSmall facebook/blenderbot_small-90M style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BlenderbotSmallConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the facebook/blenderbot_small-90M style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotSmallModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){a=i("p"),a.textContent=T,m=n(),h(p.$$.fragment)},l(l){a=c(l,"P",{"data-svelte-h":!0}),k(a)!=="svelte-11lpom8"&&(a.textContent=T),m=s(l),u(p.$$.fragment,l)},m(l,v){d(l,a,v),d(l,m,v),f(p,l,v),y=!0},p:Le,i(l){y||(_(p.$$.fragment,l),y=!0)},o(l){g(p.$$.fragment,l),y=!1},d(l){l&&(o(a),o(m)),b(p,l)}}}function on(B){let a,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=i("p"),a.innerHTML=T},l(m){a=c(m,"P",{"data-svelte-h":!0}),k(a)!=="svelte-fincs2"&&(a.innerHTML=T)},m(m,p){d(m,a,p)},p:Le,d(m){m&&o(a)}}}function nn(B){let a,T="Example:",m,p,y;return p=new Gt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCbGVuZGVyYm90U21hbGxNb2RlbCUwQSUwQW1vZGVsJTIwJTNEJTIwQmxlbmRlcmJvdFNtYWxsTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGYmxlbmRlcmJvdF9zbWFsbC05ME0lMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZibGVuZGVyYm90X3NtYWxsLTkwTSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyU3R1ZGllcyUyMGhhdmUlMjBiZWVuJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBZGVjb2Rlcl9pbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyU3R1ZGllcyUyMHNob3clMjB0aGF0JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUzRGlucHV0cy5pbnB1dF9pZHMlMkMlMjBkZWNvZGVyX2lucHV0X2lkcyUzRGRlY29kZXJfaW5wdXRzLmlucHV0X2lkcyklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBbGlzdChsYXN0X2hpZGRlbl9zdGF0ZXMuc2hhcGUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BlenderbotSmallModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotSmallModel.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_inputs = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">3</span>, <span class="hljs-number">512</span>]`,wrap:!1}}),{c(){a=i("p"),a.textContent=T,m=n(),h(p.$$.fragment)},l(l){a=c(l,"P",{"data-svelte-h":!0}),k(a)!=="svelte-11lpom8"&&(a.textContent=T),m=s(l),u(p.$$.fragment,l)},m(l,v){d(l,a,v),d(l,m,v),f(p,l,v),y=!0},p:Le,i(l){y||(_(p.$$.fragment,l),y=!0)},o(l){g(p.$$.fragment,l),y=!1},d(l){l&&(o(a),o(m)),b(p,l)}}}function sn(B){let a,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=i("p"),a.innerHTML=T},l(m){a=c(m,"P",{"data-svelte-h":!0}),k(a)!=="svelte-fincs2"&&(a.innerHTML=T)},m(m,p){d(m,a,p)},p:Le,d(m){m&&o(a)}}}function an(B){let a,T="Example Conversation:",m,p,y;return p=new Gt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCbGVuZGVyYm90U21hbGxGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEFtbmFtZSUyMCUzRCUyMCUyMmZhY2Vib29rJTJGYmxlbmRlcmJvdF9zbWFsbC05ME0lMjIlMEFtb2RlbCUyMCUzRCUyMEJsZW5kZXJib3RTbWFsbEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW5hbWUpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQobW5hbWUpJTBBVVRURVJBTkNFJTIwJTNEJTIwJTIyTXklMjBmcmllbmRzJTIwYXJlJTIwY29vbCUyMGJ1dCUyMHRoZXklMjBlYXQlMjB0b28lMjBtYW55JTIwY2FyYnMuJTIyJTBBcHJpbnQoJTIySHVtYW4lM0ElMjAlMjIlMkMlMjBVVFRFUkFOQ0UpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCU1QlVUVEVSQU5DRSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBcmVwbHlfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMpJTBBcHJpbnQoJTIyQm90JTNBJTIwJTIyJTJDJTIwdG9rZW5pemVyLmJhdGNoX2RlY29kZShyZXBseV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklNUIwJTVEKSUwQSUwQVJFUExZJTIwJTNEJTIwJTIySSdtJTIwbm90JTIwc3VyZSUyMiUwQXByaW50KCUyMkh1bWFuJTNBJTIwJTIyJTJDJTIwUkVQTFkpJTBBJTBBTkVYVF9VVFRFUkFOQ0UlMjAlM0QlMjAoJTBBJTIwJTIwJTIwJTIwJTIyTXklMjBmcmllbmRzJTIwYXJlJTIwY29vbCUyMGJ1dCUyMHRoZXklMjBlYXQlMjB0b28lMjBtYW55JTIwY2FyYnMuX19lbmRfXyUyMF9fc3RhcnRfX3doYXQlMjBraW5kJTIwb2YlMjBjYXJicyUyMGRvJTIwdGhleSUyMGVhdCUzRiUyMCUyMiUwQSUyMCUyMCUyMCUyMCUyMmklMjBkb24ndCUyMGtub3clMjBtdWNoJTIwYWJvdXQlMjBjYXJic19fZW5kX18lMjAlMjIlMEElMjAlMjAlMjAlMjAlMjJfX3N0YXJ0X18lMjBJJ20lMjBub3QlMjBzdXJlLiUyMiUwQSklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTVCTkVYVF9VVFRFUkFOQ0UlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW5leHRfcmVwbHlfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMpJTBBcHJpbnQoJTIyQm90JTNBJTIwJTIyJTJDJTIwdG9rZW5pemVyLmJhdGNoX2RlY29kZShuZXh0X3JlcGx5X2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BlenderbotSmallForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>mname = <span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>UTTERANCE = <span class="hljs-string">&quot;My friends are cool but they eat too many carbs.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Human: &quot;</span>, UTTERANCE)
Human:  My friends are cool but they eat too many carbs.

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([UTTERANCE], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>reply_ids = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Bot: &quot;</span>, tokenizer.batch_decode(reply_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>])
Bot:  what kind of carbs do they eat? i don<span class="hljs-string">&#x27;t know much about carbs.

&gt;&gt;&gt; REPLY = &quot;I&#x27;</span>m <span class="hljs-keyword">not</span> sure<span class="hljs-string">&quot;
&gt;&gt;&gt; print(&quot;</span>Human: <span class="hljs-string">&quot;, REPLY)
Human: I&#x27;m not sure

&gt;&gt;&gt; NEXT_UTTERANCE = (
...     &quot;</span>My friends are cool but they eat too many carbs.__end__ __start__what kind of carbs do they eat? <span class="hljs-string">&quot;
...     &quot;</span>i don<span class="hljs-string">&#x27;t know much about carbs__end__ &quot;
...     &quot;__start__ I&#x27;</span>m <span class="hljs-keyword">not</span> sure.<span class="hljs-string">&quot;
... )
&gt;&gt;&gt; inputs = tokenizer([NEXT_UTTERANCE], return_tensors=&quot;</span>pt<span class="hljs-string">&quot;)
&gt;&gt;&gt; next_reply_ids = model.generate(**inputs)
&gt;&gt;&gt; print(&quot;</span>Bot: <span class="hljs-string">&quot;, tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
Bot:  they eat a lot of carbs. carbs are high in fat, protein, and fats.</span>`,wrap:!1}}),{c(){a=i("p"),a.textContent=T,m=n(),h(p.$$.fragment)},l(l){a=c(l,"P",{"data-svelte-h":!0}),k(a)!=="svelte-ileb1l"&&(a.textContent=T),m=s(l),u(p.$$.fragment,l)},m(l,v){d(l,a,v),d(l,m,v),f(p,l,v),y=!0},p:Le,i(l){y||(_(p.$$.fragment,l),y=!0)},o(l){g(p.$$.fragment,l),y=!1},d(l){l&&(o(a),o(m)),b(p,l)}}}function rn(B){let a,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=i("p"),a.innerHTML=T},l(m){a=c(m,"P",{"data-svelte-h":!0}),k(a)!=="svelte-fincs2"&&(a.innerHTML=T)},m(m,p){d(m,a,p)},p:Le,d(m){m&&o(a)}}}function ln(B){let a,T="Example:",m,p,y;return p=new Gt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCbGVuZGVyYm90U21hbGxGb3JDYXVzYWxMTSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGYmxlbmRlcmJvdF9zbWFsbC05ME0lMjIpJTBBbW9kZWwlMjAlM0QlMjBCbGVuZGVyYm90U21hbGxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZibGVuZGVyYm90X3NtYWxsLTkwTSUyMiUyQyUyMGFkZF9jcm9zc19hdHRlbnRpb24lM0RGYWxzZSklMEFhc3NlcnQlMjBtb2RlbC5jb25maWcuaXNfZGVjb2RlciUyQyUyMGYlMjIlN0Jtb2RlbC5fX2NsYXNzX18lN0QlMjBoYXMlMjB0byUyMGJlJTIwY29uZmlndXJlZCUyMGFzJTIwYSUyMGRlY29kZXIuJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQWV4cGVjdGVkX3NoYXBlJTIwJTNEJTIwJTVCMSUyQyUyMGlucHV0cy5pbnB1dF9pZHMuc2hhcGUlNUItMSU1RCUyQyUyMG1vZGVsLmNvbmZpZy52b2NhYl9zaXplJTVEJTBBbGlzdChsb2dpdHMuc2hhcGUpJTIwJTNEJTNEJTIwZXhwZWN0ZWRfc2hhcGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BlenderbotSmallForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotSmallForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot_small-90M&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_shape = [<span class="hljs-number">1</span>, inputs.input_ids.shape[-<span class="hljs-number">1</span>], model.config.vocab_size]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape) == expected_shape
<span class="hljs-literal">True</span>`,wrap:!1}}),{c(){a=i("p"),a.textContent=T,m=n(),h(p.$$.fragment)},l(l){a=c(l,"P",{"data-svelte-h":!0}),k(a)!=="svelte-11lpom8"&&(a.textContent=T),m=s(l),u(p.$$.fragment,l)},m(l,v){d(l,a,v),d(l,m,v),f(p,l,v),y=!0},p:Le,i(l){y||(_(p.$$.fragment,l),y=!0)},o(l){g(p.$$.fragment,l),y=!1},d(l){l&&(o(a),o(m)),b(p,l)}}}function dn(B){let a,T,m,p,y,l="<em>This model was released on 2020-04-28 and added to Hugging Face Transformers on 2021-01-05.</em>",v,ae,dt,X,To='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',it,re,vo=`Note that <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel">BlenderbotSmallModel</a> and
<a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForConditionalGeneration">BlenderbotSmallForConditionalGeneration</a> are only used in combination with the checkpoint
<a href="https://huggingface.co/facebook/blenderbot-90M" rel="nofollow">facebook/blenderbot-90M</a>. Larger Blenderbot checkpoints should
instead be used with <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotModel">BlenderbotModel</a> and
<a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotForConditionalGeneration">BlenderbotForConditionalGeneration</a>`,ct,le,mt,de,Mo=`The Blender chatbot model was proposed in <a href="https://huggingface.co/papers/2004.13637" rel="nofollow">Recipes for building an open-domain chatbot</a> Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.`,pt,ie,wo="The abstract of the paper is the following:",ht,ce,$o=`<em>Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that
scaling neural models in the number of parameters and the size of the data they are trained on gives improved results,
we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of
skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to
their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent
persona. We show that large scale models can learn these skills when given appropriate training data and choice of
generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models
and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn
dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing
failure cases of our models.</em>`,ut,me,Bo=`This model was contributed by <a href="https://huggingface.co/patrickvonplaten" rel="nofollow">patrickvonplaten</a>. The authors’ code can be
found <a href="https://github.com/facebookresearch/ParlAI" rel="nofollow">here</a>.`,ft,pe,_t,he,Co=`Blenderbot Small is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than
the left.`,gt,ue,bt,fe,xo='<li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/translation">Translation task guide</a></li> <li><a href="../tasks/summarization">Summarization task guide</a></li>',kt,_e,yt,J,ge,Lt,Ze,So=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel">BlenderbotSmallModel</a>. It is used to instantiate
an BlenderbotSmall model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the BlenderbotSmall
<a href="https://huggingface.co/facebook/blenderbot_small-90M" rel="nofollow">facebook/blenderbot_small-90M</a> architecture.`,Zt,Ne,zo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Nt,P,Tt,be,vt,M,ke,Wt,We,Jo="Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)",Vt,Ve,Fo=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
the superclass for more information regarding methods.`,Et,G,ye,Ht,Ee,qo=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens.`,Rt,He,jo="This implementation does not add special tokens and this method should be overridden in a subclass.",Xt,O,Te,Pt,Re,Io=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> or <code>encode_plus</code> methods.`,Ot,L,ve,Dt,Xe,Uo=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Yt,Pe,Go="Should be overridden in a subclass if the model has a special way of building those.",At,Oe,Me,Mt,we,wt,I,$e,Qt,De,Lo="Construct a “fast” BlenderbotSmall tokenizer (backed by HuggingFace’s <em>tokenizers</em> library).",Kt,D,Be,eo,Ye,Zo=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. BlenderbotSmall
does not make use of token type ids, therefore a list of zeros is returned.`,$t,Ce,Bt,x,xe,to,Ae,No="The bare Blenderbot Small Model outputting raw hidden-states without any specific head on top.",oo,Qe,Wo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,no,Ke,Vo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,so,F,Se,ao,et,Eo='The <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel">BlenderbotSmallModel</a> forward method, overrides the <code>__call__</code> special method.',ro,Y,lo,A,Ct,ze,xt,S,Je,io,tt,Ho="The BlenderbotSmall Model with a language modeling head. Can be used for summarization.",co,ot,Ro=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,mo,nt,Xo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,po,q,Fe,ho,st,Po='The <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForConditionalGeneration">BlenderbotSmallForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',uo,Q,fo,K,St,qe,zt,V,je,_o,j,Ie,go,at,Oo='The <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForCausalLM">BlenderbotSmallForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',bo,ee,ko,te,Jt,Ue,Ft,rt,qt;return ae=new W({props:{title:"Blenderbot Small",local:"blenderbot-small",headingTag:"h1"}}),le=new W({props:{title:"Overview",local:"overview",headingTag:"h2"}}),pe=new W({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),ue=new W({props:{title:"Resources",local:"resources",headingTag:"h2"}}),_e=new W({props:{title:"BlenderbotSmallConfig",local:"transformers.BlenderbotSmallConfig",headingTag:"h2"}}),ge=new z({props:{name:"class transformers.BlenderbotSmallConfig",anchor:"transformers.BlenderbotSmallConfig",parameters:[{name:"vocab_size",val:" = 50265"},{name:"max_position_embeddings",val:" = 512"},{name:"encoder_layers",val:" = 8"},{name:"encoder_ffn_dim",val:" = 2048"},{name:"encoder_attention_heads",val:" = 16"},{name:"decoder_layers",val:" = 8"},{name:"decoder_ffn_dim",val:" = 2048"},{name:"decoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 512"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 1"},{name:"scale_embedding",val:" = False"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"forced_eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotSmallConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the BlenderbotSmall model. Defines the number of different tokens that can be
represented by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel">BlenderbotSmallModel</a> or <code>TFBlenderbotSmallModel</code>.`,name:"vocab_size"},{anchor:"transformers.BlenderbotSmallConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.BlenderbotSmallConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.BlenderbotSmallConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.BlenderbotSmallConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.BlenderbotSmallConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.BlenderbotSmallConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.BlenderbotSmallConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.BlenderbotSmallConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.BlenderbotSmallConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.BlenderbotSmallConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.BlenderbotSmallConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.BlenderbotSmallConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.BlenderbotSmallConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.BlenderbotSmallConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.BlenderbotSmallConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.BlenderbotSmallConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.BlenderbotSmallConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.BlenderbotSmallConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/configuration_blenderbot_small.py#L32"}}),P=new Ut({props:{anchor:"transformers.BlenderbotSmallConfig.example",$$slots:{default:[tn]},$$scope:{ctx:B}}}),be=new W({props:{title:"BlenderbotSmallTokenizer",local:"transformers.BlenderbotSmallTokenizer",headingTag:"h2"}}),ke=new z({props:{name:"class transformers.BlenderbotSmallTokenizer",anchor:"transformers.BlenderbotSmallTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"bos_token",val:" = '__start__'"},{name:"eos_token",val:" = '__end__'"},{name:"unk_token",val:" = '__unk__'"},{name:"pad_token",val:" = '__null__'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.BlenderbotSmallTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.BlenderbotSmallTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;__start__&quot;</code>) &#x2014;
The beginning of sentence token.`,name:"bos_token"},{anchor:"transformers.BlenderbotSmallTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;__end__&quot;</code>) &#x2014;
The end of sentence token.`,name:"eos_token"},{anchor:"transformers.BlenderbotSmallTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;__unk__&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BlenderbotSmallTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;__null__&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BlenderbotSmallTokenizer.kwargs",description:`<strong>kwargs</strong> (<em>optional</em>) &#x2014;
Additional keyword arguments passed along to <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/tokenization_blenderbot_small.py#L53"}}),ye=new z({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BlenderbotSmallTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizer.build_inputs_with_special_tokens.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.BlenderbotSmallTokenizer.build_inputs_with_special_tokens.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3456",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The model input with special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Te=new z({props:{name:"get_special_tokens_mask",anchor:"transformers.BlenderbotSmallTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids of the first sequence.`,name:"token_ids_0"},{anchor:"transformers.BlenderbotSmallTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
List of ids of the second sequence.`,name:"token_ids_1"},{anchor:"transformers.BlenderbotSmallTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils.py#L1008",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]</p>
`}}),ve=new z({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.BlenderbotSmallTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.BlenderbotSmallTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Me=new z({props:{name:"save_vocabulary",anchor:"transformers.BlenderbotSmallTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/tokenization_blenderbot_small.py#L192"}}),we=new W({props:{title:"BlenderbotSmallTokenizerFast",local:"transformers.BlenderbotSmallTokenizerFast",headingTag:"h2"}}),$e=new z({props:{name:"class transformers.BlenderbotSmallTokenizerFast",anchor:"transformers.BlenderbotSmallTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"unk_token",val:" = '<|endoftext|>'"},{name:"bos_token",val:" = '<|endoftext|>'"},{name:"eos_token",val:" = '<|endoftext|>'"},{name:"add_prefix_space",val:" = False"},{name:"trim_offsets",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/tokenization_blenderbot_small_fast.py#L35"}}),Be=new z({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.BlenderbotSmallTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.BlenderbotSmallTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/tokenization_blenderbot_small_fast.py#L79",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ce=new W({props:{title:"BlenderbotSmallModel",local:"transformers.BlenderbotSmallModel",headingTag:"h2"}}),xe=new z({props:{name:"class transformers.BlenderbotSmallModel",anchor:"transformers.BlenderbotSmallModel",parameters:[{name:"config",val:": BlenderbotSmallConfig"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig">BlenderbotSmallConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1104"}}),Se=new z({props:{name:"forward",anchor:"transformers.BlenderbotSmallModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[tuple, transformers.modeling_outputs.BaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotSmallModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>BlenderbotSmall uses the <code>bos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.BlenderbotSmallModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotSmallModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>Union[tuple, ~modeling_outputs.BaseModelOutput, NoneType]</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BlenderbotSmallModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BlenderbotSmallModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BlenderbotSmallModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BlenderbotSmallModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BlenderbotSmallModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotSmallModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotSmallModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BlenderbotSmallModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1130",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
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
`}}),Y=new yo({props:{$$slots:{default:[on]},$$scope:{ctx:B}}}),A=new Ut({props:{anchor:"transformers.BlenderbotSmallModel.forward.example",$$slots:{default:[nn]},$$scope:{ctx:B}}}),ze=new W({props:{title:"BlenderbotSmallForConditionalGeneration",local:"transformers.BlenderbotSmallForConditionalGeneration",headingTag:"h2"}}),Je=new z({props:{name:"class transformers.BlenderbotSmallForConditionalGeneration",anchor:"transformers.BlenderbotSmallForConditionalGeneration",parameters:[{name:"config",val:": BlenderbotSmallConfig"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig">BlenderbotSmallConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1250"}}),Fe=new z({props:{name:"forward",anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[tuple, transformers.modeling_outputs.BaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>BlenderbotSmall uses the <code>bos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>Union[tuple, ~modeling_outputs.BaseModelOutput, NoneType]</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1286",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
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
`}}),Q=new yo({props:{$$slots:{default:[sn]},$$scope:{ctx:B}}}),K=new Ut({props:{anchor:"transformers.BlenderbotSmallForConditionalGeneration.forward.example",$$slots:{default:[an]},$$scope:{ctx:B}}}),qe=new W({props:{title:"BlenderbotSmallForCausalLM",local:"transformers.BlenderbotSmallForCausalLM",headingTag:"h2"}}),je=new z({props:{name:"class transformers.BlenderbotSmallForCausalLM",anchor:"transformers.BlenderbotSmallForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1434"}}),Ie=new z({props:{name:"forward",anchor:"transformers.BlenderbotSmallForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotSmallForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BlenderbotSmallForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1460",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig"
>BlenderbotSmallConfig</a>) and inputs.</p>
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
`}}),ee=new yo({props:{$$slots:{default:[rn]},$$scope:{ctx:B}}}),te=new Ut({props:{anchor:"transformers.BlenderbotSmallForCausalLM.forward.example",$$slots:{default:[ln]},$$scope:{ctx:B}}}),Ue=new en({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/blenderbot-small.md"}}),{c(){a=i("meta"),T=n(),m=i("p"),p=n(),y=i("p"),y.innerHTML=l,v=n(),h(ae.$$.fragment),dt=n(),X=i("div"),X.innerHTML=To,it=n(),re=i("p"),re.innerHTML=vo,ct=n(),h(le.$$.fragment),mt=n(),de=i("p"),de.innerHTML=Mo,pt=n(),ie=i("p"),ie.textContent=wo,ht=n(),ce=i("p"),ce.innerHTML=$o,ut=n(),me=i("p"),me.innerHTML=Bo,ft=n(),h(pe.$$.fragment),_t=n(),he=i("p"),he.textContent=Co,gt=n(),h(ue.$$.fragment),bt=n(),fe=i("ul"),fe.innerHTML=xo,kt=n(),h(_e.$$.fragment),yt=n(),J=i("div"),h(ge.$$.fragment),Lt=n(),Ze=i("p"),Ze.innerHTML=So,Zt=n(),Ne=i("p"),Ne.innerHTML=zo,Nt=n(),h(P.$$.fragment),Tt=n(),h(be.$$.fragment),vt=n(),M=i("div"),h(ke.$$.fragment),Wt=n(),We=i("p"),We.textContent=Jo,Vt=n(),Ve=i("p"),Ve.innerHTML=Fo,Et=n(),G=i("div"),h(ye.$$.fragment),Ht=n(),Ee=i("p"),Ee.textContent=qo,Rt=n(),He=i("p"),He.textContent=jo,Xt=n(),O=i("div"),h(Te.$$.fragment),Pt=n(),Re=i("p"),Re.innerHTML=Io,Ot=n(),L=i("div"),h(ve.$$.fragment),Dt=n(),Xe=i("p"),Xe.innerHTML=Uo,Yt=n(),Pe=i("p"),Pe.textContent=Go,At=n(),Oe=i("div"),h(Me.$$.fragment),Mt=n(),h(we.$$.fragment),wt=n(),I=i("div"),h($e.$$.fragment),Qt=n(),De=i("p"),De.innerHTML=Lo,Kt=n(),D=i("div"),h(Be.$$.fragment),eo=n(),Ye=i("p"),Ye.textContent=Zo,$t=n(),h(Ce.$$.fragment),Bt=n(),x=i("div"),h(xe.$$.fragment),to=n(),Ae=i("p"),Ae.textContent=No,oo=n(),Qe=i("p"),Qe.innerHTML=Wo,no=n(),Ke=i("p"),Ke.innerHTML=Vo,so=n(),F=i("div"),h(Se.$$.fragment),ao=n(),et=i("p"),et.innerHTML=Eo,ro=n(),h(Y.$$.fragment),lo=n(),h(A.$$.fragment),Ct=n(),h(ze.$$.fragment),xt=n(),S=i("div"),h(Je.$$.fragment),io=n(),tt=i("p"),tt.textContent=Ho,co=n(),ot=i("p"),ot.innerHTML=Ro,mo=n(),nt=i("p"),nt.innerHTML=Xo,po=n(),q=i("div"),h(Fe.$$.fragment),ho=n(),st=i("p"),st.innerHTML=Po,uo=n(),h(Q.$$.fragment),fo=n(),h(K.$$.fragment),St=n(),h(qe.$$.fragment),zt=n(),V=i("div"),h(je.$$.fragment),_o=n(),j=i("div"),h(Ie.$$.fragment),go=n(),at=i("p"),at.innerHTML=Oo,bo=n(),h(ee.$$.fragment),ko=n(),h(te.$$.fragment),Jt=n(),h(Ue.$$.fragment),Ft=n(),rt=i("p"),this.h()},l(e){const t=Ko("svelte-u9bgzb",document.head);a=c(t,"META",{name:!0,content:!0}),t.forEach(o),T=s(e),m=c(e,"P",{}),C(m).forEach(o),p=s(e),y=c(e,"P",{"data-svelte-h":!0}),k(y)!=="svelte-qfwngr"&&(y.innerHTML=l),v=s(e),u(ae.$$.fragment,e),dt=s(e),X=c(e,"DIV",{class:!0,"data-svelte-h":!0}),k(X)!=="svelte-b95w5j"&&(X.innerHTML=To),it=s(e),re=c(e,"P",{"data-svelte-h":!0}),k(re)!=="svelte-rif1t0"&&(re.innerHTML=vo),ct=s(e),u(le.$$.fragment,e),mt=s(e),de=c(e,"P",{"data-svelte-h":!0}),k(de)!=="svelte-13bjtiw"&&(de.innerHTML=Mo),pt=s(e),ie=c(e,"P",{"data-svelte-h":!0}),k(ie)!=="svelte-wu27l3"&&(ie.textContent=wo),ht=s(e),ce=c(e,"P",{"data-svelte-h":!0}),k(ce)!=="svelte-1t366g8"&&(ce.innerHTML=$o),ut=s(e),me=c(e,"P",{"data-svelte-h":!0}),k(me)!=="svelte-t1s4wn"&&(me.innerHTML=Bo),ft=s(e),u(pe.$$.fragment,e),_t=s(e),he=c(e,"P",{"data-svelte-h":!0}),k(he)!=="svelte-sco7tr"&&(he.textContent=Co),gt=s(e),u(ue.$$.fragment,e),bt=s(e),fe=c(e,"UL",{"data-svelte-h":!0}),k(fe)!=="svelte-jwyjs9"&&(fe.innerHTML=xo),kt=s(e),u(_e.$$.fragment,e),yt=s(e),J=c(e,"DIV",{class:!0});var U=C(J);u(ge.$$.fragment,U),Lt=s(U),Ze=c(U,"P",{"data-svelte-h":!0}),k(Ze)!=="svelte-v9w6zm"&&(Ze.innerHTML=So),Zt=s(U),Ne=c(U,"P",{"data-svelte-h":!0}),k(Ne)!=="svelte-1ek1ss9"&&(Ne.innerHTML=zo),Nt=s(U),u(P.$$.fragment,U),U.forEach(o),Tt=s(e),u(be.$$.fragment,e),vt=s(e),M=c(e,"DIV",{class:!0});var w=C(M);u(ke.$$.fragment,w),Wt=s(w),We=c(w,"P",{"data-svelte-h":!0}),k(We)!=="svelte-12wmyl5"&&(We.textContent=Jo),Vt=s(w),Ve=c(w,"P",{"data-svelte-h":!0}),k(Ve)!=="svelte-1tuwqhk"&&(Ve.innerHTML=Fo),Et=s(w),G=c(w,"DIV",{class:!0});var E=C(G);u(ye.$$.fragment,E),Ht=s(E),Ee=c(E,"P",{"data-svelte-h":!0}),k(Ee)!=="svelte-xip562"&&(Ee.textContent=qo),Rt=s(E),He=c(E,"P",{"data-svelte-h":!0}),k(He)!=="svelte-1yvfiyo"&&(He.textContent=jo),E.forEach(o),Xt=s(w),O=c(w,"DIV",{class:!0});var Ge=C(O);u(Te.$$.fragment,Ge),Pt=s(Ge),Re=c(Ge,"P",{"data-svelte-h":!0}),k(Re)!=="svelte-1wmjg8a"&&(Re.innerHTML=Io),Ge.forEach(o),Ot=s(w),L=c(w,"DIV",{class:!0});var H=C(L);u(ve.$$.fragment,H),Dt=s(H),Xe=c(H,"P",{"data-svelte-h":!0}),k(Xe)!=="svelte-zj1vf1"&&(Xe.innerHTML=Uo),Yt=s(H),Pe=c(H,"P",{"data-svelte-h":!0}),k(Pe)!=="svelte-9vptpw"&&(Pe.textContent=Go),H.forEach(o),At=s(w),Oe=c(w,"DIV",{class:!0});var lt=C(Oe);u(Me.$$.fragment,lt),lt.forEach(o),w.forEach(o),Mt=s(e),u(we.$$.fragment,e),wt=s(e),I=c(e,"DIV",{class:!0});var R=C(I);u($e.$$.fragment,R),Qt=s(R),De=c(R,"P",{"data-svelte-h":!0}),k(De)!=="svelte-1w2pysy"&&(De.innerHTML=Lo),Kt=s(R),D=c(R,"DIV",{class:!0});var jt=C(D);u(Be.$$.fragment,jt),eo=s(jt),Ye=c(jt,"P",{"data-svelte-h":!0}),k(Ye)!=="svelte-a0gg85"&&(Ye.textContent=Zo),jt.forEach(o),R.forEach(o),$t=s(e),u(Ce.$$.fragment,e),Bt=s(e),x=c(e,"DIV",{class:!0});var Z=C(x);u(xe.$$.fragment,Z),to=s(Z),Ae=c(Z,"P",{"data-svelte-h":!0}),k(Ae)!=="svelte-1n46k6q"&&(Ae.textContent=No),oo=s(Z),Qe=c(Z,"P",{"data-svelte-h":!0}),k(Qe)!=="svelte-q52n56"&&(Qe.innerHTML=Wo),no=s(Z),Ke=c(Z,"P",{"data-svelte-h":!0}),k(Ke)!=="svelte-hswkmf"&&(Ke.innerHTML=Vo),so=s(Z),F=c(Z,"DIV",{class:!0});var oe=C(F);u(Se.$$.fragment,oe),ao=s(oe),et=c(oe,"P",{"data-svelte-h":!0}),k(et)!=="svelte-1mxfqhq"&&(et.innerHTML=Eo),ro=s(oe),u(Y.$$.fragment,oe),lo=s(oe),u(A.$$.fragment,oe),oe.forEach(o),Z.forEach(o),Ct=s(e),u(ze.$$.fragment,e),xt=s(e),S=c(e,"DIV",{class:!0});var N=C(S);u(Je.$$.fragment,N),io=s(N),tt=c(N,"P",{"data-svelte-h":!0}),k(tt)!=="svelte-1rd47ce"&&(tt.textContent=Ho),co=s(N),ot=c(N,"P",{"data-svelte-h":!0}),k(ot)!=="svelte-q52n56"&&(ot.innerHTML=Ro),mo=s(N),nt=c(N,"P",{"data-svelte-h":!0}),k(nt)!=="svelte-hswkmf"&&(nt.innerHTML=Xo),po=s(N),q=c(N,"DIV",{class:!0});var ne=C(q);u(Fe.$$.fragment,ne),ho=s(ne),st=c(ne,"P",{"data-svelte-h":!0}),k(st)!=="svelte-1w3es1w"&&(st.innerHTML=Po),uo=s(ne),u(Q.$$.fragment,ne),fo=s(ne),u(K.$$.fragment,ne),ne.forEach(o),N.forEach(o),St=s(e),u(qe.$$.fragment,e),zt=s(e),V=c(e,"DIV",{class:!0});var It=C(V);u(je.$$.fragment,It),_o=s(It),j=c(It,"DIV",{class:!0});var se=C(j);u(Ie.$$.fragment,se),go=s(se),at=c(se,"P",{"data-svelte-h":!0}),k(at)!=="svelte-1iwm4m2"&&(at.innerHTML=Oo),bo=s(se),u(ee.$$.fragment,se),ko=s(se),u(te.$$.fragment,se),se.forEach(o),It.forEach(o),Jt=s(e),u(Ue.$$.fragment,e),Ft=s(e),rt=c(e,"P",{}),C(rt).forEach(o),this.h()},h(){$(a,"name","hf:doc:metadata"),$(a,"content",cn),$(X,"class","flex flex-wrap space-x-1"),$(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){r(document.head,a),d(e,T,t),d(e,m,t),d(e,p,t),d(e,y,t),d(e,v,t),f(ae,e,t),d(e,dt,t),d(e,X,t),d(e,it,t),d(e,re,t),d(e,ct,t),f(le,e,t),d(e,mt,t),d(e,de,t),d(e,pt,t),d(e,ie,t),d(e,ht,t),d(e,ce,t),d(e,ut,t),d(e,me,t),d(e,ft,t),f(pe,e,t),d(e,_t,t),d(e,he,t),d(e,gt,t),f(ue,e,t),d(e,bt,t),d(e,fe,t),d(e,kt,t),f(_e,e,t),d(e,yt,t),d(e,J,t),f(ge,J,null),r(J,Lt),r(J,Ze),r(J,Zt),r(J,Ne),r(J,Nt),f(P,J,null),d(e,Tt,t),f(be,e,t),d(e,vt,t),d(e,M,t),f(ke,M,null),r(M,Wt),r(M,We),r(M,Vt),r(M,Ve),r(M,Et),r(M,G),f(ye,G,null),r(G,Ht),r(G,Ee),r(G,Rt),r(G,He),r(M,Xt),r(M,O),f(Te,O,null),r(O,Pt),r(O,Re),r(M,Ot),r(M,L),f(ve,L,null),r(L,Dt),r(L,Xe),r(L,Yt),r(L,Pe),r(M,At),r(M,Oe),f(Me,Oe,null),d(e,Mt,t),f(we,e,t),d(e,wt,t),d(e,I,t),f($e,I,null),r(I,Qt),r(I,De),r(I,Kt),r(I,D),f(Be,D,null),r(D,eo),r(D,Ye),d(e,$t,t),f(Ce,e,t),d(e,Bt,t),d(e,x,t),f(xe,x,null),r(x,to),r(x,Ae),r(x,oo),r(x,Qe),r(x,no),r(x,Ke),r(x,so),r(x,F),f(Se,F,null),r(F,ao),r(F,et),r(F,ro),f(Y,F,null),r(F,lo),f(A,F,null),d(e,Ct,t),f(ze,e,t),d(e,xt,t),d(e,S,t),f(Je,S,null),r(S,io),r(S,tt),r(S,co),r(S,ot),r(S,mo),r(S,nt),r(S,po),r(S,q),f(Fe,q,null),r(q,ho),r(q,st),r(q,uo),f(Q,q,null),r(q,fo),f(K,q,null),d(e,St,t),f(qe,e,t),d(e,zt,t),d(e,V,t),f(je,V,null),r(V,_o),r(V,j),f(Ie,j,null),r(j,go),r(j,at),r(j,bo),f(ee,j,null),r(j,ko),f(te,j,null),d(e,Jt,t),f(Ue,e,t),d(e,Ft,t),d(e,rt,t),qt=!0},p(e,[t]){const U={};t&2&&(U.$$scope={dirty:t,ctx:e}),P.$set(U);const w={};t&2&&(w.$$scope={dirty:t,ctx:e}),Y.$set(w);const E={};t&2&&(E.$$scope={dirty:t,ctx:e}),A.$set(E);const Ge={};t&2&&(Ge.$$scope={dirty:t,ctx:e}),Q.$set(Ge);const H={};t&2&&(H.$$scope={dirty:t,ctx:e}),K.$set(H);const lt={};t&2&&(lt.$$scope={dirty:t,ctx:e}),ee.$set(lt);const R={};t&2&&(R.$$scope={dirty:t,ctx:e}),te.$set(R)},i(e){qt||(_(ae.$$.fragment,e),_(le.$$.fragment,e),_(pe.$$.fragment,e),_(ue.$$.fragment,e),_(_e.$$.fragment,e),_(ge.$$.fragment,e),_(P.$$.fragment,e),_(be.$$.fragment,e),_(ke.$$.fragment,e),_(ye.$$.fragment,e),_(Te.$$.fragment,e),_(ve.$$.fragment,e),_(Me.$$.fragment,e),_(we.$$.fragment,e),_($e.$$.fragment,e),_(Be.$$.fragment,e),_(Ce.$$.fragment,e),_(xe.$$.fragment,e),_(Se.$$.fragment,e),_(Y.$$.fragment,e),_(A.$$.fragment,e),_(ze.$$.fragment,e),_(Je.$$.fragment,e),_(Fe.$$.fragment,e),_(Q.$$.fragment,e),_(K.$$.fragment,e),_(qe.$$.fragment,e),_(je.$$.fragment,e),_(Ie.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(Ue.$$.fragment,e),qt=!0)},o(e){g(ae.$$.fragment,e),g(le.$$.fragment,e),g(pe.$$.fragment,e),g(ue.$$.fragment,e),g(_e.$$.fragment,e),g(ge.$$.fragment,e),g(P.$$.fragment,e),g(be.$$.fragment,e),g(ke.$$.fragment,e),g(ye.$$.fragment,e),g(Te.$$.fragment,e),g(ve.$$.fragment,e),g(Me.$$.fragment,e),g(we.$$.fragment,e),g($e.$$.fragment,e),g(Be.$$.fragment,e),g(Ce.$$.fragment,e),g(xe.$$.fragment,e),g(Se.$$.fragment,e),g(Y.$$.fragment,e),g(A.$$.fragment,e),g(ze.$$.fragment,e),g(Je.$$.fragment,e),g(Fe.$$.fragment,e),g(Q.$$.fragment,e),g(K.$$.fragment,e),g(qe.$$.fragment,e),g(je.$$.fragment,e),g(Ie.$$.fragment,e),g(ee.$$.fragment,e),g(te.$$.fragment,e),g(Ue.$$.fragment,e),qt=!1},d(e){e&&(o(T),o(m),o(p),o(y),o(v),o(dt),o(X),o(it),o(re),o(ct),o(mt),o(de),o(pt),o(ie),o(ht),o(ce),o(ut),o(me),o(ft),o(_t),o(he),o(gt),o(bt),o(fe),o(kt),o(yt),o(J),o(Tt),o(vt),o(M),o(Mt),o(wt),o(I),o($t),o(Bt),o(x),o(Ct),o(xt),o(S),o(St),o(zt),o(V),o(Jt),o(Ft),o(rt)),o(a),b(ae,e),b(le,e),b(pe,e),b(ue,e),b(_e,e),b(ge),b(P),b(be,e),b(ke),b(ye),b(Te),b(ve),b(Me),b(we,e),b($e),b(Be),b(Ce,e),b(xe),b(Se),b(Y),b(A),b(ze,e),b(Je),b(Fe),b(Q),b(K),b(qe,e),b(je),b(Ie),b(ee),b(te),b(Ue,e)}}}const cn='{"title":"Blenderbot Small","local":"blenderbot-small","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"BlenderbotSmallConfig","local":"transformers.BlenderbotSmallConfig","sections":[],"depth":2},{"title":"BlenderbotSmallTokenizer","local":"transformers.BlenderbotSmallTokenizer","sections":[],"depth":2},{"title":"BlenderbotSmallTokenizerFast","local":"transformers.BlenderbotSmallTokenizerFast","sections":[],"depth":2},{"title":"BlenderbotSmallModel","local":"transformers.BlenderbotSmallModel","sections":[],"depth":2},{"title":"BlenderbotSmallForConditionalGeneration","local":"transformers.BlenderbotSmallForConditionalGeneration","sections":[],"depth":2},{"title":"BlenderbotSmallForCausalLM","local":"transformers.BlenderbotSmallForCausalLM","sections":[],"depth":2}],"depth":1}';function mn(B){return Yo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class kn extends Ao{constructor(a){super(),Qo(this,a,mn,dn,Do,{})}}export{kn as component};
