import{s as un,o as fn,n as E}from"../chunks/scheduler.18a86fab.js";import{S as gn,i as _n,g as p,s as a,r as u,A as bn,h,f as s,c as r,j as U,x as m,u as f,k as z,y as d,a as i,v as g,d as _,t as b,w as k}from"../chunks/index.98837b22.js";import{T as ft}from"../chunks/Tip.77304350.js";import{D as N}from"../chunks/Docstring.a1ef7999.js";import{C as Re}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as mt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as R,E as kn}from"../chunks/getInferenceSnippets.06c2775f.js";function yn(M){let t,y="Example:",l,c,v;return c=new Re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJsZW5kZXJib3RDb25maWclMkMlMjBCbGVuZGVyYm90TW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwQmxlbmRlcmJvdCUyMGZhY2Vib29rJTJGYmxlbmRlcmJvdC0zQiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBCbGVuZGVyYm90Q29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGZhY2Vib29rJTJGYmxlbmRlcmJvdC0zQiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwQmxlbmRlcmJvdE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotConfig, BlenderbotModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Blenderbot facebook/blenderbot-3B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BlenderbotConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the facebook/blenderbot-3B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(c.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(o),f(c.$$.fragment,o)},m(o,T){i(o,t,T),i(o,l,T),g(c,o,T),v=!0},p:E,i(o){v||(_(c.$$.fragment,o),v=!0)},o(o){b(c.$$.fragment,o),v=!1},d(o){o&&(s(t),s(l)),k(c,o)}}}function vn(M){let t,y="be encoded differently whether it is at the beginning of the sentence (without space) or not:",l,c,v;return c=new Re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJsZW5kZXJib3RUb2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBCbGVuZGVyYm90VG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmJsZW5kZXJib3QtM0IlMjIpJTBBdG9rZW5pemVyLmFkZF9wcmVmaXhfc3BhY2UlMjAlM0QlMjBGYWxzZSUwQXRva2VuaXplciglMjJIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUwQXRva2VuaXplciglMjIlMjBIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot-3B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.add_prefix_space = <span class="hljs-literal">False</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">47</span>, <span class="hljs-number">921</span>, <span class="hljs-number">86</span>, <span class="hljs-number">1085</span>, <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">6950</span>, <span class="hljs-number">1085</span>, <span class="hljs-number">2</span>]`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(c.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-12atnao"&&(t.textContent=y),l=r(o),f(c.$$.fragment,o)},m(o,T){i(o,t,T),i(o,l,T),g(c,o,T),v=!0},p:E,i(o){v||(_(c.$$.fragment,o),v=!0)},o(o){b(c.$$.fragment,o),v=!1},d(o){o&&(s(t),s(l)),k(c,o)}}}function Tn(M){let t,y="When used with <code>is_split_into_words=True</code>, this tokenizer will add a space before each word (even the first one).";return{c(){t=p("p"),t.innerHTML=y},l(l){t=h(l,"P",{"data-svelte-h":!0}),m(t)!=="svelte-jhmxzm"&&(t.innerHTML=y)},m(l,c){i(l,t,c)},p:E,d(l){l&&s(t)}}}function Mn(M){let t,y="be encoded differently whether it is at the beginning of the sentence (without space) or not:",l,c,v;return c=new Re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJsZW5kZXJib3RUb2tlbml6ZXJGYXN0JTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQmxlbmRlcmJvdFRva2VuaXplckZhc3QuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGYmxlbmRlcmJvdC0zQiUyMiklMEF0b2tlbml6ZXIoJTIySGVsbG8lMjB3b3JsZCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMEF0b2tlbml6ZXIoJTIyJTIwSGVsbG8lMjB3b3JsZCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotTokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotTokenizerFast.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot-3B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">6950</span>, <span class="hljs-number">1085</span>, <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">6950</span>, <span class="hljs-number">1085</span>, <span class="hljs-number">2</span>]`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(c.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-12atnao"&&(t.textContent=y),l=r(o),f(c.$$.fragment,o)},m(o,T){i(o,t,T),i(o,l,T),g(c,o,T),v=!0},p:E,i(o){v||(_(c.$$.fragment,o),v=!0)},o(o){b(c.$$.fragment,o),v=!1},d(o){o&&(s(t),s(l)),k(c,o)}}}function wn(M){let t,y="When used with <code>is_split_into_words=True</code>, this tokenizer needs to be instantiated with <code>add_prefix_space=True</code>.";return{c(){t=p("p"),t.innerHTML=y},l(l){t=h(l,"P",{"data-svelte-h":!0}),m(t)!=="svelte-9gg91e"&&(t.innerHTML=y)},m(l,c){i(l,t,c)},p:E,d(l){l&&s(t)}}}function $n(M){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=h(l,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,c){i(l,t,c)},p:E,d(l){l&&s(t)}}}function Bn(M){let t,y="Example:",l,c,v;return c=new Re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCbGVuZGVyYm90TW9kZWwlMEElMEFtb2RlbCUyMCUzRCUyMEJsZW5kZXJib3RNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZibGVuZGVyYm90LTQwME0tZGlzdGlsbCUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmJsZW5kZXJib3QtNDAwTS1kaXN0aWxsJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJTdHVkaWVzJTIwaGF2ZSUyMGJlZW4lMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFkZWNvZGVyX2lucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJTdHVkaWVzJTIwc2hvdyUyMHRoYXQlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUzRGlucHV0cy5pbnB1dF9pZHMlMkMlMjBkZWNvZGVyX2lucHV0X2lkcyUzRGRlY29kZXJfaW5wdXRfaWRzKSUwQSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGUlMEFsaXN0KGxhc3RfaGlkZGVuX3N0YXRlcy5zaGFwZSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BlenderbotModel

<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotModel.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot-400M-distill&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot-400M-distill&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">6</span>, <span class="hljs-number">1280</span>]`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(c.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(o),f(c.$$.fragment,o)},m(o,T){i(o,t,T),i(o,l,T),g(c,o,T),v=!0},p:E,i(o){v||(_(c.$$.fragment,o),v=!0)},o(o){b(c.$$.fragment,o),v=!1},d(o){o&&(s(t),s(l)),k(c,o)}}}function xn(M){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=h(l,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,c){i(l,t,c)},p:E,d(l){l&&s(t)}}}function Cn(M){let t,y="Example conversation:",l,c,v;return c=new Re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCbGVuZGVyYm90Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBbW5hbWUlMjAlM0QlMjAlMjJmYWNlYm9vayUyRmJsZW5kZXJib3QtNDAwTS1kaXN0aWxsJTIyJTBBbW9kZWwlMjAlM0QlMjBCbGVuZGVyYm90Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZChtbmFtZSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtbmFtZSklMEFVVFRFUkFOQ0UlMjAlM0QlMjAlMjJNeSUyMGZyaWVuZHMlMjBhcmUlMjBjb29sJTIwYnV0JTIwdGhleSUyMGVhdCUyMHRvbyUyMG1hbnklMjBjYXJicy4lMjIlMEFwcmludCglMjJIdW1hbiUzQSUyMCUyMiUyQyUyMFVUVEVSQU5DRSklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTVCVVRURVJBTkNFJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFyZXBseV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyklMEFwcmludCglMjJCb3QlM0ElMjAlMjIlMkMlMjB0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKHJlcGx5X2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQpJTBBJTBBUkVQTFklMjAlM0QlMjAlMjJJJ20lMjBub3QlMjBzdXJlJTIyJTBBcHJpbnQoJTIySHVtYW4lM0ElMjAlMjIlMkMlMjBSRVBMWSklMEElMEFORVhUX1VUVEVSQU5DRSUyMCUzRCUyMCglMEElMjAlMjAlMjAlMjAlMjJNeSUyMGZyaWVuZHMlMjBhcmUlMjBjb29sJTIwYnV0JTIwdGhleSUyMGVhdCUyMHRvbyUyMG1hbnklMjBjYXJicy4lM0MlMkZzJTNFJTIwJTNDcyUzRVRoYXQncyUyMHVuZm9ydHVuYXRlLiUyMCUyMiUwQSUyMCUyMCUyMCUyMCUyMkFyZSUyMHRoZXklMjB0cnlpbmclMjB0byUyMGxvc2UlMjB3ZWlnaHQlMjBvciUyMGFyZSUyMHRoZXklMjBqdXN0JTIwdHJ5aW5nJTIwdG8lMjBiZSUyMGhlYWx0aGllciUzRiUzQyUyRnMlM0UlMjAlMjIlMEElMjAlMjAlMjAlMjAlMjIlM0NzJTNFJTIwSSdtJTIwbm90JTIwc3VyZS4lMjIlMEEpJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCU1Qk5FWFRfVVRURVJBTkNFJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFuZXh0X3JlcGx5X2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzKSUwQXByaW50KCUyMkJvdCUzQSUyMCUyMiUyQyUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUobmV4dF9yZXBseV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklNUIwJTVEKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BlenderbotForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>mname = <span class="hljs-string">&quot;facebook/blenderbot-400M-distill&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotForConditionalGeneration.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>UTTERANCE = <span class="hljs-string">&quot;My friends are cool but they eat too many carbs.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Human: &quot;</span>, UTTERANCE)
Human:  My friends are cool but they eat too many carbs.

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([UTTERANCE], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>reply_ids = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Bot: &quot;</span>, tokenizer.batch_decode(reply_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>])
Bot: That<span class="hljs-string">&#x27;s unfortunate. Are they trying to lose weight or are they just trying to be healthier?

&gt;&gt;&gt; REPLY = &quot;I&#x27;</span>m <span class="hljs-keyword">not</span> sure<span class="hljs-string">&quot;
&gt;&gt;&gt; print(&quot;</span>Human: <span class="hljs-string">&quot;, REPLY)
Human: I&#x27;m not sure

&gt;&gt;&gt; NEXT_UTTERANCE = (
...     &quot;</span>My friends are cool but they eat too many carbs.&lt;/s&gt; &lt;s&gt;That<span class="hljs-string">&#x27;s unfortunate. &quot;
...     &quot;Are they trying to lose weight or are they just trying to be healthier?&lt;/s&gt; &quot;
...     &quot;&lt;s&gt; I&#x27;</span>m <span class="hljs-keyword">not</span> sure.<span class="hljs-string">&quot;
... )
&gt;&gt;&gt; inputs = tokenizer([NEXT_UTTERANCE], return_tensors=&quot;</span>pt<span class="hljs-string">&quot;)
&gt;&gt;&gt; next_reply_ids = model.generate(**inputs)
&gt;&gt;&gt; print(&quot;</span>Bot: <span class="hljs-string">&quot;, tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
Bot:   I see. Well, it&#x27;s good that they&#x27;re trying to change their eating habits.</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(c.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-1h939jd"&&(t.textContent=y),l=r(o),f(c.$$.fragment,o)},m(o,T){i(o,t,T),i(o,l,T),g(c,o,T),v=!0},p:E,i(o){v||(_(c.$$.fragment,o),v=!0)},o(o){b(c.$$.fragment,o),v=!1},d(o){o&&(s(t),s(l)),k(c,o)}}}function jn(M){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=h(l,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,c){i(l,t,c)},p:E,d(l){l&&s(t)}}}function zn(M){let t,y="Example:",l,c,v;return c=new Re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCbGVuZGVyYm90Rm9yQ2F1c2FsTE0lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmJsZW5kZXJib3QtNDAwTS1kaXN0aWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwQmxlbmRlcmJvdEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmJsZW5kZXJib3QtNDAwTS1kaXN0aWxsJTIyJTJDJTIwYWRkX2Nyb3NzX2F0dGVudGlvbiUzREZhbHNlKSUwQWFzc2VydCUyMG1vZGVsLmNvbmZpZy5pc19kZWNvZGVyJTJDJTIwZiUyMiU3Qm1vZGVsLl9fY2xhc3NfXyU3RCUyMGhhcyUyMHRvJTIwYmUlMjBjb25maWd1cmVkJTIwYXMlMjBhJTIwZGVjb2Rlci4lMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBZXhwZWN0ZWRfc2hhcGUlMjAlM0QlMjAlNUIxJTJDJTIwaW5wdXRzLmlucHV0X2lkcy5zaGFwZSU1Qi0xJTVEJTJDJTIwbW9kZWwuY29uZmlnLnZvY2FiX3NpemUlNUQlMEFsaXN0KGxvZ2l0cy5zaGFwZSklMjAlM0QlM0QlMjBleHBlY3RlZF9zaGFwZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BlenderbotForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot-400M-distill&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/blenderbot-400M-distill&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_shape = [<span class="hljs-number">1</span>, inputs.input_ids.shape[-<span class="hljs-number">1</span>], model.config.vocab_size]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape) == expected_shape
<span class="hljs-literal">True</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(c.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(o),f(c.$$.fragment,o)},m(o,T){i(o,t,T),i(o,l,T),g(c,o,T),v=!0},p:E,i(o){v||(_(c.$$.fragment,o),v=!0)},o(o){b(c.$$.fragment,o),v=!1},d(o){o&&(s(t),s(l)),k(c,o)}}}function qn(M){let t,y,l,c,v,o="<em>This model was released on 2020-04-28 and added to Hugging Face Transformers on 2020-11-16.</em>",T,de,gt,D,Jo='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',_t,le,bt,ce,Uo=`The Blender chatbot model was proposed in <a href="https://huggingface.co/papers/2004.13637" rel="nofollow">Recipes for building an open-domain chatbot</a> Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.`,kt,pe,Zo="The abstract of the paper is the following:",yt,he,Wo=`<em>Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that
scaling neural models in the number of parameters and the size of the data they are trained on gives improved results,
we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of
skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to
their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent
persona. We show that large scale models can learn these skills when given appropriate training data and choice of
generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models
and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn
dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing
failure cases of our models.</em>`,vt,me,Go='This model was contributed by <a href="https://huggingface.co/sshleifer" rel="nofollow">sshleifer</a>. The authors’ code can be found <a href="https://github.com/facebookresearch/ParlAI" rel="nofollow">here</a> .',Tt,ue,Mt,fe,Io=`Blenderbot is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right
rather than the left.`,wt,ge,Ho="An example:",$t,_e,Bt,be,xt,ke,Vo=`<li>Blenderbot uses a standard <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">seq2seq model transformer</a> based architecture.</li> <li>Available checkpoints can be found in the <a href="https://huggingface.co/models?search=blenderbot" rel="nofollow">model hub</a>.</li> <li>This is the <em>default</em> Blenderbot model class. However, some smaller checkpoints, such as
<code>facebook/blenderbot_small_90M</code>, have a different architecture and consequently should be used with
<a href="blenderbot-small">BlenderbotSmall</a>.</li>`,Ct,ye,jt,ve,Lo='<li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/translation">Translation task guide</a></li> <li><a href="../tasks/summarization">Summarization task guide</a></li>',zt,Te,qt,q,Me,Pt,Ee,No=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotModel">BlenderbotModel</a>. It is used to instantiate an
Blenderbot model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Blenderbot
<a href="https://huggingface.co/facebook/blenderbot-3B" rel="nofollow">facebook/blenderbot-3B</a> architecture.`,At,Xe,Ro=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Qt,Y,Ft,we,Jt,w,$e,Dt,Se,Eo="Constructs a Blenderbot tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.",Yt,Pe,Xo="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",Ot,O,Kt,Ae,So=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.`,eo,K,to,Qe,Po=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,oo,X,Be,no,De,Ao=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A Blenderbot sequence has the following format:`,so,Ye,Qo="<li>single sequence: <code>X &lt;/s&gt;</code></li>",Ut,xe,Zt,$,Ce,ao,Oe,Do=`Construct a “fast” Blenderbot tokenizer (backed by HuggingFace’s <em>tokenizers</em> library), derived from the GPT-2
tokenizer, using byte-level Byte-Pair-Encoding.`,ro,Ke,Yo="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",io,ee,lo,et,Oo=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.`,co,te,po,tt,Ko=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,ho,S,je,mo,ot,en=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A Blenderbot sequence has the following format:`,uo,nt,tn="<li>single sequence: <code>X &lt;/s&gt;</code></li>",Wt,ze,Gt,qe,on='See <a href="/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartModel">BartModel</a> for arguments to <em>forward</em> and <em>generate</em>',It,C,Fe,fo,st,nn="The bare Blenderbot Model outputting raw hidden-states without any specific head on top.",go,at,sn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_o,rt,an=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,bo,Z,Je,ko,it,rn='The <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotModel">BlenderbotModel</a> forward method, overrides the <code>__call__</code> special method.',yo,oe,vo,ne,Ht,Ue,Vt,Ze,dn='See <a href="/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForConditionalGeneration">BartForConditionalGeneration</a> for arguments to <em>forward</em> and <em>generate</em>',Lt,j,We,To,dt,ln="The Blenderbot Model with a language modeling head. Can be used for summarization.",Mo,lt,cn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,wo,ct,pn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$o,W,Ge,Bo,pt,hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotForConditionalGeneration">BlenderbotForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',xo,se,Co,ae,Nt,Ie,Rt,P,He,jo,G,Ve,zo,ht,mn='The <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotForCausalLM">BlenderbotForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',qo,re,Fo,ie,Et,Le,Xt,ut,St;return de=new R({props:{title:"Blenderbot",local:"blenderbot",headingTag:"h1"}}),le=new R({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ue=new R({props:{title:"Usage tips and example",local:"usage-tips-and-example",headingTag:"h2"}}),_e=new Re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJsZW5kZXJib3RUb2tlbml6ZXIlMkMlMjBCbGVuZGVyYm90Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBbW5hbWUlMjAlM0QlMjAlMjJmYWNlYm9vayUyRmJsZW5kZXJib3QtNDAwTS1kaXN0aWxsJTIyJTBBbW9kZWwlMjAlM0QlMjBCbGVuZGVyYm90Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZChtbmFtZSklMEF0b2tlbml6ZXIlMjAlM0QlMjBCbGVuZGVyYm90VG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtbmFtZSklMEFVVFRFUkFOQ0UlMjAlM0QlMjAlMjJNeSUyMGZyaWVuZHMlMjBhcmUlMjBjb29sJTIwYnV0JTIwdGhleSUyMGVhdCUyMHRvbyUyMG1hbnklMjBjYXJicy4lMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTVCVVRURVJBTkNFJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFyZXBseV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyklMEFwcmludCh0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKHJlcGx5X2lkcykp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BlenderbotTokenizer, BlenderbotForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>mname = <span class="hljs-string">&quot;facebook/blenderbot-400M-distill&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BlenderbotForConditionalGeneration.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BlenderbotTokenizer.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>UTTERANCE = <span class="hljs-string">&quot;My friends are cool but they eat too many carbs.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([UTTERANCE], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>reply_ids = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(tokenizer.batch_decode(reply_ids))
[<span class="hljs-string">&quot;&lt;s&gt; That&#x27;s unfortunate. Are they trying to lose weight or are they just trying to be healthier?&lt;/s&gt;&quot;</span>]`,wrap:!1}}),be=new R({props:{title:"Implementation Notes",local:"implementation-notes",headingTag:"h2"}}),ye=new R({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Te=new R({props:{title:"BlenderbotConfig",local:"transformers.BlenderbotConfig",headingTag:"h2"}}),Me=new N({props:{name:"class transformers.BlenderbotConfig",anchor:"transformers.BlenderbotConfig",parameters:[{name:"vocab_size",val:" = 8008"},{name:"max_position_embeddings",val:" = 128"},{name:"encoder_layers",val:" = 2"},{name:"encoder_ffn_dim",val:" = 10240"},{name:"encoder_attention_heads",val:" = 32"},{name:"decoder_layers",val:" = 24"},{name:"decoder_ffn_dim",val:" = 10240"},{name:"decoder_attention_heads",val:" = 32"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 2560"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 1"},{name:"scale_embedding",val:" = False"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"encoder_no_repeat_ngram_size",val:" = 3"},{name:"forced_eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the Blenderbot model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotModel">BlenderbotModel</a> or <code>TFBlenderbotModel</code>.`,name:"vocab_size"},{anchor:"transformers.BlenderbotConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.BlenderbotConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.BlenderbotConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.BlenderbotConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.BlenderbotConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.BlenderbotConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.BlenderbotConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.BlenderbotConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.BlenderbotConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.BlenderbotConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.BlenderbotConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.BlenderbotConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.BlenderbotConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.BlenderbotConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.BlenderbotConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.BlenderbotConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.BlenderbotConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.BlenderbotConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/configuration_blenderbot.py#L32"}}),Y=new mt({props:{anchor:"transformers.BlenderbotConfig.example",$$slots:{default:[yn]},$$scope:{ctx:M}}}),we=new R({props:{title:"BlenderbotTokenizer",local:"transformers.BlenderbotTokenizer",headingTag:"h2"}}),$e=new N({props:{name:"class transformers.BlenderbotTokenizer",anchor:"transformers.BlenderbotTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.BlenderbotTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.BlenderbotTokenizer.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.BlenderbotTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.BlenderbotTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.BlenderbotTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BlenderbotTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.BlenderbotTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BlenderbotTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BlenderbotTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.BlenderbotTokenizer.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (Blenderbot tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/tokenization_blenderbot.py#L79"}}),O=new mt({props:{anchor:"transformers.BlenderbotTokenizer.example",$$slots:{default:[vn]},$$scope:{ctx:M}}}),K=new ft({props:{$$slots:{default:[Tn]},$$scope:{ctx:M}}}),Be=new N({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BlenderbotTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added`,name:"token_ids_0"},{anchor:"transformers.BlenderbotTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Will be ignored`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/tokenization_blenderbot.py#L393",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),xe=new R({props:{title:"BlenderbotTokenizerFast",local:"transformers.BlenderbotTokenizerFast",headingTag:"h2"}}),Ce=new N({props:{name:"class transformers.BlenderbotTokenizerFast",anchor:"transformers.BlenderbotTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"trim_offsets",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BlenderbotTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.BlenderbotTokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.BlenderbotTokenizerFast.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.BlenderbotTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.BlenderbotTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.BlenderbotTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BlenderbotTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.BlenderbotTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BlenderbotTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BlenderbotTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.BlenderbotTokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (Blenderbot tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"},{anchor:"transformers.BlenderbotTokenizerFast.trim_offsets",description:`<strong>trim_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the post processing step should trim offsets to avoid including whitespaces.`,name:"trim_offsets"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/tokenization_blenderbot_fast.py#L38"}}),ee=new mt({props:{anchor:"transformers.BlenderbotTokenizerFast.example",$$slots:{default:[Mn]},$$scope:{ctx:M}}}),te=new ft({props:{$$slots:{default:[wn]},$$scope:{ctx:M}}}),je=new N({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BlenderbotTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added`,name:"token_ids_0"},{anchor:"transformers.BlenderbotTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Will be ignored`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/tokenization_blenderbot_fast.py#L267",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ze=new R({props:{title:"BlenderbotModel",local:"transformers.BlenderbotModel",headingTag:"h2"}}),Fe=new N({props:{name:"class transformers.BlenderbotModel",anchor:"transformers.BlenderbotModel",parameters:[{name:"config",val:": BlenderbotConfig"}],parametersDescription:[{anchor:"transformers.BlenderbotModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig">BlenderbotConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/modeling_blenderbot.py#L1119"}}),Je=new N({props:{name:"forward",anchor:"transformers.BlenderbotModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[tuple, transformers.modeling_outputs.BaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Blenderbot uses the <code>bos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.BlenderbotModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.BlenderbotModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BlenderbotModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>Union[tuple, ~modeling_outputs.BaseModelOutput, NoneType]</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BlenderbotModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BlenderbotModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BlenderbotModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BlenderbotModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BlenderbotModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BlenderbotModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/modeling_blenderbot.py#L1158",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig"
>BlenderbotConfig</a>) and inputs.</p>
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
`}}),oe=new ft({props:{$$slots:{default:[$n]},$$scope:{ctx:M}}}),ne=new mt({props:{anchor:"transformers.BlenderbotModel.forward.example",$$slots:{default:[Bn]},$$scope:{ctx:M}}}),Ue=new R({props:{title:"BlenderbotForConditionalGeneration",local:"transformers.BlenderbotForConditionalGeneration",headingTag:"h2"}}),We=new N({props:{name:"class transformers.BlenderbotForConditionalGeneration",anchor:"transformers.BlenderbotForConditionalGeneration",parameters:[{name:"config",val:": BlenderbotConfig"}],parametersDescription:[{anchor:"transformers.BlenderbotForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig">BlenderbotConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/modeling_blenderbot.py#L1278"}}),Ge=new N({props:{name:"forward",anchor:"transformers.BlenderbotForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[tuple, transformers.modeling_outputs.BaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Blenderbot uses the <code>bos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>Union[tuple, ~modeling_outputs.BaseModelOutput, NoneType]</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BlenderbotForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/modeling_blenderbot.py#L1327",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig"
>BlenderbotConfig</a>) and inputs.</p>
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
`}}),se=new ft({props:{$$slots:{default:[xn]},$$scope:{ctx:M}}}),ae=new mt({props:{anchor:"transformers.BlenderbotForConditionalGeneration.forward.example",$$slots:{default:[Cn]},$$scope:{ctx:M}}}),Ie=new R({props:{title:"BlenderbotForCausalLM",local:"transformers.BlenderbotForCausalLM",headingTag:"h2"}}),He=new N({props:{name:"class transformers.BlenderbotForCausalLM",anchor:"transformers.BlenderbotForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/modeling_blenderbot.py#L1475"}}),Ve=new N({props:{name:"forward",anchor:"transformers.BlenderbotForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.BlenderbotForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BlenderbotForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BlenderbotForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BlenderbotForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BlenderbotForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BlenderbotForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BlenderbotForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BlenderbotForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BlenderbotForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BlenderbotForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BlenderbotForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BlenderbotForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BlenderbotForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BlenderbotForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/blenderbot/modeling_blenderbot.py#L1501",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/blenderbot#transformers.BlenderbotConfig"
>BlenderbotConfig</a>) and inputs.</p>
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
`}}),re=new ft({props:{$$slots:{default:[jn]},$$scope:{ctx:M}}}),ie=new mt({props:{anchor:"transformers.BlenderbotForCausalLM.forward.example",$$slots:{default:[zn]},$$scope:{ctx:M}}}),Le=new kn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/blenderbot.md"}}),{c(){t=p("meta"),y=a(),l=p("p"),c=a(),v=p("p"),v.innerHTML=o,T=a(),u(de.$$.fragment),gt=a(),D=p("div"),D.innerHTML=Jo,_t=a(),u(le.$$.fragment),bt=a(),ce=p("p"),ce.innerHTML=Uo,kt=a(),pe=p("p"),pe.textContent=Zo,yt=a(),he=p("p"),he.innerHTML=Wo,vt=a(),me=p("p"),me.innerHTML=Go,Tt=a(),u(ue.$$.fragment),Mt=a(),fe=p("p"),fe.textContent=Io,wt=a(),ge=p("p"),ge.textContent=Ho,$t=a(),u(_e.$$.fragment),Bt=a(),u(be.$$.fragment),xt=a(),ke=p("ul"),ke.innerHTML=Vo,Ct=a(),u(ye.$$.fragment),jt=a(),ve=p("ul"),ve.innerHTML=Lo,zt=a(),u(Te.$$.fragment),qt=a(),q=p("div"),u(Me.$$.fragment),Pt=a(),Ee=p("p"),Ee.innerHTML=No,At=a(),Xe=p("p"),Xe.innerHTML=Ro,Qt=a(),u(Y.$$.fragment),Ft=a(),u(we.$$.fragment),Jt=a(),w=p("div"),u($e.$$.fragment),Dt=a(),Se=p("p"),Se.textContent=Eo,Yt=a(),Pe=p("p"),Pe.textContent=Xo,Ot=a(),u(O.$$.fragment),Kt=a(),Ae=p("p"),Ae.innerHTML=So,eo=a(),u(K.$$.fragment),to=a(),Qe=p("p"),Qe.innerHTML=Po,oo=a(),X=p("div"),u(Be.$$.fragment),no=a(),De=p("p"),De.textContent=Ao,so=a(),Ye=p("ul"),Ye.innerHTML=Qo,Ut=a(),u(xe.$$.fragment),Zt=a(),$=p("div"),u(Ce.$$.fragment),ao=a(),Oe=p("p"),Oe.innerHTML=Do,ro=a(),Ke=p("p"),Ke.textContent=Yo,io=a(),u(ee.$$.fragment),lo=a(),et=p("p"),et.innerHTML=Oo,co=a(),u(te.$$.fragment),po=a(),tt=p("p"),tt.innerHTML=Ko,ho=a(),S=p("div"),u(je.$$.fragment),mo=a(),ot=p("p"),ot.textContent=en,uo=a(),nt=p("ul"),nt.innerHTML=tn,Wt=a(),u(ze.$$.fragment),Gt=a(),qe=p("p"),qe.innerHTML=on,It=a(),C=p("div"),u(Fe.$$.fragment),fo=a(),st=p("p"),st.textContent=nn,go=a(),at=p("p"),at.innerHTML=sn,_o=a(),rt=p("p"),rt.innerHTML=an,bo=a(),Z=p("div"),u(Je.$$.fragment),ko=a(),it=p("p"),it.innerHTML=rn,yo=a(),u(oe.$$.fragment),vo=a(),u(ne.$$.fragment),Ht=a(),u(Ue.$$.fragment),Vt=a(),Ze=p("p"),Ze.innerHTML=dn,Lt=a(),j=p("div"),u(We.$$.fragment),To=a(),dt=p("p"),dt.textContent=ln,Mo=a(),lt=p("p"),lt.innerHTML=cn,wo=a(),ct=p("p"),ct.innerHTML=pn,$o=a(),W=p("div"),u(Ge.$$.fragment),Bo=a(),pt=p("p"),pt.innerHTML=hn,xo=a(),u(se.$$.fragment),Co=a(),u(ae.$$.fragment),Nt=a(),u(Ie.$$.fragment),Rt=a(),P=p("div"),u(He.$$.fragment),jo=a(),G=p("div"),u(Ve.$$.fragment),zo=a(),ht=p("p"),ht.innerHTML=mn,qo=a(),u(re.$$.fragment),Fo=a(),u(ie.$$.fragment),Et=a(),u(Le.$$.fragment),Xt=a(),ut=p("p"),this.h()},l(e){const n=bn("svelte-u9bgzb",document.head);t=h(n,"META",{name:!0,content:!0}),n.forEach(s),y=r(e),l=h(e,"P",{}),U(l).forEach(s),c=r(e),v=h(e,"P",{"data-svelte-h":!0}),m(v)!=="svelte-1aysrmb"&&(v.innerHTML=o),T=r(e),f(de.$$.fragment,e),gt=r(e),D=h(e,"DIV",{class:!0,"data-svelte-h":!0}),m(D)!=="svelte-b95w5j"&&(D.innerHTML=Jo),_t=r(e),f(le.$$.fragment,e),bt=r(e),ce=h(e,"P",{"data-svelte-h":!0}),m(ce)!=="svelte-13bjtiw"&&(ce.innerHTML=Uo),kt=r(e),pe=h(e,"P",{"data-svelte-h":!0}),m(pe)!=="svelte-wu27l3"&&(pe.textContent=Zo),yt=r(e),he=h(e,"P",{"data-svelte-h":!0}),m(he)!=="svelte-1t366g8"&&(he.innerHTML=Wo),vt=r(e),me=h(e,"P",{"data-svelte-h":!0}),m(me)!=="svelte-1m719jj"&&(me.innerHTML=Go),Tt=r(e),f(ue.$$.fragment,e),Mt=r(e),fe=h(e,"P",{"data-svelte-h":!0}),m(fe)!=="svelte-c524ca"&&(fe.textContent=Io),wt=r(e),ge=h(e,"P",{"data-svelte-h":!0}),m(ge)!=="svelte-839ylp"&&(ge.textContent=Ho),$t=r(e),f(_e.$$.fragment,e),Bt=r(e),f(be.$$.fragment,e),xt=r(e),ke=h(e,"UL",{"data-svelte-h":!0}),m(ke)!=="svelte-egkfu8"&&(ke.innerHTML=Vo),Ct=r(e),f(ye.$$.fragment,e),jt=r(e),ve=h(e,"UL",{"data-svelte-h":!0}),m(ve)!=="svelte-jwyjs9"&&(ve.innerHTML=Lo),zt=r(e),f(Te.$$.fragment,e),qt=r(e),q=h(e,"DIV",{class:!0});var I=U(q);f(Me.$$.fragment,I),Pt=r(I),Ee=h(I,"P",{"data-svelte-h":!0}),m(Ee)!=="svelte-18mu2mk"&&(Ee.innerHTML=No),At=r(I),Xe=h(I,"P",{"data-svelte-h":!0}),m(Xe)!=="svelte-1ek1ss9"&&(Xe.innerHTML=Ro),Qt=r(I),f(Y.$$.fragment,I),I.forEach(s),Ft=r(e),f(we.$$.fragment,e),Jt=r(e),w=h(e,"DIV",{class:!0});var B=U(w);f($e.$$.fragment,B),Dt=r(B),Se=h(B,"P",{"data-svelte-h":!0}),m(Se)!=="svelte-7wwln4"&&(Se.textContent=Eo),Yt=r(B),Pe=h(B,"P",{"data-svelte-h":!0}),m(Pe)!=="svelte-1s077p3"&&(Pe.textContent=Xo),Ot=r(B),f(O.$$.fragment,B),Kt=r(B),Ae=h(B,"P",{"data-svelte-h":!0}),m(Ae)!=="svelte-1jfcabo"&&(Ae.innerHTML=So),eo=r(B),f(K.$$.fragment,B),to=r(B),Qe=h(B,"P",{"data-svelte-h":!0}),m(Qe)!=="svelte-ntrhio"&&(Qe.innerHTML=Po),oo=r(B),X=h(B,"DIV",{class:!0});var A=U(X);f(Be.$$.fragment,A),no=r(A),De=h(A,"P",{"data-svelte-h":!0}),m(De)!=="svelte-kc5n5c"&&(De.textContent=Ao),so=r(A),Ye=h(A,"UL",{"data-svelte-h":!0}),m(Ye)!=="svelte-g8j1jo"&&(Ye.innerHTML=Qo),A.forEach(s),B.forEach(s),Ut=r(e),f(xe.$$.fragment,e),Zt=r(e),$=h(e,"DIV",{class:!0});var x=U($);f(Ce.$$.fragment,x),ao=r(x),Oe=h(x,"P",{"data-svelte-h":!0}),m(Oe)!=="svelte-mpku7q"&&(Oe.innerHTML=Do),ro=r(x),Ke=h(x,"P",{"data-svelte-h":!0}),m(Ke)!=="svelte-1s077p3"&&(Ke.textContent=Yo),io=r(x),f(ee.$$.fragment,x),lo=r(x),et=h(x,"P",{"data-svelte-h":!0}),m(et)!=="svelte-1jfcabo"&&(et.innerHTML=Oo),co=r(x),f(te.$$.fragment,x),po=r(x),tt=h(x,"P",{"data-svelte-h":!0}),m(tt)!=="svelte-gxzj9w"&&(tt.innerHTML=Ko),ho=r(x),S=h(x,"DIV",{class:!0});var Q=U(S);f(je.$$.fragment,Q),mo=r(Q),ot=h(Q,"P",{"data-svelte-h":!0}),m(ot)!=="svelte-kc5n5c"&&(ot.textContent=en),uo=r(Q),nt=h(Q,"UL",{"data-svelte-h":!0}),m(nt)!=="svelte-g8j1jo"&&(nt.innerHTML=tn),Q.forEach(s),x.forEach(s),Wt=r(e),f(ze.$$.fragment,e),Gt=r(e),qe=h(e,"P",{"data-svelte-h":!0}),m(qe)!=="svelte-vxp8pg"&&(qe.innerHTML=on),It=r(e),C=h(e,"DIV",{class:!0});var F=U(C);f(Fe.$$.fragment,F),fo=r(F),st=h(F,"P",{"data-svelte-h":!0}),m(st)!=="svelte-w0tdit"&&(st.textContent=nn),go=r(F),at=h(F,"P",{"data-svelte-h":!0}),m(at)!=="svelte-q52n56"&&(at.innerHTML=sn),_o=r(F),rt=h(F,"P",{"data-svelte-h":!0}),m(rt)!=="svelte-hswkmf"&&(rt.innerHTML=an),bo=r(F),Z=h(F,"DIV",{class:!0});var H=U(Z);f(Je.$$.fragment,H),ko=r(H),it=h(H,"P",{"data-svelte-h":!0}),m(it)!=="svelte-5k3va"&&(it.innerHTML=rn),yo=r(H),f(oe.$$.fragment,H),vo=r(H),f(ne.$$.fragment,H),H.forEach(s),F.forEach(s),Ht=r(e),f(Ue.$$.fragment,e),Vt=r(e),Ze=h(e,"P",{"data-svelte-h":!0}),m(Ze)!=="svelte-1ki59r4"&&(Ze.innerHTML=dn),Lt=r(e),j=h(e,"DIV",{class:!0});var J=U(j);f(We.$$.fragment,J),To=r(J),dt=h(J,"P",{"data-svelte-h":!0}),m(dt)!=="svelte-1gxq22f"&&(dt.textContent=ln),Mo=r(J),lt=h(J,"P",{"data-svelte-h":!0}),m(lt)!=="svelte-q52n56"&&(lt.innerHTML=cn),wo=r(J),ct=h(J,"P",{"data-svelte-h":!0}),m(ct)!=="svelte-hswkmf"&&(ct.innerHTML=pn),$o=r(J),W=h(J,"DIV",{class:!0});var V=U(W);f(Ge.$$.fragment,V),Bo=r(V),pt=h(V,"P",{"data-svelte-h":!0}),m(pt)!=="svelte-1vs4gvu"&&(pt.innerHTML=hn),xo=r(V),f(se.$$.fragment,V),Co=r(V),f(ae.$$.fragment,V),V.forEach(s),J.forEach(s),Nt=r(e),f(Ie.$$.fragment,e),Rt=r(e),P=h(e,"DIV",{class:!0});var Ne=U(P);f(He.$$.fragment,Ne),jo=r(Ne),G=h(Ne,"DIV",{class:!0});var L=U(G);f(Ve.$$.fragment,L),zo=r(L),ht=h(L,"P",{"data-svelte-h":!0}),m(ht)!=="svelte-3x7tpy"&&(ht.innerHTML=mn),qo=r(L),f(re.$$.fragment,L),Fo=r(L),f(ie.$$.fragment,L),L.forEach(s),Ne.forEach(s),Et=r(e),f(Le.$$.fragment,e),Xt=r(e),ut=h(e,"P",{}),U(ut).forEach(s),this.h()},h(){z(t,"name","hf:doc:metadata"),z(t,"content",Fn),z(D,"class","flex flex-wrap space-x-1"),z(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){d(document.head,t),i(e,y,n),i(e,l,n),i(e,c,n),i(e,v,n),i(e,T,n),g(de,e,n),i(e,gt,n),i(e,D,n),i(e,_t,n),g(le,e,n),i(e,bt,n),i(e,ce,n),i(e,kt,n),i(e,pe,n),i(e,yt,n),i(e,he,n),i(e,vt,n),i(e,me,n),i(e,Tt,n),g(ue,e,n),i(e,Mt,n),i(e,fe,n),i(e,wt,n),i(e,ge,n),i(e,$t,n),g(_e,e,n),i(e,Bt,n),g(be,e,n),i(e,xt,n),i(e,ke,n),i(e,Ct,n),g(ye,e,n),i(e,jt,n),i(e,ve,n),i(e,zt,n),g(Te,e,n),i(e,qt,n),i(e,q,n),g(Me,q,null),d(q,Pt),d(q,Ee),d(q,At),d(q,Xe),d(q,Qt),g(Y,q,null),i(e,Ft,n),g(we,e,n),i(e,Jt,n),i(e,w,n),g($e,w,null),d(w,Dt),d(w,Se),d(w,Yt),d(w,Pe),d(w,Ot),g(O,w,null),d(w,Kt),d(w,Ae),d(w,eo),g(K,w,null),d(w,to),d(w,Qe),d(w,oo),d(w,X),g(Be,X,null),d(X,no),d(X,De),d(X,so),d(X,Ye),i(e,Ut,n),g(xe,e,n),i(e,Zt,n),i(e,$,n),g(Ce,$,null),d($,ao),d($,Oe),d($,ro),d($,Ke),d($,io),g(ee,$,null),d($,lo),d($,et),d($,co),g(te,$,null),d($,po),d($,tt),d($,ho),d($,S),g(je,S,null),d(S,mo),d(S,ot),d(S,uo),d(S,nt),i(e,Wt,n),g(ze,e,n),i(e,Gt,n),i(e,qe,n),i(e,It,n),i(e,C,n),g(Fe,C,null),d(C,fo),d(C,st),d(C,go),d(C,at),d(C,_o),d(C,rt),d(C,bo),d(C,Z),g(Je,Z,null),d(Z,ko),d(Z,it),d(Z,yo),g(oe,Z,null),d(Z,vo),g(ne,Z,null),i(e,Ht,n),g(Ue,e,n),i(e,Vt,n),i(e,Ze,n),i(e,Lt,n),i(e,j,n),g(We,j,null),d(j,To),d(j,dt),d(j,Mo),d(j,lt),d(j,wo),d(j,ct),d(j,$o),d(j,W),g(Ge,W,null),d(W,Bo),d(W,pt),d(W,xo),g(se,W,null),d(W,Co),g(ae,W,null),i(e,Nt,n),g(Ie,e,n),i(e,Rt,n),i(e,P,n),g(He,P,null),d(P,jo),d(P,G),g(Ve,G,null),d(G,zo),d(G,ht),d(G,qo),g(re,G,null),d(G,Fo),g(ie,G,null),i(e,Et,n),g(Le,e,n),i(e,Xt,n),i(e,ut,n),St=!0},p(e,[n]){const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),Y.$set(I);const B={};n&2&&(B.$$scope={dirty:n,ctx:e}),O.$set(B);const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),K.$set(A);const x={};n&2&&(x.$$scope={dirty:n,ctx:e}),ee.$set(x);const Q={};n&2&&(Q.$$scope={dirty:n,ctx:e}),te.$set(Q);const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),oe.$set(F);const H={};n&2&&(H.$$scope={dirty:n,ctx:e}),ne.$set(H);const J={};n&2&&(J.$$scope={dirty:n,ctx:e}),se.$set(J);const V={};n&2&&(V.$$scope={dirty:n,ctx:e}),ae.$set(V);const Ne={};n&2&&(Ne.$$scope={dirty:n,ctx:e}),re.$set(Ne);const L={};n&2&&(L.$$scope={dirty:n,ctx:e}),ie.$set(L)},i(e){St||(_(de.$$.fragment,e),_(le.$$.fragment,e),_(ue.$$.fragment,e),_(_e.$$.fragment,e),_(be.$$.fragment,e),_(ye.$$.fragment,e),_(Te.$$.fragment,e),_(Me.$$.fragment,e),_(Y.$$.fragment,e),_(we.$$.fragment,e),_($e.$$.fragment,e),_(O.$$.fragment,e),_(K.$$.fragment,e),_(Be.$$.fragment,e),_(xe.$$.fragment,e),_(Ce.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(je.$$.fragment,e),_(ze.$$.fragment,e),_(Fe.$$.fragment,e),_(Je.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(Ue.$$.fragment,e),_(We.$$.fragment,e),_(Ge.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(Ie.$$.fragment,e),_(He.$$.fragment,e),_(Ve.$$.fragment,e),_(re.$$.fragment,e),_(ie.$$.fragment,e),_(Le.$$.fragment,e),St=!0)},o(e){b(de.$$.fragment,e),b(le.$$.fragment,e),b(ue.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(ye.$$.fragment,e),b(Te.$$.fragment,e),b(Me.$$.fragment,e),b(Y.$$.fragment,e),b(we.$$.fragment,e),b($e.$$.fragment,e),b(O.$$.fragment,e),b(K.$$.fragment,e),b(Be.$$.fragment,e),b(xe.$$.fragment,e),b(Ce.$$.fragment,e),b(ee.$$.fragment,e),b(te.$$.fragment,e),b(je.$$.fragment,e),b(ze.$$.fragment,e),b(Fe.$$.fragment,e),b(Je.$$.fragment,e),b(oe.$$.fragment,e),b(ne.$$.fragment,e),b(Ue.$$.fragment,e),b(We.$$.fragment,e),b(Ge.$$.fragment,e),b(se.$$.fragment,e),b(ae.$$.fragment,e),b(Ie.$$.fragment,e),b(He.$$.fragment,e),b(Ve.$$.fragment,e),b(re.$$.fragment,e),b(ie.$$.fragment,e),b(Le.$$.fragment,e),St=!1},d(e){e&&(s(y),s(l),s(c),s(v),s(T),s(gt),s(D),s(_t),s(bt),s(ce),s(kt),s(pe),s(yt),s(he),s(vt),s(me),s(Tt),s(Mt),s(fe),s(wt),s(ge),s($t),s(Bt),s(xt),s(ke),s(Ct),s(jt),s(ve),s(zt),s(qt),s(q),s(Ft),s(Jt),s(w),s(Ut),s(Zt),s($),s(Wt),s(Gt),s(qe),s(It),s(C),s(Ht),s(Vt),s(Ze),s(Lt),s(j),s(Nt),s(Rt),s(P),s(Et),s(Xt),s(ut)),s(t),k(de,e),k(le,e),k(ue,e),k(_e,e),k(be,e),k(ye,e),k(Te,e),k(Me),k(Y),k(we,e),k($e),k(O),k(K),k(Be),k(xe,e),k(Ce),k(ee),k(te),k(je),k(ze,e),k(Fe),k(Je),k(oe),k(ne),k(Ue,e),k(We),k(Ge),k(se),k(ae),k(Ie,e),k(He),k(Ve),k(re),k(ie),k(Le,e)}}}const Fn='{"title":"Blenderbot","local":"blenderbot","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips and example","local":"usage-tips-and-example","sections":[],"depth":2},{"title":"Implementation Notes","local":"implementation-notes","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"BlenderbotConfig","local":"transformers.BlenderbotConfig","sections":[],"depth":2},{"title":"BlenderbotTokenizer","local":"transformers.BlenderbotTokenizer","sections":[],"depth":2},{"title":"BlenderbotTokenizerFast","local":"transformers.BlenderbotTokenizerFast","sections":[],"depth":2},{"title":"BlenderbotModel","local":"transformers.BlenderbotModel","sections":[],"depth":2},{"title":"BlenderbotForConditionalGeneration","local":"transformers.BlenderbotForConditionalGeneration","sections":[],"depth":2},{"title":"BlenderbotForCausalLM","local":"transformers.BlenderbotForCausalLM","sections":[],"depth":2}],"depth":1}';function Jn(M){return fn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Ln extends gn{constructor(t){super(),_n(this,t,Jn,qn,un,{})}}export{Ln as component};
