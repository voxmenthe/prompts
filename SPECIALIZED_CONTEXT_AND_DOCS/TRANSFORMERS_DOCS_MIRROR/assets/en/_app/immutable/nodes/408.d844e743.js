import{s as Bo,o as Xo,n as L}from"../chunks/scheduler.18a86fab.js";import{S as Vo,i as Lo,g as p,s as r,r as g,A as Po,h,f as s,c as a,j as z,x as f,u as _,k as U,l as Ao,y as i,a as l,v as T,d as w,t as y,w as b}from"../chunks/index.98837b22.js";import{T as Ut}from"../chunks/Tip.77304350.js";import{D as H}from"../chunks/Docstring.a1ef7999.js";import{C as Ze}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as mo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Re,E as Qo}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Oo,a as uo}from"../chunks/HfOption.6641485e.js";function Yo(v){let t,u='This model was contributed by <a href="https://huggingface.co/ybelkada" rel="nofollow">ybelkada</a> and <a href="https://huggingface.co/ArthurZ" rel="nofollow">ArthurZ</a>.',o,c,M="Click on the Switch Transformers models in the right sidebar for more examples of how to apply Switch Transformers to different natural language tasks.";return{c(){t=p("p"),t.innerHTML=u,o=r(),c=p("p"),c.textContent=M},l(d){t=h(d,"P",{"data-svelte-h":!0}),f(t)!=="svelte-hr7sy3"&&(t.innerHTML=u),o=a(d),c=h(d,"P",{"data-svelte-h":!0}),f(c)!=="svelte-1d1qxm0"&&(c.textContent=M)},m(d,m){l(d,t,m),l(d,o,m),l(d,c,m)},p:L,d(d){d&&(s(t),s(o),s(c))}}}function Do(v){let t,u;return t=new Ze({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0MnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUyMCUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyZ29vZ2xlJTJGc3dpdGNoLWJhc2UtOCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZSUzRDAlMEEpJTBBcHJpbnQocGlwZWxpbmUoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlM0NleHRyYV9pZF8wJTNFLiUyMikp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text2text-generation&quot;</span>, 
    model=<span class="hljs-string">&quot;google/switch-base-8&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
<span class="hljs-built_in">print</span>(pipeline(<span class="hljs-string">&quot;The capital of France is &lt;extra_id_0&gt;.&quot;</span>))`,wrap:!1}}),{c(){g(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,c){T(t,o,c),u=!0},p:L,i(o){u||(w(t.$$.fragment,o),u=!0)},o(o){y(t.$$.fragment,o),u=!1},d(o){b(t,o)}}}function Ko(v){let t,u;return t=new Ze({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnN3aXRjaC1iYXNlLTglMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXEyU2VxTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnN3aXRjaC1iYXNlLTglMjIlMkMlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiklMEElMEFpbnB1dF90ZXh0JTIwJTNEJTIwJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlM0NleHRyYV9pZF8wJTNFLiUyMiUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcihpbnB1dF90ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikuaW5wdXRfaWRzLnRvKDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0X2lkcyklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>)
model = AutoModelForSeq2SeqLM.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>, dtype=torch.float16)

input_text = <span class="hljs-string">&quot;The capital of France is &lt;extra_id_0&gt;.&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids.to(<span class="hljs-number">0</span>)

outputs = model.generate(input_ids)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>]))`,wrap:!1}}),{c(){g(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,c){T(t,o,c),u=!0},p:L,i(o){u||(w(t.$$.fragment,o),u=!0)},o(o){y(t.$$.fragment,o),u=!1},d(o){b(t,o)}}}function en(v){let t,u;return t=new Ze({props:{code:"ZWNobyUyMC1lJTIwJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlM0NleHRyYV9pZF8wJTNFLiUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQydGV4dC1nZW5lcmF0aW9uJTIwLS1tb2RlbCUyMGdvb2dsZSUyRnN3aXRjaC1iYXNlLTglMjAtLWRldmljZSUyMDAlMEElMjMlMjAlNUIlN0InZ2VuZXJhdGVkX3RleHQnJTNBJTIwJ1BhcmlzLiclN0QlNUQ=",highlighted:`<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;The capital of France is &lt;extra_id_0&gt;.&quot;</span> | transformers run --task text2text-generation --model google/switch-base-8 --device 0
<span class="hljs-comment"># [{&#x27;generated_text&#x27;: &#x27;Paris.&#x27;}]</span>`,wrap:!1}}),{c(){g(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,c){T(t,o,c),u=!0},p:L,i(o){u||(w(t.$$.fragment,o),u=!0)},o(o){y(t.$$.fragment,o),u=!1},d(o){b(t,o)}}}function tn(v){let t,u,o,c,M,d;return t=new uo({props:{id:"usage",option:"Pipeline",$$slots:{default:[Do]},$$scope:{ctx:v}}}),o=new uo({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ko]},$$scope:{ctx:v}}}),M=new uo({props:{id:"usage",option:"transformers CLI",$$slots:{default:[en]},$$scope:{ctx:v}}}),{c(){g(t.$$.fragment),u=r(),g(o.$$.fragment),c=r(),g(M.$$.fragment)},l(m){_(t.$$.fragment,m),u=a(m),_(o.$$.fragment,m),c=a(m),_(M.$$.fragment,m)},m(m,k){T(t,m,k),l(m,u,k),T(o,m,k),l(m,c,k),T(M,m,k),d=!0},p(m,k){const rt={};k&2&&(rt.$$scope={dirty:k,ctx:m}),t.$set(rt);const ie={};k&2&&(ie.$$scope={dirty:k,ctx:m}),o.$set(ie);const E={};k&2&&(E.$$scope={dirty:k,ctx:m}),M.$set(E)},i(m){d||(w(t.$$.fragment,m),w(o.$$.fragment,m),w(M.$$.fragment,m),d=!0)},o(m){y(t.$$.fragment,m),y(o.$$.fragment,m),y(M.$$.fragment,m),d=!1},d(m){m&&(s(u),s(c)),b(t,m),b(o,m),b(M,m)}}}function on(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=h(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,c){l(o,t,c)},p:L,d(o){o&&s(t)}}}function nn(v){let t,u="Example:",o,c,M;return c=new Ze({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTd2l0Y2hUcmFuc2Zvcm1lcnNNb2RlbCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnN3aXRjaC1iYXNlLTglMjIpJTBBbW9kZWwlMjAlM0QlMjBTd2l0Y2hUcmFuc2Zvcm1lcnNNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGc3dpdGNoLWJhc2UtOCUyMiklMEElMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIyU3R1ZGllcyUyMGhhdmUlMjBiZWVuJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEFkZWNvZGVyX2lucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJTdHVkaWVzJTIwc2hvdyUyMHRoYXQlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBJTBBJTIzJTIwcHJlcHJvY2VzcyUzQSUyMFByZXBlbmQlMjBkZWNvZGVyX2lucHV0X2lkcyUyMHdpdGglMjBzdGFydCUyMHRva2VuJTIwd2hpY2glMjBpcyUyMHBhZCUyMHRva2VuJTIwZm9yJTIwU3dpdGNoVHJhbnNmb3JtZXJzTW9kZWwuJTBBJTIzJTIwVGhpcyUyMGlzJTIwbm90JTIwbmVlZGVkJTIwZm9yJTIwdG9yY2gncyUyMFN3aXRjaFRyYW5zZm9ybWVyc0ZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyMGFzJTIwaXQlMjBkb2VzJTIwdGhpcyUyMGludGVybmFsbHklMjB1c2luZyUyMGxhYmVscyUyMGFyZy4lMEFkZWNvZGVyX2lucHV0X2lkcyUyMCUzRCUyMG1vZGVsLl9zaGlmdF9yaWdodChkZWNvZGVyX2lucHV0X2lkcyklMEElMEElMjMlMjBmb3J3YXJkJTIwcGFzcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dF9pZHMlMkMlMjBkZWNvZGVyX2lucHV0X2lkcyUzRGRlY29kZXJfaW5wdXRfaWRzKSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SwitchTransformersModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SwitchTransformersModel.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># preprocess: Prepend decoder_input_ids with start token which is pad token for SwitchTransformersModel.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># This is not needed for torch&#x27;s SwitchTransformersForConditionalGeneration as it does this internally using labels arg.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = model._shift_right(decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=r(),g(c.$$.fragment)},l(d){t=h(d,"P",{"data-svelte-h":!0}),f(t)!=="svelte-11lpom8"&&(t.textContent=u),o=a(d),_(c.$$.fragment,d)},m(d,m){l(d,t,m),l(d,o,m),T(c,d,m),M=!0},p:L,i(d){M||(w(c.$$.fragment,d),M=!0)},o(d){y(c.$$.fragment,d),M=!1},d(d){d&&(s(t),s(o)),b(c,d)}}}function sn(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=h(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,c){l(o,t,c)},p:L,d(o){o&&s(t)}}}function rn(v){let t,u="Examples:",o,c,M;return c=new Ze({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTd2l0Y2hUcmFuc2Zvcm1lcnNGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZzd2l0Y2gtYmFzZS04JTIyKSUwQW1vZGVsJTIwJTNEJTIwU3dpdGNoVHJhbnNmb3JtZXJzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZzd2l0Y2gtYmFzZS04JTIyKSUwQSUwQSUyMyUyMHRyYWluaW5nJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMCUzQ2V4dHJhX2lkXzAlM0UlMjB3YWxrcyUyMGluJTIwJTNDZXh0cmFfaWRfMSUzRSUyMHBhcmslMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyJTNDZXh0cmFfaWRfMCUzRSUyMGN1dGUlMjBkb2clMjAlM0NleHRyYV9pZF8xJTNFJTIwdGhlJTIwJTNDZXh0cmFfaWRfMiUzRSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dF9pZHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTBBJTIzJTIwaW5mZXJlbmNlJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMnN1bW1hcml6ZSUzQSUyMHN0dWRpZXMlMjBoYXZlJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRfaWRzKSUwQSUyMyUyMC4lMjBUbyUyQyUyMGxldCVFMiU4MCU5OXMlMjBzYXklMjB5b3UlMjBoYXZlJTIwYSUyMGRvZy4lMjBUbyUyMHN1bW1hcml6ZSUzQSUwQSUyMyUyMFNpbmNlJTIwdGhlJTIwbW9kZWwlMjBoYXMlMjBiZWVuJTIwdHJhaW5lZCUyMG9uJTIwTUxNJTJDJTIwdGhpcyUyMHdpbGwlMjBvdXRwdXQlMjBnaWJiZXJpc2g=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SwitchTransformersForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SwitchTransformersForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># training</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(<span class="hljs-string">&quot;The &lt;extra_id_0&gt; walks in &lt;extra_id_1&gt; park&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;&lt;extra_id_0&gt; cute dog &lt;extra_id_1&gt; the &lt;extra_id_2&gt;&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># inference</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;summarize: studies have shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.generate(input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># . To, let’s say you have a dog. To summarize:</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Since the model has been trained on MLM, this will output gibberish</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=r(),g(c.$$.fragment)},l(d){t=h(d,"P",{"data-svelte-h":!0}),f(t)!=="svelte-kvfsh7"&&(t.textContent=u),o=a(d),_(c.$$.fragment,d)},m(d,m){l(d,t,m),l(d,o,m),T(c,d,m),M=!0},p:L,i(d){M||(w(c.$$.fragment,d),M=!0)},o(d){y(c.$$.fragment,d),M=!1},d(d){d&&(s(t),s(o)),b(c,d)}}}function an(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=h(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,c){l(o,t,c)},p:L,d(o){o&&s(t)}}}function dn(v){let t,u="Example:",o,c,M;return c=new Ze({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTd2l0Y2hUcmFuc2Zvcm1lcnNFbmNvZGVyTW9kZWwlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZzd2l0Y2gtYmFzZS04JTIyKSUwQW1vZGVsJTIwJTNEJTIwU3dpdGNoVHJhbnNmb3JtZXJzRW5jb2Rlck1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZzd2l0Y2gtYmFzZS04JTIyKSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJTdHVkaWVzJTIwaGF2ZSUyMGJlZW4lMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSkuaW5wdXRfaWRzJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dF9pZHMpJTBBbGFzdF9oaWRkZW5fc3RhdGVzJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SwitchTransformersEncoderModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SwitchTransformersEncoderModel.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=r(),g(c.$$.fragment)},l(d){t=h(d,"P",{"data-svelte-h":!0}),f(t)!=="svelte-11lpom8"&&(t.textContent=u),o=a(d),_(c.$$.fragment,d)},m(d,m){l(d,t,m),l(d,o,m),T(c,d,m),M=!0},p:L,i(d){M||(w(c.$$.fragment,d),M=!0)},o(d){y(c.$$.fragment,d),M=!1},d(d){d&&(s(t),s(o)),b(c,d)}}}function ln(v){let t,u,o,c,M,d="<em>This model was released on 2021-01-11 and added to Hugging Face Transformers on 2022-11-15.</em>",m,k,rt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',ie,E,it,de,fo='<a href="https://huggingface.co/papers/2101.03961" rel="nofollow">Switch Transformers</a> is a sparse T5 model where the MLP layer is replaced by a Mixture-of-Experts (MoE). A routing mechanism associates each token with an expert and each expert is a dense MLP. Sparsity enables better scaling and the routing mechanism allows the model to select relevant weights on the fly which increases model capacity.',dt,le,go='You can find all the original Switch Transformers checkpoints under the <a href="https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f" rel="nofollow">Switch Transformer</a> collection.',lt,O,ct,ce,_o='The example below demonstrates how to predict the masked token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',pt,Y,ht,pe,To='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',mt,he,wo='The example below uses <a href="../quantization/bitsandbytes/">bitsandbytes</a> to only quantize the weights to 8-bits.',ut,me,ft,ue,gt,q,fe,Jt,Ie,yo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersModel">SwitchTransformersModel</a>. It is used to
instantiate a SwitchTransformers model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the
SwitchTransformers <a href="https://huggingface.co/google/switch-base-8" rel="nofollow">google/switch-base-8</a> architecture.`,jt,He,bo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,_t,ge,Tt,$,_e,Ft,qe,Mo="Router using tokens choose top-1 experts assignment.",Rt,Ge,vo=`This router uses the same mechanism as in Switch Transformer (<a href="https://huggingface.co/papers/2101.03961" rel="nofollow">https://huggingface.co/papers/2101.03961</a>) and V-MoE
(<a href="https://huggingface.co/papers/2106.05974" rel="nofollow">https://huggingface.co/papers/2106.05974</a>): tokens choose their top experts. Items are sorted by router_probs and then
routed to their choice of expert until the expert’s expert_capacity is reached. <strong>There is no guarantee that each
token is processed by an expert</strong>, or that each expert receives at least one token.`,Zt,D,Te,It,We,ko="Computes router probabilities from input hidden states.",Ht,B,we,qt,Ne,$o=`Generic forward function for every Router class. Each Router expects to have the same input hidden states
(<code>hidden_states</code>) corresponding to the hidden states for each token, the <code>expert_capacity</code> corresponding to the
number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.`,Gt,Ee,So=`Each Router works as the following: it expects the hidden states for each token, gets the <code>router_probs</code> and
<code>router_logits</code> from the <code>router_weights</code>. This will assign for each token, the raw probability to be assigned
to an expert. Then each Router class will have to define its own <code>_compute_routing_instructions</code>.`,wt,ye,yt,G,be,Wt,Be,xo="Implementation of the Switch Transformers Sparse MLP module.",Nt,F,Me,Et,Xe,Co="Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:",Bt,Ve,zo=`1- Gets the <code>router_mask</code> from the router. The shape of the mask is <code>(batch_size, sequence_length, num_expert)</code>
and corresponds to the argmax of the <code>router_probs</code>. The probabilities are needed in the computation of the
hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).`,Xt,Le,Uo=`2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
expert the corresponding hidden states.`,bt,ve,Mt,S,ke,Vt,Pe,Jo="The bare Switch Transformers Model outputting raw hidden-states without any specific head on top.",Lt,Ae,jo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Pt,Qe,Fo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,At,R,$e,Qt,Oe,Ro='The <a href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersModel">SwitchTransformersModel</a> forward method, overrides the <code>__call__</code> special method.',Ot,K,Yt,ee,vt,Se,kt,x,xe,Dt,Ye,Zo="SWITCH_TRANSFORMERS Model with a <code>language modeling</code> head on top.",Kt,De,Io=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,eo,Ke,Ho=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,to,Z,Ce,oo,et,qo='The <a href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersForConditionalGeneration">SwitchTransformersForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',no,te,so,oe,$t,ze,St,C,Ue,ro,tt,Go="The bare SWITCH_TRANSFORMERS Model transformer outputting encoder’s raw hidden-states without any specific head",ao,ot,Wo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,io,nt,No=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,lo,I,Je,co,st,Eo='The <a href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersEncoderModel">SwitchTransformersEncoderModel</a> forward method, overrides the <code>__call__</code> special method.',po,ne,ho,se,xt,je,Ct,at,zt;return E=new Re({props:{title:"Switch Transformers",local:"switch-transformers",headingTag:"h1"}}),O=new Ut({props:{warning:!1,$$slots:{default:[Yo]},$$scope:{ctx:v}}}),Y=new Oo({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[tn]},$$scope:{ctx:v}}}),me=new Ze({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMGJpdHNhbmRieXRlcyUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvclNlcTJTZXFMTSUyQyUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaXRzQW5kQnl0ZXNDb25maWclMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZzd2l0Y2gtYmFzZS04JTIyKSUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBCaXRzQW5kQnl0ZXNDb25maWcobG9hZF9pbl84Yml0JTNEVHJ1ZSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvclNlcTJTZXFMTS5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGc3dpdGNoLWJhc2UtOCUyMiUyQyUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRHF1YW50aXphdGlvbl9jb25maWcpJTBBJTBBaW5wdXRfdGV4dCUyMCUzRCUyMCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDZXh0cmFfaWRfMCUzRS4lMjIlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoaW5wdXRfdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcy50bygwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dF9pZHMpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXRzJTVCMCU1RCkp",highlighted:`<span class="hljs-comment"># pip install bitsandbytes</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>)
quantization_config = BitsAndBytesConfig(load_in_8bit=<span class="hljs-literal">True</span>)
model = AutoModelForSeq2SeqLM.from_pretrained(<span class="hljs-string">&quot;google/switch-base-8&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>, quantization_config=quantization_config)

input_text = <span class="hljs-string">&quot;The capital of France is &lt;extra_id_0&gt;.&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids.to(<span class="hljs-number">0</span>)

outputs = model.generate(input_ids)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>]))`,wrap:!1}}),ue=new Re({props:{title:"SwitchTransformersConfig",local:"transformers.SwitchTransformersConfig",headingTag:"h2"}}),fe=new H({props:{name:"class transformers.SwitchTransformersConfig",anchor:"transformers.SwitchTransformersConfig",parameters:[{name:"vocab_size",val:" = 32128"},{name:"d_model",val:" = 768"},{name:"d_kv",val:" = 64"},{name:"d_ff",val:" = 2048"},{name:"expert_capacity",val:" = 64"},{name:"num_layers",val:" = 12"},{name:"num_sparse_encoder_layers",val:" = 3"},{name:"num_decoder_layers",val:" = 12"},{name:"num_sparse_decoder_layers",val:" = 3"},{name:"num_heads",val:" = 12"},{name:"num_experts",val:" = 8"},{name:"router_bias",val:" = False"},{name:"router_jitter_noise",val:" = 0.01"},{name:"router_dtype",val:" = 'float32'"},{name:"router_ignore_padding_tokens",val:" = False"},{name:"relative_attention_num_buckets",val:" = 32"},{name:"relative_attention_max_distance",val:" = 128"},{name:"dropout_rate",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-06"},{name:"router_z_loss_coef",val:" = 0.001"},{name:"router_aux_loss_coef",val:" = 0.001"},{name:"initializer_factor",val:" = 1.0"},{name:"dense_act_fn",val:" = 'relu'"},{name:"is_encoder_decoder",val:" = True"},{name:"add_router_probs",val:" = False"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SwitchTransformersConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32128) &#x2014;
Vocabulary size of the SwitchTransformers model. Defines the number of different tokens that can be
represented by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersModel">SwitchTransformersModel</a>.`,name:"vocab_size"},{anchor:"transformers.SwitchTransformersConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Size of the encoder layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.SwitchTransformersConfig.d_kv",description:`<strong>d_kv</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Size of the key, query, value projections per attention head. <code>d_kv</code> has to be equal to <code>d_model // num_heads</code>.`,name:"d_kv"},{anchor:"transformers.SwitchTransformersConfig.d_ff",description:`<strong>d_ff</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Size of the intermediate feed forward layer in each <code>SwitchTransformersBlock</code>.`,name:"d_ff"},{anchor:"transformers.SwitchTransformersConfig.expert_capacity",description:`<strong>expert_capacity</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of tokens that can be stored in each expert. If set to 1, the model will behave like a regular
Transformer.`,name:"expert_capacity"},{anchor:"transformers.SwitchTransformersConfig.num_layers",description:`<strong>num_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of dense hidden layers in the Transformer encoder layer.`,name:"num_layers"},{anchor:"transformers.SwitchTransformersConfig.num_sparse_encoder_layers",description:`<strong>num_sparse_encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Number of sparse (MoE) dense hidden layers in the Transformer encoder layer.`,name:"num_sparse_encoder_layers"},{anchor:"transformers.SwitchTransformersConfig.num_decoder_layers",description:`<strong>num_decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer decoder. Will use the same value as <code>num_layers</code> if not set.`,name:"num_decoder_layers"},{anchor:"transformers.SwitchTransformersConfig.num_sparse_decoder_layers",description:`<strong>num_sparse_decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Number of sparse (MoE) dense hidden layers in the Transformer decoder layer.`,name:"num_sparse_decoder_layers"},{anchor:"transformers.SwitchTransformersConfig.num_heads",description:`<strong>num_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_heads"},{anchor:"transformers.SwitchTransformersConfig.num_experts",description:`<strong>num_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of experts for each SwitchTransformer layer.`,name:"num_experts"},{anchor:"transformers.SwitchTransformersConfig.router_bias",description:`<strong>router_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to add a bias to the router.`,name:"router_bias"},{anchor:"transformers.SwitchTransformersConfig.router_jitter_noise",description:`<strong>router_jitter_noise</strong> (<code>float</code>, <em>optional</em>, defaults to 0.01) &#x2014;
Amount of noise to add to the router.`,name:"router_jitter_noise"},{anchor:"transformers.SwitchTransformersConfig.router_dtype",description:`<strong>router_dtype</strong> (<code>str</code>, <em>optional</em>, default to <code>&quot;float32&quot;</code>) &#x2014;
The <code>dtype</code> used for the routers. It is preferable to keep the <code>dtype</code> to <code>&quot;float32&quot;</code> as specified in the
<em>selective precision</em> discussion in <a href="https://huggingface.co/papers/2101.03961" rel="nofollow">the paper</a>.`,name:"router_dtype"},{anchor:"transformers.SwitchTransformersConfig.router_ignore_padding_tokens",description:`<strong>router_ignore_padding_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to ignore padding tokens when routing.`,name:"router_ignore_padding_tokens"},{anchor:"transformers.SwitchTransformersConfig.relative_attention_num_buckets",description:`<strong>relative_attention_num_buckets</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of buckets to use for each attention layer.`,name:"relative_attention_num_buckets"},{anchor:"transformers.SwitchTransformersConfig.relative_attention_max_distance",description:`<strong>relative_attention_max_distance</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The maximum distance of the longer sequences for the bucket separation.`,name:"relative_attention_max_distance"},{anchor:"transformers.SwitchTransformersConfig.dropout_rate",description:`<strong>dropout_rate</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The ratio for all dropout layers.`,name:"dropout_rate"},{anchor:"transformers.SwitchTransformersConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-6) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.SwitchTransformersConfig.router_z_loss_coef",description:`<strong>router_z_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014;
The z loss factor for the total loss.`,name:"router_z_loss_coef"},{anchor:"transformers.SwitchTransformersConfig.router_aux_loss_coef",description:`<strong>router_aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014;
The aux loss factor for the total loss.`,name:"router_aux_loss_coef"},{anchor:"transformers.SwitchTransformersConfig.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"},{anchor:"transformers.SwitchTransformersConfig.dense_act_fn",description:`<strong>dense_act_fn</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
Type of feed forward layer to be used. Should be one of <code>&quot;relu&quot;</code> or <code>&quot;gated-gelu&quot;</code>. SwitchTransformersv1.1
uses the <code>&quot;gated-gelu&quot;</code> feed forward projection. Original SwitchTransformers uses <code>&quot;relu&quot;</code>.`,name:"dense_act_fn"},{anchor:"transformers.SwitchTransformersConfig.add_router_probs",description:`<strong>add_router_probs</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to output router probabilities to compute router auxiliary loss.`,name:"add_router_probs"},{anchor:"transformers.SwitchTransformersConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/configuration_switch_transformers.py#L24"}}),ge=new Re({props:{title:"SwitchTransformersTop1Router",local:"transformers.SwitchTransformersTop1Router",headingTag:"h2"}}),_e=new H({props:{name:"class transformers.SwitchTransformersTop1Router",anchor:"transformers.SwitchTransformersTop1Router",parameters:[{name:"config",val:": SwitchTransformersConfig"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L126"}}),Te=new H({props:{name:"_compute_router_probabilities",anchor:"transformers.SwitchTransformersTop1Router._compute_router_probabilities",parameters:[{name:"hidden_states",val:": Tensor"}],parametersDescription:[{anchor:"transformers.SwitchTransformersTop1Router._compute_router_probabilities.hidden_states",description:`<strong>hidden_states</strong> (<code>torch.Tensor</code>) &#x2014;
(batch_size, sequence_length, hidden_dim) from which router probabilities are computed.`,name:"hidden_states"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L146",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
token and expert. Used for routing tokens to experts.
router_logits (<code>torch.Tensor</code>):
Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
This is used later for computing router z-loss.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>router_probabilities (<code>torch.Tensor</code>)</p>
`}}),we=new H({props:{name:"forward",anchor:"transformers.SwitchTransformersTop1Router.forward",parameters:[{name:"hidden_states",val:": Tensor"}],parametersDescription:[{anchor:"transformers.SwitchTransformersTop1Router.forward.hidden_states",description:`<strong>hidden_states</strong> (<code>torch.Tensor</code>)  &#x2014;
[num_groups, tokens_per_group, hidden_dim] inputs to send to experts.`,name:"hidden_states"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L187",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>tuple[<code>torch.Tensor</code>, <code>torch.Tensor</code>, <code>torch.Tensor</code>] Tuple containing the expert index, the router probs
and the router logits. The router probabilities and logits are required to compute the loss.</p>
`}}),ye=new Re({props:{title:"SwitchTransformersSparseMLP",local:"transformers.SwitchTransformersSparseMLP",headingTag:"h2"}}),be=new H({props:{name:"class transformers.SwitchTransformersSparseMLP",anchor:"transformers.SwitchTransformersSparseMLP",parameters:[{name:"config",val:": SwitchTransformersConfig"},{name:"expert_class",val:": Module = <class 'transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersDenseActDense'>"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L268"}}),Me=new H({props:{name:"forward",anchor:"transformers.SwitchTransformersSparseMLP.forward",parameters:[{name:"hidden_states",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L283"}}),ve=new Re({props:{title:"SwitchTransformersModel",local:"transformers.SwitchTransformersModel",headingTag:"h2"}}),ke=new H({props:{name:"class transformers.SwitchTransformersModel",anchor:"transformers.SwitchTransformersModel",parameters:[{name:"config",val:": SwitchTransformersConfig"}],parametersDescription:[{anchor:"transformers.SwitchTransformersModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig">SwitchTransformersConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1224"}}),$e=new H({props:{name:"forward",anchor:"transformers.SwitchTransformersModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.SwitchTransformersModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. SWITCH_TRANSFORMERS is a model with relative position
embeddings so you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./switch_transformers#training">SWITCH_TRANSFORMERS
Training</a>.`,name:"input_ids"},{anchor:"transformers.SwitchTransformersModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SwitchTransformersModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>SWITCH_TRANSFORMERS uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./switch_transformers#training">SWITCH_TRANSFORMERS
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.SwitchTransformersModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.SwitchTransformersModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SwitchTransformersModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.SwitchTransformersModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.SwitchTransformersModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.SwitchTransformersModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SwitchTransformersModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SwitchTransformersModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.SwitchTransformersModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SwitchTransformersModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SwitchTransformersModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SwitchTransformersModel.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.SwitchTransformersModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.SwitchTransformersModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1272",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.Seq2SeqMoEModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig"
>SwitchTransformersConfig</a>) and inputs.</p>
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
<p><strong>decoder_router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_logits=True</code> is passed or when <code>config.add_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Router logits of the decoder model, useful to compute the auxiliary loss for Mixture of Experts models.</p>
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
<li>
<p><strong>encoder_router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_logits=True</code> is passed or when <code>config.add_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
modules.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.modeling_outputs.Seq2SeqMoEModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),K=new Ut({props:{$$slots:{default:[on]},$$scope:{ctx:v}}}),ee=new mo({props:{anchor:"transformers.SwitchTransformersModel.forward.example",$$slots:{default:[nn]},$$scope:{ctx:v}}}),Se=new Re({props:{title:"SwitchTransformersForConditionalGeneration",local:"transformers.SwitchTransformersForConditionalGeneration",headingTag:"h2"}}),xe=new H({props:{name:"class transformers.SwitchTransformersForConditionalGeneration",anchor:"transformers.SwitchTransformersForConditionalGeneration",parameters:[{name:"config",val:": SwitchTransformersConfig"}],parametersDescription:[{anchor:"transformers.SwitchTransformersForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig">SwitchTransformersConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1436"}}),Ce=new H({props:{name:"forward",anchor:"transformers.SwitchTransformersForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = True"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. SWITCH_TRANSFORMERS is a model with relative position
embeddings so you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./switch_transformers#training">SWITCH_TRANSFORMERS
Training</a>.`,name:"input_ids"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>SWITCH_TRANSFORMERS uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./switch_transformers#training">SWITCH_TRANSFORMERS
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[-100, 0, ..., config.vocab_size - 1]</code>. All labels set to <code>-100</code> are ignored (masked), the loss is only computed for
labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1484",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.Seq2SeqMoEOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig"
>SwitchTransformersConfig</a>) and inputs.</p>
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
<p><strong>decoder_router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_logits=True</code> is passed or when <code>config.add_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Router logits of the decoder model, useful to compute the auxiliary loss for Mixture of Experts models.</p>
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
<li>
<p><strong>encoder_router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_logits=True</code> is passed or when <code>config.add_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Router logits of the encoder model, useful to compute the auxiliary loss and z_loss for Mixture of Experts
models.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.modeling_outputs.Seq2SeqMoEOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),te=new Ut({props:{$$slots:{default:[sn]},$$scope:{ctx:v}}}),oe=new mo({props:{anchor:"transformers.SwitchTransformersForConditionalGeneration.forward.example",$$slots:{default:[rn]},$$scope:{ctx:v}}}),ze=new Re({props:{title:"SwitchTransformersEncoderModel",local:"transformers.SwitchTransformersEncoderModel",headingTag:"h2"}}),Ue=new H({props:{name:"class transformers.SwitchTransformersEncoderModel",anchor:"transformers.SwitchTransformersEncoderModel",parameters:[{name:"config",val:": SwitchTransformersConfig"}],parametersDescription:[{anchor:"transformers.SwitchTransformersEncoderModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig">SwitchTransformersConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1720"}}),Je=new H({props:{name:"forward",anchor:"transformers.SwitchTransformersEncoderModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = True"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.SwitchTransformersEncoderModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. SWITCH_TRANSFORMERS is a model with relative position
embeddings so you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./switch_transformers#training">SWITCH_TRANSFORMERS
Training</a>.`,name:"input_ids"},{anchor:"transformers.SwitchTransformersEncoderModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SwitchTransformersEncoderModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SwitchTransformersEncoderModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SwitchTransformersEncoderModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SwitchTransformersEncoderModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SwitchTransformersEncoderModel.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.SwitchTransformersEncoderModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L1760",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoEModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/switch_transformers#transformers.SwitchTransformersConfig"
>SwitchTransformersConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
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
<p><strong>router_probs</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_probs=True</code> and <code>config.add_router_probs=True</code> is passed or when <code>config.output_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Raw router probabilities that are computed by MoE routers, these terms are used to compute the auxiliary
loss and the z_loss for Mixture of Experts models.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.modeling_outputs.MoEModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ne=new Ut({props:{$$slots:{default:[an]},$$scope:{ctx:v}}}),se=new mo({props:{anchor:"transformers.SwitchTransformersEncoderModel.forward.example",$$slots:{default:[dn]},$$scope:{ctx:v}}}),je=new Qo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/switch_transformers.md"}}),{c(){t=p("meta"),u=r(),o=p("p"),c=r(),M=p("p"),M.innerHTML=d,m=r(),k=p("div"),k.innerHTML=rt,ie=r(),g(E.$$.fragment),it=r(),de=p("p"),de.innerHTML=fo,dt=r(),le=p("p"),le.innerHTML=go,lt=r(),g(O.$$.fragment),ct=r(),ce=p("p"),ce.innerHTML=_o,pt=r(),g(Y.$$.fragment),ht=r(),pe=p("p"),pe.innerHTML=To,mt=r(),he=p("p"),he.innerHTML=wo,ut=r(),g(me.$$.fragment),ft=r(),g(ue.$$.fragment),gt=r(),q=p("div"),g(fe.$$.fragment),Jt=r(),Ie=p("p"),Ie.innerHTML=yo,jt=r(),He=p("p"),He.innerHTML=bo,_t=r(),g(ge.$$.fragment),Tt=r(),$=p("div"),g(_e.$$.fragment),Ft=r(),qe=p("p"),qe.textContent=Mo,Rt=r(),Ge=p("p"),Ge.innerHTML=vo,Zt=r(),D=p("div"),g(Te.$$.fragment),It=r(),We=p("p"),We.textContent=ko,Ht=r(),B=p("div"),g(we.$$.fragment),qt=r(),Ne=p("p"),Ne.innerHTML=$o,Gt=r(),Ee=p("p"),Ee.innerHTML=So,wt=r(),g(ye.$$.fragment),yt=r(),G=p("div"),g(be.$$.fragment),Wt=r(),Be=p("p"),Be.textContent=xo,Nt=r(),F=p("div"),g(Me.$$.fragment),Et=r(),Xe=p("p"),Xe.textContent=Co,Bt=r(),Ve=p("p"),Ve.innerHTML=zo,Xt=r(),Le=p("p"),Le.textContent=Uo,bt=r(),g(ve.$$.fragment),Mt=r(),S=p("div"),g(ke.$$.fragment),Vt=r(),Pe=p("p"),Pe.textContent=Jo,Lt=r(),Ae=p("p"),Ae.innerHTML=jo,Pt=r(),Qe=p("p"),Qe.innerHTML=Fo,At=r(),R=p("div"),g($e.$$.fragment),Qt=r(),Oe=p("p"),Oe.innerHTML=Ro,Ot=r(),g(K.$$.fragment),Yt=r(),g(ee.$$.fragment),vt=r(),g(Se.$$.fragment),kt=r(),x=p("div"),g(xe.$$.fragment),Dt=r(),Ye=p("p"),Ye.innerHTML=Zo,Kt=r(),De=p("p"),De.innerHTML=Io,eo=r(),Ke=p("p"),Ke.innerHTML=Ho,to=r(),Z=p("div"),g(Ce.$$.fragment),oo=r(),et=p("p"),et.innerHTML=qo,no=r(),g(te.$$.fragment),so=r(),g(oe.$$.fragment),$t=r(),g(ze.$$.fragment),St=r(),C=p("div"),g(Ue.$$.fragment),ro=r(),tt=p("p"),tt.textContent=Go,ao=r(),ot=p("p"),ot.innerHTML=Wo,io=r(),nt=p("p"),nt.innerHTML=No,lo=r(),I=p("div"),g(Je.$$.fragment),co=r(),st=p("p"),st.innerHTML=Eo,po=r(),g(ne.$$.fragment),ho=r(),g(se.$$.fragment),xt=r(),g(je.$$.fragment),Ct=r(),at=p("p"),this.h()},l(e){const n=Po("svelte-u9bgzb",document.head);t=h(n,"META",{name:!0,content:!0}),n.forEach(s),u=a(e),o=h(e,"P",{}),z(o).forEach(s),c=a(e),M=h(e,"P",{"data-svelte-h":!0}),f(M)!=="svelte-1coruhc"&&(M.innerHTML=d),m=a(e),k=h(e,"DIV",{style:!0,"data-svelte-h":!0}),f(k)!=="svelte-wa5t4p"&&(k.innerHTML=rt),ie=a(e),_(E.$$.fragment,e),it=a(e),de=h(e,"P",{"data-svelte-h":!0}),f(de)!=="svelte-14rcnp2"&&(de.innerHTML=fo),dt=a(e),le=h(e,"P",{"data-svelte-h":!0}),f(le)!=="svelte-zt6vxf"&&(le.innerHTML=go),lt=a(e),_(O.$$.fragment,e),ct=a(e),ce=h(e,"P",{"data-svelte-h":!0}),f(ce)!=="svelte-1o85lxv"&&(ce.innerHTML=_o),pt=a(e),_(Y.$$.fragment,e),ht=a(e),pe=h(e,"P",{"data-svelte-h":!0}),f(pe)!=="svelte-nf5ooi"&&(pe.innerHTML=To),mt=a(e),he=h(e,"P",{"data-svelte-h":!0}),f(he)!=="svelte-lkfx3x"&&(he.innerHTML=wo),ut=a(e),_(me.$$.fragment,e),ft=a(e),_(ue.$$.fragment,e),gt=a(e),q=h(e,"DIV",{class:!0});var P=z(q);_(fe.$$.fragment,P),Jt=a(P),Ie=h(P,"P",{"data-svelte-h":!0}),f(Ie)!=="svelte-10yvj5a"&&(Ie.innerHTML=yo),jt=a(P),He=h(P,"P",{"data-svelte-h":!0}),f(He)!=="svelte-1ek1ss9"&&(He.innerHTML=bo),P.forEach(s),_t=a(e),_(ge.$$.fragment,e),Tt=a(e),$=h(e,"DIV",{class:!0});var J=z($);_(_e.$$.fragment,J),Ft=a(J),qe=h(J,"P",{"data-svelte-h":!0}),f(qe)!=="svelte-12fm05d"&&(qe.textContent=Mo),Rt=a(J),Ge=h(J,"P",{"data-svelte-h":!0}),f(Ge)!=="svelte-3rdijk"&&(Ge.innerHTML=vo),Zt=a(J),D=h(J,"DIV",{class:!0});var Fe=z(D);_(Te.$$.fragment,Fe),It=a(Fe),We=h(Fe,"P",{"data-svelte-h":!0}),f(We)!=="svelte-jnjz7k"&&(We.textContent=ko),Fe.forEach(s),Ht=a(J),B=h(J,"DIV",{class:!0});var A=z(B);_(we.$$.fragment,A),qt=a(A),Ne=h(A,"P",{"data-svelte-h":!0}),f(Ne)!=="svelte-wi3ols"&&(Ne.innerHTML=$o),Gt=a(A),Ee=h(A,"P",{"data-svelte-h":!0}),f(Ee)!=="svelte-1xte4m0"&&(Ee.innerHTML=So),A.forEach(s),J.forEach(s),wt=a(e),_(ye.$$.fragment,e),yt=a(e),G=h(e,"DIV",{class:!0});var Q=z(G);_(be.$$.fragment,Q),Wt=a(Q),Be=h(Q,"P",{"data-svelte-h":!0}),f(Be)!=="svelte-1avy50b"&&(Be.textContent=xo),Nt=a(Q),F=h(Q,"DIV",{class:!0});var W=z(F);_(Me.$$.fragment,W),Et=a(W),Xe=h(W,"P",{"data-svelte-h":!0}),f(Xe)!=="svelte-a1z96i"&&(Xe.textContent=Co),Bt=a(W),Ve=h(W,"P",{"data-svelte-h":!0}),f(Ve)!=="svelte-1mybtpr"&&(Ve.innerHTML=zo),Xt=a(W),Le=h(W,"P",{"data-svelte-h":!0}),f(Le)!=="svelte-1n21hyz"&&(Le.textContent=Uo),W.forEach(s),Q.forEach(s),bt=a(e),_(ve.$$.fragment,e),Mt=a(e),S=h(e,"DIV",{class:!0});var j=z(S);_(ke.$$.fragment,j),Vt=a(j),Pe=h(j,"P",{"data-svelte-h":!0}),f(Pe)!=="svelte-wxp7du"&&(Pe.textContent=Jo),Lt=a(j),Ae=h(j,"P",{"data-svelte-h":!0}),f(Ae)!=="svelte-q52n56"&&(Ae.innerHTML=jo),Pt=a(j),Qe=h(j,"P",{"data-svelte-h":!0}),f(Qe)!=="svelte-hswkmf"&&(Qe.innerHTML=Fo),At=a(j),R=h(j,"DIV",{class:!0});var N=z(R);_($e.$$.fragment,N),Qt=a(N),Oe=h(N,"P",{"data-svelte-h":!0}),f(Oe)!=="svelte-1evgxx8"&&(Oe.innerHTML=Ro),Ot=a(N),_(K.$$.fragment,N),Yt=a(N),_(ee.$$.fragment,N),N.forEach(s),j.forEach(s),vt=a(e),_(Se.$$.fragment,e),kt=a(e),x=h(e,"DIV",{class:!0});var X=z(x);_(xe.$$.fragment,X),Dt=a(X),Ye=h(X,"P",{"data-svelte-h":!0}),f(Ye)!=="svelte-yohvjb"&&(Ye.innerHTML=Zo),Kt=a(X),De=h(X,"P",{"data-svelte-h":!0}),f(De)!=="svelte-q52n56"&&(De.innerHTML=Io),eo=a(X),Ke=h(X,"P",{"data-svelte-h":!0}),f(Ke)!=="svelte-hswkmf"&&(Ke.innerHTML=Ho),to=a(X),Z=h(X,"DIV",{class:!0});var re=z(Z);_(Ce.$$.fragment,re),oo=a(re),et=h(re,"P",{"data-svelte-h":!0}),f(et)!=="svelte-1w64ada"&&(et.innerHTML=qo),no=a(re),_(te.$$.fragment,re),so=a(re),_(oe.$$.fragment,re),re.forEach(s),X.forEach(s),$t=a(e),_(ze.$$.fragment,e),St=a(e),C=h(e,"DIV",{class:!0});var V=z(C);_(Ue.$$.fragment,V),ro=a(V),tt=h(V,"P",{"data-svelte-h":!0}),f(tt)!=="svelte-15ds7q4"&&(tt.textContent=Go),ao=a(V),ot=h(V,"P",{"data-svelte-h":!0}),f(ot)!=="svelte-q52n56"&&(ot.innerHTML=Wo),io=a(V),nt=h(V,"P",{"data-svelte-h":!0}),f(nt)!=="svelte-hswkmf"&&(nt.innerHTML=No),lo=a(V),I=h(V,"DIV",{class:!0});var ae=z(I);_(Je.$$.fragment,ae),co=a(ae),st=h(ae,"P",{"data-svelte-h":!0}),f(st)!=="svelte-1f27zg2"&&(st.innerHTML=Eo),po=a(ae),_(ne.$$.fragment,ae),ho=a(ae),_(se.$$.fragment,ae),ae.forEach(s),V.forEach(s),xt=a(e),_(je.$$.fragment,e),Ct=a(e),at=h(e,"P",{}),z(at).forEach(s),this.h()},h(){U(t,"name","hf:doc:metadata"),U(t,"content",cn),Ao(k,"float","right"),U(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){i(document.head,t),l(e,u,n),l(e,o,n),l(e,c,n),l(e,M,n),l(e,m,n),l(e,k,n),l(e,ie,n),T(E,e,n),l(e,it,n),l(e,de,n),l(e,dt,n),l(e,le,n),l(e,lt,n),T(O,e,n),l(e,ct,n),l(e,ce,n),l(e,pt,n),T(Y,e,n),l(e,ht,n),l(e,pe,n),l(e,mt,n),l(e,he,n),l(e,ut,n),T(me,e,n),l(e,ft,n),T(ue,e,n),l(e,gt,n),l(e,q,n),T(fe,q,null),i(q,Jt),i(q,Ie),i(q,jt),i(q,He),l(e,_t,n),T(ge,e,n),l(e,Tt,n),l(e,$,n),T(_e,$,null),i($,Ft),i($,qe),i($,Rt),i($,Ge),i($,Zt),i($,D),T(Te,D,null),i(D,It),i(D,We),i($,Ht),i($,B),T(we,B,null),i(B,qt),i(B,Ne),i(B,Gt),i(B,Ee),l(e,wt,n),T(ye,e,n),l(e,yt,n),l(e,G,n),T(be,G,null),i(G,Wt),i(G,Be),i(G,Nt),i(G,F),T(Me,F,null),i(F,Et),i(F,Xe),i(F,Bt),i(F,Ve),i(F,Xt),i(F,Le),l(e,bt,n),T(ve,e,n),l(e,Mt,n),l(e,S,n),T(ke,S,null),i(S,Vt),i(S,Pe),i(S,Lt),i(S,Ae),i(S,Pt),i(S,Qe),i(S,At),i(S,R),T($e,R,null),i(R,Qt),i(R,Oe),i(R,Ot),T(K,R,null),i(R,Yt),T(ee,R,null),l(e,vt,n),T(Se,e,n),l(e,kt,n),l(e,x,n),T(xe,x,null),i(x,Dt),i(x,Ye),i(x,Kt),i(x,De),i(x,eo),i(x,Ke),i(x,to),i(x,Z),T(Ce,Z,null),i(Z,oo),i(Z,et),i(Z,no),T(te,Z,null),i(Z,so),T(oe,Z,null),l(e,$t,n),T(ze,e,n),l(e,St,n),l(e,C,n),T(Ue,C,null),i(C,ro),i(C,tt),i(C,ao),i(C,ot),i(C,io),i(C,nt),i(C,lo),i(C,I),T(Je,I,null),i(I,co),i(I,st),i(I,po),T(ne,I,null),i(I,ho),T(se,I,null),l(e,xt,n),T(je,e,n),l(e,Ct,n),l(e,at,n),zt=!0},p(e,[n]){const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),O.$set(P);const J={};n&2&&(J.$$scope={dirty:n,ctx:e}),Y.$set(J);const Fe={};n&2&&(Fe.$$scope={dirty:n,ctx:e}),K.$set(Fe);const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),ee.$set(A);const Q={};n&2&&(Q.$$scope={dirty:n,ctx:e}),te.$set(Q);const W={};n&2&&(W.$$scope={dirty:n,ctx:e}),oe.$set(W);const j={};n&2&&(j.$$scope={dirty:n,ctx:e}),ne.$set(j);const N={};n&2&&(N.$$scope={dirty:n,ctx:e}),se.$set(N)},i(e){zt||(w(E.$$.fragment,e),w(O.$$.fragment,e),w(Y.$$.fragment,e),w(me.$$.fragment,e),w(ue.$$.fragment,e),w(fe.$$.fragment,e),w(ge.$$.fragment,e),w(_e.$$.fragment,e),w(Te.$$.fragment,e),w(we.$$.fragment,e),w(ye.$$.fragment,e),w(be.$$.fragment,e),w(Me.$$.fragment,e),w(ve.$$.fragment,e),w(ke.$$.fragment,e),w($e.$$.fragment,e),w(K.$$.fragment,e),w(ee.$$.fragment,e),w(Se.$$.fragment,e),w(xe.$$.fragment,e),w(Ce.$$.fragment,e),w(te.$$.fragment,e),w(oe.$$.fragment,e),w(ze.$$.fragment,e),w(Ue.$$.fragment,e),w(Je.$$.fragment,e),w(ne.$$.fragment,e),w(se.$$.fragment,e),w(je.$$.fragment,e),zt=!0)},o(e){y(E.$$.fragment,e),y(O.$$.fragment,e),y(Y.$$.fragment,e),y(me.$$.fragment,e),y(ue.$$.fragment,e),y(fe.$$.fragment,e),y(ge.$$.fragment,e),y(_e.$$.fragment,e),y(Te.$$.fragment,e),y(we.$$.fragment,e),y(ye.$$.fragment,e),y(be.$$.fragment,e),y(Me.$$.fragment,e),y(ve.$$.fragment,e),y(ke.$$.fragment,e),y($e.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(Se.$$.fragment,e),y(xe.$$.fragment,e),y(Ce.$$.fragment,e),y(te.$$.fragment,e),y(oe.$$.fragment,e),y(ze.$$.fragment,e),y(Ue.$$.fragment,e),y(Je.$$.fragment,e),y(ne.$$.fragment,e),y(se.$$.fragment,e),y(je.$$.fragment,e),zt=!1},d(e){e&&(s(u),s(o),s(c),s(M),s(m),s(k),s(ie),s(it),s(de),s(dt),s(le),s(lt),s(ct),s(ce),s(pt),s(ht),s(pe),s(mt),s(he),s(ut),s(ft),s(gt),s(q),s(_t),s(Tt),s($),s(wt),s(yt),s(G),s(bt),s(Mt),s(S),s(vt),s(kt),s(x),s($t),s(St),s(C),s(xt),s(Ct),s(at)),s(t),b(E,e),b(O,e),b(Y,e),b(me,e),b(ue,e),b(fe),b(ge,e),b(_e),b(Te),b(we),b(ye,e),b(be),b(Me),b(ve,e),b(ke),b($e),b(K),b(ee),b(Se,e),b(xe),b(Ce),b(te),b(oe),b(ze,e),b(Ue),b(Je),b(ne),b(se),b(je,e)}}}const cn='{"title":"Switch Transformers","local":"switch-transformers","sections":[{"title":"SwitchTransformersConfig","local":"transformers.SwitchTransformersConfig","sections":[],"depth":2},{"title":"SwitchTransformersTop1Router","local":"transformers.SwitchTransformersTop1Router","sections":[],"depth":2},{"title":"SwitchTransformersSparseMLP","local":"transformers.SwitchTransformersSparseMLP","sections":[],"depth":2},{"title":"SwitchTransformersModel","local":"transformers.SwitchTransformersModel","sections":[],"depth":2},{"title":"SwitchTransformersForConditionalGeneration","local":"transformers.SwitchTransformersForConditionalGeneration","sections":[],"depth":2},{"title":"SwitchTransformersEncoderModel","local":"transformers.SwitchTransformersEncoderModel","sections":[],"depth":2}],"depth":1}';function pn(v){return Xo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class yn extends Vo{constructor(t){super(),Lo(this,t,pn,ln,Bo,{})}}export{yn as component};
