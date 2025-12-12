import{s as Rs,o as Ss,n as j}from"../chunks/scheduler.18a86fab.js";import{S as Xs,i as Qs,g as c,s as a,r as u,A as Es,h as p,f as d,c as r,j as v,x as k,u as f,k as $,l as Ls,y as s,a as m,v as g,d as _,t as T,w as b}from"../chunks/index.98837b22.js";import{T as yt}from"../chunks/Tip.77304350.js";import{D as J}from"../chunks/Docstring.a1ef7999.js";import{C as K}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as kt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as O,E as As}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Ps,a as Dn}from"../chunks/HfOption.6641485e.js";function Ys(w){let t,h="Click on the T5 models in the right sidebar for more examples of how to apply T5 to different language tasks.";return{c(){t=c("p"),t.textContent=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-t4ran1"&&(t.textContent=h)},m(o,l){m(o,t,l)},p:j,d(o){o&&d(t)}}}function Ds(w){let t,h;return t=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0MnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyZ29vZ2xlLXQ1JTJGdDUtYmFzZSUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZSUzRDAlMEEpJTBBcGlwZWxpbmUoJTIydHJhbnNsYXRlJTIwRW5nbGlzaCUyMHRvJTIwRnJlbmNoJTNBJTIwVGhlJTIwd2VhdGhlciUyMGlzJTIwbmljZSUyMHRvZGF5LiUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text2text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;google-t5/t5-base&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;translate English to French: The weather is nice today.&quot;</span>)`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,l){g(t,o,l),h=!0},p:j,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){T(t.$$.fragment,o),h=!1},d(o){b(t,o)}}}function Os(w){let t,h;return t=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZS10NSUyRnQ1LWJhc2UlMjIlMEElMjAlMjAlMjAlMjApJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXEyU2VxTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZS10NSUyRnQ1LWJhc2UlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUwQSUyMCUyMCUyMCUyMCklMEElMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIydHJhbnNsYXRlJTIwRW5nbGlzaCUyMHRvJTIwRnJlbmNoJTNBJTIwVGhlJTIwd2VhdGhlciUyMGlzJTIwbmljZSUyMHRvZGF5LiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google-t5/t5-base&quot;</span>
    )
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;google-t5/t5-base&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
    )

input_ids = tokenizer(<span class="hljs-string">&quot;translate English to French: The weather is nice today.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,l){g(t,o,l),h=!0},p:j,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){T(t.$$.fragment,o),h=!1},d(o){b(t,o)}}}function Ks(w){let t,h;return t=new K({props:{code:"ZWNobyUyMC1lJTIwJTIydHJhbnNsYXRlJTIwRW5nbGlzaCUyMHRvJTIwRnJlbmNoJTNBJTIwVGhlJTIwd2VhdGhlciUyMGlzJTIwbmljZSUyMHRvZGF5LiUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQydGV4dC1nZW5lcmF0aW9uJTIwLS1tb2RlbCUyMGdvb2dsZS10NSUyRnQ1LWJhc2UlMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;translate English to French: The weather is nice today.&quot;</span> | transformers run --task text2text-generation --model google-t5/t5-base --device 0',wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,l){g(t,o,l),h=!0},p:j,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){T(t.$$.fragment,o),h=!1},d(o){b(t,o)}}}function ea(w){let t,h,o,l,M,n;return t=new Dn({props:{id:"usage",option:"Pipeline",$$slots:{default:[Ds]},$$scope:{ctx:w}}}),o=new Dn({props:{id:"usage",option:"AutoModel",$$slots:{default:[Os]},$$scope:{ctx:w}}}),M=new Dn({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Ks]},$$scope:{ctx:w}}}),{c(){u(t.$$.fragment),h=a(),u(o.$$.fragment),l=a(),u(M.$$.fragment)},l(y){f(t.$$.fragment,y),h=r(y),f(o.$$.fragment,y),l=r(y),f(M.$$.fragment,y)},m(y,z){g(t,y,z),m(y,h,z),g(o,y,z),m(y,l,z),g(M,y,z),n=!0},p(y,z){const ro={};z&2&&(ro.$$scope={dirty:z,ctx:y}),t.$set(ro);const je={};z&2&&(je.$$scope={dirty:z,ctx:y}),o.$set(je);const ee={};z&2&&(ee.$$scope={dirty:z,ctx:y}),M.$set(ee)},i(y){n||(_(t.$$.fragment,y),_(o.$$.fragment,y),_(M.$$.fragment,y),n=!0)},o(y){T(t.$$.fragment,y),T(o.$$.fragment,y),T(M.$$.fragment,y),n=!1},d(y){y&&(d(h),d(l)),b(t,y),b(o,y),b(M,y)}}}function ta(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,l){m(o,t,l)},p:j,d(o){o&&d(t)}}}function oa(w){let t,h="Example:",o,l,M;return l=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBUNU1vZGVsJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLXQ1JTJGdDUtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBUNU1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtdDUlMkZ0NS1zbWFsbCUyMiklMEElMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIyU3R1ZGllcyUyMGhhdmUlMjBiZWVuJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEFkZWNvZGVyX2lucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJTdHVkaWVzJTIwc2hvdyUyMHRoYXQlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBJTBBJTIzJTIwcHJlcHJvY2VzcyUzQSUyMFByZXBlbmQlMjBkZWNvZGVyX2lucHV0X2lkcyUyMHdpdGglMjBzdGFydCUyMHRva2VuJTIwd2hpY2glMjBpcyUyMHBhZCUyMHRva2VuJTIwZm9yJTIwVDVNb2RlbC4lMEElMjMlMjBUaGlzJTIwaXMlMjBub3QlMjBuZWVkZWQlMjBmb3IlMjB0b3JjaCdzJTIwVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMjBhcyUyMGl0JTIwZG9lcyUyMHRoaXMlMjBpbnRlcm5hbGx5JTIwdXNpbmclMjBsYWJlbHMlMjBhcmcuJTBBZGVjb2Rlcl9pbnB1dF9pZHMlMjAlM0QlMjBtb2RlbC5fc2hpZnRfcmlnaHQoZGVjb2Rlcl9pbnB1dF9pZHMpJTBBJTBBJTIzJTIwZm9yd2FyZCUyMHBhc3MlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RkZWNvZGVyX2lucHV0X2lkcyklMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5Model

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5Model.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># This is not needed for torch&#x27;s T5ForConditionalGeneration as it does this internally using labels arg.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = model._shift_right(decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),u(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),f(l.$$.fragment,n)},m(n,y){m(n,t,y),m(n,o,y),g(l,n,y),M=!0},p:j,i(n){M||(_(l.$$.fragment,n),M=!0)},o(n){T(l.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),b(l,n)}}}function na(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,l){m(o,t,l)},p:j,d(o){o&&d(t)}}}function sa(w){let t,h="Examples:",o,l,M;return l=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBUNUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS10NSUyRnQ1LXNtYWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS10NSUyRnQ1LXNtYWxsJTIyKSUwQSUwQSUyMyUyMHRyYWluaW5nJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMCUzQ2V4dHJhX2lkXzAlM0UlMjB3YWxrcyUyMGluJTIwJTNDZXh0cmFfaWRfMSUzRSUyMHBhcmslMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyJTNDZXh0cmFfaWRfMCUzRSUyMGN1dGUlMjBkb2clMjAlM0NleHRyYV9pZF8xJTNFJTIwdGhlJTIwJTNDZXh0cmFfaWRfMiUzRSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dF9pZHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTBBJTIzJTIwaW5mZXJlbmNlJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMnN1bW1hcml6ZSUzQSUyMHN0dWRpZXMlMjBoYXZlJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRfaWRzKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkpJTBBJTIzJTIwc3R1ZGllcyUyMGhhdmUlMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91Lg==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)

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
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># studies have shown that owning a dog is good for you.</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),u(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=r(n),f(l.$$.fragment,n)},m(n,y){m(n,t,y),m(n,o,y),g(l,n,y),M=!0},p:j,i(n){M||(_(l.$$.fragment,n),M=!0)},o(n){T(l.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),b(l,n)}}}function aa(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,l){m(o,t,l)},p:j,d(o){o&&d(t)}}}function ra(w){let t,h="Example:",o,l,M;return l=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBUNUVuY29kZXJNb2RlbCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS10NSUyRnQ1LXNtYWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwVDVFbmNvZGVyTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS10NSUyRnQ1LXNtYWxsJTIyKSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJTdHVkaWVzJTIwaGF2ZSUyMGJlZW4lMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSkuaW5wdXRfaWRzJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dF9pZHMpJTBBbGFzdF9oaWRkZW5fc3RhdGVzJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5EncoderModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5EncoderModel.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),u(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),f(l.$$.fragment,n)},m(n,y){m(n,t,y),m(n,o,y),g(l,n,y),M=!0},p:j,i(n){M||(_(l.$$.fragment,n),M=!0)},o(n){T(l.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),b(l,n)}}}function ia(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,l){m(o,t,l)},p:j,d(o){o&&d(t)}}}function da(w){let t,h="Example of single-label classification:",o,l,M;return l=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFQ1Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS10NSUyRnQ1LXNtYWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtdDUlMkZ0NS1zbWFsbCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtdDUlMkZ0NS1zbWFsbCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),u(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-ykxpe4"&&(t.textContent=h),o=r(n),f(l.$$.fragment,n)},m(n,y){m(n,t,y),m(n,o,y),g(l,n,y),M=!0},p:j,i(n){M||(_(l.$$.fragment,n),M=!0)},o(n){T(l.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),b(l,n)}}}function la(w){let t,h="Example of multi-label classification:",o,l,M;return l=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFQ1Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS10NSUyRnQ1LXNtYWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtdDUlMkZ0NS1zbWFsbCUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJnb29nbGUtdDUlMkZ0NS1zbWFsbCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5ForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google-t5/t5-small&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),u(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-1l8e32d"&&(t.textContent=h),o=r(n),f(l.$$.fragment,n)},m(n,y){m(n,t,y),m(n,o,y),g(l,n,y),M=!0},p:j,i(n){M||(_(l.$$.fragment,n),M=!0)},o(n){T(l.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),b(l,n)}}}function ca(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,l){m(o,t,l)},p:j,d(o){o&&d(t)}}}function pa(w){let t,h="Example:",o,l,M;return l=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBUNUZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS10NSUyRnQ1LXNtYWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwVDVGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtdDUlMkZ0NS1zbWFsbCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5ForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5ForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),u(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),f(l.$$.fragment,n)},m(n,y){m(n,t,y),m(n,o,y),g(l,n,y),M=!0},p:j,i(n){M||(_(l.$$.fragment,n),M=!0)},o(n){T(l.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),b(l,n)}}}function ma(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,l){m(o,t,l)},p:j,d(o){o&&d(t)}}}function ha(w){let t,h="Example:",o,l,M;return l=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBUNUZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtdDUlMkZ0NS1zbWFsbCUyMiklMEFtb2RlbCUyMCUzRCUyMFQ1Rm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS10NSUyRnQ1LXNtYWxsJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5ForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5ForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google-t5/t5-small&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),u(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),f(l.$$.fragment,n)},m(n,y){m(n,t,y),m(n,o,y),g(l,n,y),M=!0},p:j,i(n){M||(_(l.$$.fragment,n),M=!0)},o(n){T(l.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),b(l,n)}}}function ua(w){let t,h,o,l,M,n="<em>This model was released on 2019-10-23 and added to Hugging Face Transformers on 2020-11-16.</em>",y,z,ro='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',je,ee,co,Ce,On='<a href="https://huggingface.co/papers/1910.10683" rel="nofollow">T5</a> is a encoder-decoder transformer available in a range of sizes from 60M to 11B parameters. It is designed to handle a wide range of NLP tasks by treating them all as text-to-text problems. This eliminates the need for task-specific architectures because T5 converts every NLP task into a text generation task.',po,Fe,Kn="To formulate every task as text generation, each task is prepended with a task-specific prefix (e.g., translate English to German: …, summarize: …). This enables T5 to handle tasks like translation, summarization, question answering, and more.",mo,qe,es='You can find all official T5 checkpoints under the <a href="https://huggingface.co/collections/google/t5-release-65005e7c520f8d7b4d037918" rel="nofollow">T5</a> collection.',ho,ce,uo,Ie,ts='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and how to translate with T5 from the command line.',fo,pe,go,Ze,os='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',_o,We,ns='The example below uses <a href="../quantization/torchao">torchao</a> to only quantize the weights to int4.',To,Ne,bo,Ge,yo,Be,ss='<li>You can pad the encoder inputs on the left or right because T5 uses relative scalar embeddings.</li> <li>T5 models need a slightly higher learning rate than the default used in <a href="/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer">Trainer</a>. Typically, values of <code>1e-4</code> and <code>3e-4</code> work well for most tasks.</li>',ko,He,Mo,A,Ve,Ro,Mt,as=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Model">T5Model</a> or a <code>TFT5Model</code>. It is used to
instantiate a T5 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the T5
<a href="https://huggingface.co/google-t5/t5-small" rel="nofollow">google-t5/t5-small</a> architecture.`,So,wt,rs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,wo,Re,vo,U,Se,Xo,vt,is='Construct a T5 tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',Qo,$t,ds=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,Eo,te,Xe,Lo,zt,ls=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A sequence has the following format:`,Ao,Jt,cs="<li>single sequence: <code>X &lt;/s&gt;</code></li> <li>pair of sequences: <code>A &lt;/s&gt; B &lt;/s&gt;</code></li>",Po,me,Qe,Yo,Ut,ps=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Do,he,Ee,Oo,xt,ms=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
use of token type ids, therefore a list of zeros is returned.`,Ko,jt,Le,$o,Ae,zo,C,Pe,en,Ct,hs=`Construct a “fast” T5 tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models" rel="nofollow">Unigram</a>.`,tn,Ft,us=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,on,oe,Ye,nn,qt,fs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A sequence has the following format:`,sn,It,gs="<li>single sequence: <code>X &lt;/s&gt;</code></li> <li>pair of sequences: <code>A &lt;/s&gt; B &lt;/s&gt;</code></li>",an,ue,De,rn,Zt,_s=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
use of token type ids, therefore a list of zeros is returned.`,Jo,Oe,Uo,F,Ke,dn,Wt,Ts="The bare T5 Model outputting raw hidden-states without any specific head on top.",ln,Nt,bs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,cn,Gt,ys=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,pn,S,et,mn,Bt,ks='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Model">T5Model</a> forward method, overrides the <code>__call__</code> special method.',hn,fe,un,ge,xo,tt,jo,q,ot,fn,Ht,Ms="T5 Model with a <code>language modeling</code> head on top.",gn,Vt,ws=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_n,Rt,vs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Tn,X,nt,bn,St,$s='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForConditionalGeneration">T5ForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',yn,_e,kn,Te,Co,st,Fo,I,at,Mn,Xt,zs="The bare T5 Model outputting raw hidden-states without any specific head on top.",wn,Qt,Js=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vn,Et,Us=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$n,Q,rt,zn,Lt,xs='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5EncoderModel">T5EncoderModel</a> forward method, overrides the <code>__call__</code> special method.',Jn,be,Un,ye,qo,it,Io,Z,dt,xn,At,js=`T5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.`,jn,Pt,Cs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Cn,Yt,Fs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fn,G,lt,qn,Dt,qs='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForSequenceClassification">T5ForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',In,ke,Zn,Me,Wn,we,Zo,ct,Wo,W,pt,Nn,Ot,Is=`The T5 transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Gn,Kt,Zs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bn,eo,Ws=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Hn,E,mt,Vn,to,Ns='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForTokenClassification">T5ForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Rn,ve,Sn,$e,No,ht,Go,N,ut,Xn,oo,Gs=`The T5 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Qn,no,Bs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,En,so,Hs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ln,L,ft,An,ao,Vs='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForQuestionAnswering">T5ForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Pn,ze,Yn,Je,Bo,gt,Ho,io,Vo;return ee=new O({props:{title:"T5",local:"t5",headingTag:"h1"}}),ce=new yt({props:{warning:!1,$$slots:{default:[Ys]},$$scope:{ctx:w}}}),pe=new Ps({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[ea]},$$scope:{ctx:w}}}),Ne=new K({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMHRvcmNoYW8lMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBUb3JjaEFvQ29uZmlnJTJDJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBUb3JjaEFvQ29uZmlnKCUyMmludDRfd2VpZ2h0X29ubHklMjIlMkMlMjBncm91cF9zaXplJTNEMTI4KSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJnb29nbGUlMkZ0NS12MV8xLXhsJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRHF1YW50aXphdGlvbl9jb25maWclMEEpJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdDUtdjFfMS14bCUyMiklMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIydHJhbnNsYXRlJTIwRW5nbGlzaCUyMHRvJTIwRnJlbmNoJTNBJTIwVGhlJTIwd2VhdGhlciUyMGlzJTIwbmljZSUyMHRvZGF5LiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-comment"># pip install torchao</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TorchAoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = TorchAoConfig(<span class="hljs-string">&quot;int4_weight_only&quot;</span>, group_size=<span class="hljs-number">128</span>)
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;google/t5-v1_1-xl&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/t5-v1_1-xl&quot;</span>)
input_ids = tokenizer(<span class="hljs-string">&quot;translate English to French: The weather is nice today.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),Ge=new O({props:{title:"Notes",local:"notes",headingTag:"h2"}}),He=new O({props:{title:"T5Config",local:"transformers.T5Config",headingTag:"h2"}}),Ve=new J({props:{name:"class transformers.T5Config",anchor:"transformers.T5Config",parameters:[{name:"vocab_size",val:" = 32128"},{name:"d_model",val:" = 512"},{name:"d_kv",val:" = 64"},{name:"d_ff",val:" = 2048"},{name:"num_layers",val:" = 6"},{name:"num_decoder_layers",val:" = None"},{name:"num_heads",val:" = 8"},{name:"relative_attention_num_buckets",val:" = 32"},{name:"relative_attention_max_distance",val:" = 128"},{name:"dropout_rate",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-06"},{name:"initializer_factor",val:" = 1.0"},{name:"feed_forward_proj",val:" = 'relu'"},{name:"is_encoder_decoder",val:" = True"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"classifier_dropout",val:" = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.T5Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32128) &#x2014;
Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Model">T5Model</a> or <code>TFT5Model</code>.`,name:"vocab_size"},{anchor:"transformers.T5Config.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Size of the encoder layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.T5Config.d_kv",description:`<strong>d_kv</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Size of the key, query, value projections per attention head. The <code>inner_dim</code> of the projection layer will
be defined as <code>num_heads * d_kv</code>.`,name:"d_kv"},{anchor:"transformers.T5Config.d_ff",description:`<strong>d_ff</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Size of the intermediate feed forward layer in each <code>T5Block</code>.`,name:"d_ff"},{anchor:"transformers.T5Config.num_layers",description:`<strong>num_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_layers"},{anchor:"transformers.T5Config.num_decoder_layers",description:`<strong>num_decoder_layers</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Number of hidden layers in the Transformer decoder. Will use the same value as <code>num_layers</code> if not set.`,name:"num_decoder_layers"},{anchor:"transformers.T5Config.num_heads",description:`<strong>num_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_heads"},{anchor:"transformers.T5Config.relative_attention_num_buckets",description:`<strong>relative_attention_num_buckets</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of buckets to use for each attention layer.`,name:"relative_attention_num_buckets"},{anchor:"transformers.T5Config.relative_attention_max_distance",description:`<strong>relative_attention_max_distance</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The maximum distance of the longer sequences for the bucket separation.`,name:"relative_attention_max_distance"},{anchor:"transformers.T5Config.dropout_rate",description:`<strong>dropout_rate</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The ratio for all dropout layers.`,name:"dropout_rate"},{anchor:"transformers.T5Config.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.T5Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-6) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.T5Config.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"},{anchor:"transformers.T5Config.feed_forward_proj",description:`<strong>feed_forward_proj</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
Type of feed forward layer to be used. Should be one of <code>&quot;relu&quot;</code> or <code>&quot;gated-gelu&quot;</code>. T5v1.1 uses the
<code>&quot;gated-gelu&quot;</code> feed forward projection. Original T5 uses <code>&quot;relu&quot;</code>.`,name:"feed_forward_proj"},{anchor:"transformers.T5Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/configuration_t5.py#L27"}}),Re=new O({props:{title:"T5Tokenizer",local:"transformers.T5Tokenizer",headingTag:"h2"}}),Se=new J({props:{name:"class transformers.T5Tokenizer",anchor:"transformers.T5Tokenizer",parameters:[{name:"vocab_file",val:""},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"extra_ids",val:" = 100"},{name:"additional_special_tokens",val:" = None"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"legacy",val:" = None"},{name:"add_prefix_space",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.T5Tokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.T5Tokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.T5Tokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.T5Tokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.T5Tokenizer.extra_ids",description:`<strong>extra_ids</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
Add a number of extra ids added to the vocabulary for use as sentinels. These tokens are
accessible as &#x201C;<extra<em>id{%d}&gt;&#x201D; where &#x201D;{%d}&#x201D; is a number between 0 and extra_ids-1. These tokens can be
retrieved by calling get_sentinel_tokens method and token ids can be by calling get_sentinel_token_ids
method
additional_special_tokens (<code>list[str]</code>, <em>optional</em>):
Additional special tokens used by the tokenizer.</extra<em>`,name:"extra_ids"},{anchor:"transformers.T5Tokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.T5Tokenizer.legacy",description:`<strong>legacy</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not the <code>legacy</code> behaviour of the tokenizer should be used. Legacy is before the merge of #24622
and #25224 which includes fixes to properly handle tokens that appear after special tokens. A simple
example:</p>
<ul>
<li><code>legacy=True</code>:</li>
</ul>`,name:"legacy"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/tokenization_t5.py#L45"}}),Xe=new J({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.T5Tokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.T5Tokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.T5Tokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/tokenization_t5.py#L317",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Qe=new J({props:{name:"get_special_tokens_mask",anchor:"transformers.T5Tokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.T5Tokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.T5Tokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.T5Tokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/tokenization_t5.py#L248",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ee=new J({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.T5Tokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.T5Tokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.T5Tokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/tokenization_t5.py#L295",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Le=new J({props:{name:"save_vocabulary",anchor:"transformers.T5Tokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/tokenization_t5.py#L430"}}),Ae=new O({props:{title:"T5TokenizerFast",local:"transformers.T5TokenizerFast",headingTag:"h2"}}),Pe=new J({props:{name:"class transformers.T5TokenizerFast",anchor:"transformers.T5TokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"extra_ids",val:" = 100"},{name:"additional_special_tokens",val:" = None"},{name:"add_prefix_space",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.T5TokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.T5TokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.T5TokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.T5TokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.T5TokenizerFast.extra_ids",description:`<strong>extra_ids</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
Add a number of extra ids added to the vocabulary for use as sentinels. These tokens are accessible as
&#x201C;<extra<em>id{%d}&gt;&#x201D; where &#x201D;{%d}&#x201D; is a number between 0 and extra_ids-1. These tokens can be retrieved by
calling get_sentinel_tokens method and token ids can be by calling get_sentinel_token_ids method</extra<em>`,name:"extra_ids"},{anchor:"transformers.T5TokenizerFast.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"},{anchor:"transformers.T5TokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not the tokenizer should automatically add a prefix space`,name:"add_prefix_space"},{anchor:"transformers.T5TokenizerFast.from_slow",description:`<strong>from_slow</strong> (<code>book</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the tokenizer should be converted from a slow one. If <code>add_prefix_space</code> is set, this will be set to <code>True</code>.`,name:"from_slow"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/tokenization_t5_fast.py#L41"}}),Ye=new J({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.T5TokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.T5TokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.T5TokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/tokenization_t5_fast.py#L176",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),De=new J({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.T5TokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.T5TokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.T5TokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/tokenization_t5_fast.py#L202",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Oe=new O({props:{title:"T5Model",local:"transformers.T5Model",headingTag:"h2"}}),Ke=new J({props:{name:"class transformers.T5Model",anchor:"transformers.T5Model",parameters:[{name:"config",val:": T5Config"}],parametersDescription:[{anchor:"transformers.T5Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config">T5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L1298"}}),et=new J({props:{name:"forward",anchor:"transformers.T5Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.T5Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./t5#training">T5 Training</a>.`,name:"input_ids"},{anchor:"transformers.T5Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5Model.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>T5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./t5#training">T5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.T5Model.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.T5Model.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.T5Model.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.T5Model.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.T5Model.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.T5Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.T5Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5Model.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.T5Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.T5Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.T5Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.T5Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.T5Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L1384",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config"
>T5Config</a>) and inputs.</p>
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
`}}),fe=new yt({props:{$$slots:{default:[ta]},$$scope:{ctx:w}}}),ge=new kt({props:{anchor:"transformers.T5Model.forward.example",$$slots:{default:[oa]},$$scope:{ctx:w}}}),tt=new O({props:{title:"T5ForConditionalGeneration",local:"transformers.T5ForConditionalGeneration",headingTag:"h2"}}),ot=new J({props:{name:"class transformers.T5ForConditionalGeneration",anchor:"transformers.T5ForConditionalGeneration",parameters:[{name:"config",val:": T5Config"}],parametersDescription:[{anchor:"transformers.T5ForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config">T5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L1542"}}),nt=new J({props:{name:"forward",anchor:"transformers.T5ForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.T5ForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./t5#training">T5 Training</a>.`,name:"input_ids"},{anchor:"transformers.T5ForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5ForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>T5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./t5#training">T5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.T5ForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.T5ForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.T5ForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.T5ForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.T5ForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.T5ForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.T5ForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5ForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.T5ForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[-100, 0, ..., config.vocab_size - 1]</code>. All labels set to <code>-100</code> are ignored (masked), the loss is only computed for
labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.T5ForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.T5ForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.T5ForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.T5ForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.T5ForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L1626",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config"
>T5Config</a>) and inputs.</p>
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
`}}),_e=new yt({props:{$$slots:{default:[na]},$$scope:{ctx:w}}}),Te=new kt({props:{anchor:"transformers.T5ForConditionalGeneration.forward.example",$$slots:{default:[sa]},$$scope:{ctx:w}}}),st=new O({props:{title:"T5EncoderModel",local:"transformers.T5EncoderModel",headingTag:"h2"}}),at=new J({props:{name:"class transformers.T5EncoderModel",anchor:"transformers.T5EncoderModel",parameters:[{name:"config",val:": T5Config"}],parametersDescription:[{anchor:"transformers.T5EncoderModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config">T5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L1823"}}),rt=new J({props:{name:"forward",anchor:"transformers.T5EncoderModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.T5EncoderModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./t5#training">T5 Training</a>.`,name:"input_ids"},{anchor:"transformers.T5EncoderModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5EncoderModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.T5EncoderModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5EncoderModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.T5EncoderModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.T5EncoderModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L1895",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config"
>T5Config</a>) and inputs.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),be=new yt({props:{$$slots:{default:[aa]},$$scope:{ctx:w}}}),ye=new kt({props:{anchor:"transformers.T5EncoderModel.forward.example",$$slots:{default:[ra]},$$scope:{ctx:w}}}),it=new O({props:{title:"T5ForSequenceClassification",local:"transformers.T5ForSequenceClassification",headingTag:"h2"}}),dt=new J({props:{name:"class transformers.T5ForSequenceClassification",anchor:"transformers.T5ForSequenceClassification",parameters:[{name:"config",val:": T5Config"}],parametersDescription:[{anchor:"transformers.T5ForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config">T5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L1950"}}),lt=new J({props:{name:"forward",anchor:"transformers.T5ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.T5ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./t5#training">T5 Training</a>.`,name:"input_ids"},{anchor:"transformers.T5ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5ForSequenceClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>T5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./t5#training">T5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.T5ForSequenceClassification.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.T5ForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.T5ForSequenceClassification.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.T5ForSequenceClassification.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.T5ForSequenceClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.T5ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5ForSequenceClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.T5ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.T5ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.T5ForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.T5ForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.T5ForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L1964",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config"
>T5Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>label</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ke=new yt({props:{$$slots:{default:[ia]},$$scope:{ctx:w}}}),Me=new kt({props:{anchor:"transformers.T5ForSequenceClassification.forward.example",$$slots:{default:[da]},$$scope:{ctx:w}}}),we=new kt({props:{anchor:"transformers.T5ForSequenceClassification.forward.example-2",$$slots:{default:[la]},$$scope:{ctx:w}}}),ct=new O({props:{title:"T5ForTokenClassification",local:"transformers.T5ForTokenClassification",headingTag:"h2"}}),pt=new J({props:{name:"class transformers.T5ForTokenClassification",anchor:"transformers.T5ForTokenClassification",parameters:[{name:"config",val:": T5Config"}],parametersDescription:[{anchor:"transformers.T5ForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config">T5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L2113"}}),mt=new J({props:{name:"forward",anchor:"transformers.T5ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.T5ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./t5#training">T5 Training</a>.`,name:"input_ids"},{anchor:"transformers.T5ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5ForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.T5ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.T5ForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.T5ForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.T5ForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L2127",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config"
>T5Config</a>) and inputs.</p>
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
`}}),ve=new yt({props:{$$slots:{default:[ca]},$$scope:{ctx:w}}}),$e=new kt({props:{anchor:"transformers.T5ForTokenClassification.forward.example",$$slots:{default:[pa]},$$scope:{ctx:w}}}),ht=new O({props:{title:"T5ForQuestionAnswering",local:"transformers.T5ForQuestionAnswering",headingTag:"h2"}}),ut=new J({props:{name:"class transformers.T5ForQuestionAnswering",anchor:"transformers.T5ForQuestionAnswering",parameters:[{name:"config",val:": T5Config"}],parametersDescription:[{anchor:"transformers.T5ForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config">T5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L2187"}}),ft=new J({props:{name:"forward",anchor:"transformers.T5ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.T5ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./t5#training">T5 Training</a>.`,name:"input_ids"},{anchor:"transformers.T5ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5ForQuestionAnswering.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>T5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./t5#training">T5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.T5ForQuestionAnswering.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.T5ForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.T5ForQuestionAnswering.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.T5ForQuestionAnswering.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.T5ForQuestionAnswering.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.T5ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.T5ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.T5ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5ForQuestionAnswering.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.T5ForQuestionAnswering.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.T5ForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.T5ForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.T5ForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5/modeling_t5.py#L2233",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput"
>transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Config"
>T5Config</a>) and inputs.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput"
>transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ze=new yt({props:{$$slots:{default:[ma]},$$scope:{ctx:w}}}),Je=new kt({props:{anchor:"transformers.T5ForQuestionAnswering.forward.example",$$slots:{default:[ha]},$$scope:{ctx:w}}}),gt=new As({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/t5.md"}}),{c(){t=c("meta"),h=a(),o=c("p"),l=a(),M=c("p"),M.innerHTML=n,y=a(),z=c("div"),z.innerHTML=ro,je=a(),u(ee.$$.fragment),co=a(),Ce=c("p"),Ce.innerHTML=On,po=a(),Fe=c("p"),Fe.textContent=Kn,mo=a(),qe=c("p"),qe.innerHTML=es,ho=a(),u(ce.$$.fragment),uo=a(),Ie=c("p"),Ie.innerHTML=ts,fo=a(),u(pe.$$.fragment),go=a(),Ze=c("p"),Ze.innerHTML=os,_o=a(),We=c("p"),We.innerHTML=ns,To=a(),u(Ne.$$.fragment),bo=a(),u(Ge.$$.fragment),yo=a(),Be=c("ul"),Be.innerHTML=ss,ko=a(),u(He.$$.fragment),Mo=a(),A=c("div"),u(Ve.$$.fragment),Ro=a(),Mt=c("p"),Mt.innerHTML=as,So=a(),wt=c("p"),wt.innerHTML=rs,wo=a(),u(Re.$$.fragment),vo=a(),U=c("div"),u(Se.$$.fragment),Xo=a(),vt=c("p"),vt.innerHTML=is,Qo=a(),$t=c("p"),$t.innerHTML=ds,Eo=a(),te=c("div"),u(Xe.$$.fragment),Lo=a(),zt=c("p"),zt.textContent=ls,Ao=a(),Jt=c("ul"),Jt.innerHTML=cs,Po=a(),me=c("div"),u(Qe.$$.fragment),Yo=a(),Ut=c("p"),Ut.innerHTML=ps,Do=a(),he=c("div"),u(Ee.$$.fragment),Oo=a(),xt=c("p"),xt.textContent=ms,Ko=a(),jt=c("div"),u(Le.$$.fragment),$o=a(),u(Ae.$$.fragment),zo=a(),C=c("div"),u(Pe.$$.fragment),en=a(),Ct=c("p"),Ct.innerHTML=hs,tn=a(),Ft=c("p"),Ft.innerHTML=us,on=a(),oe=c("div"),u(Ye.$$.fragment),nn=a(),qt=c("p"),qt.textContent=fs,sn=a(),It=c("ul"),It.innerHTML=gs,an=a(),ue=c("div"),u(De.$$.fragment),rn=a(),Zt=c("p"),Zt.textContent=_s,Jo=a(),u(Oe.$$.fragment),Uo=a(),F=c("div"),u(Ke.$$.fragment),dn=a(),Wt=c("p"),Wt.textContent=Ts,ln=a(),Nt=c("p"),Nt.innerHTML=bs,cn=a(),Gt=c("p"),Gt.innerHTML=ys,pn=a(),S=c("div"),u(et.$$.fragment),mn=a(),Bt=c("p"),Bt.innerHTML=ks,hn=a(),u(fe.$$.fragment),un=a(),u(ge.$$.fragment),xo=a(),u(tt.$$.fragment),jo=a(),q=c("div"),u(ot.$$.fragment),fn=a(),Ht=c("p"),Ht.innerHTML=Ms,gn=a(),Vt=c("p"),Vt.innerHTML=ws,_n=a(),Rt=c("p"),Rt.innerHTML=vs,Tn=a(),X=c("div"),u(nt.$$.fragment),bn=a(),St=c("p"),St.innerHTML=$s,yn=a(),u(_e.$$.fragment),kn=a(),u(Te.$$.fragment),Co=a(),u(st.$$.fragment),Fo=a(),I=c("div"),u(at.$$.fragment),Mn=a(),Xt=c("p"),Xt.textContent=zs,wn=a(),Qt=c("p"),Qt.innerHTML=Js,vn=a(),Et=c("p"),Et.innerHTML=Us,$n=a(),Q=c("div"),u(rt.$$.fragment),zn=a(),Lt=c("p"),Lt.innerHTML=xs,Jn=a(),u(be.$$.fragment),Un=a(),u(ye.$$.fragment),qo=a(),u(it.$$.fragment),Io=a(),Z=c("div"),u(dt.$$.fragment),xn=a(),At=c("p"),At.textContent=js,jn=a(),Pt=c("p"),Pt.innerHTML=Cs,Cn=a(),Yt=c("p"),Yt.innerHTML=Fs,Fn=a(),G=c("div"),u(lt.$$.fragment),qn=a(),Dt=c("p"),Dt.innerHTML=qs,In=a(),u(ke.$$.fragment),Zn=a(),u(Me.$$.fragment),Wn=a(),u(we.$$.fragment),Zo=a(),u(ct.$$.fragment),Wo=a(),W=c("div"),u(pt.$$.fragment),Nn=a(),Ot=c("p"),Ot.textContent=Is,Gn=a(),Kt=c("p"),Kt.innerHTML=Zs,Bn=a(),eo=c("p"),eo.innerHTML=Ws,Hn=a(),E=c("div"),u(mt.$$.fragment),Vn=a(),to=c("p"),to.innerHTML=Ns,Rn=a(),u(ve.$$.fragment),Sn=a(),u($e.$$.fragment),No=a(),u(ht.$$.fragment),Go=a(),N=c("div"),u(ut.$$.fragment),Xn=a(),oo=c("p"),oo.innerHTML=Gs,Qn=a(),no=c("p"),no.innerHTML=Bs,En=a(),so=c("p"),so.innerHTML=Hs,Ln=a(),L=c("div"),u(ft.$$.fragment),An=a(),ao=c("p"),ao.innerHTML=Vs,Pn=a(),u(ze.$$.fragment),Yn=a(),u(Je.$$.fragment),Bo=a(),u(gt.$$.fragment),Ho=a(),io=c("p"),this.h()},l(e){const i=Es("svelte-u9bgzb",document.head);t=p(i,"META",{name:!0,content:!0}),i.forEach(d),h=r(e),o=p(e,"P",{}),v(o).forEach(d),l=r(e),M=p(e,"P",{"data-svelte-h":!0}),k(M)!=="svelte-1mlkrfx"&&(M.innerHTML=n),y=r(e),z=p(e,"DIV",{style:!0,"data-svelte-h":!0}),k(z)!=="svelte-wa5t4p"&&(z.innerHTML=ro),je=r(e),f(ee.$$.fragment,e),co=r(e),Ce=p(e,"P",{"data-svelte-h":!0}),k(Ce)!=="svelte-12r4ri0"&&(Ce.innerHTML=On),po=r(e),Fe=p(e,"P",{"data-svelte-h":!0}),k(Fe)!=="svelte-1igb8jp"&&(Fe.textContent=Kn),mo=r(e),qe=p(e,"P",{"data-svelte-h":!0}),k(qe)!=="svelte-14k49us"&&(qe.innerHTML=es),ho=r(e),f(ce.$$.fragment,e),uo=r(e),Ie=p(e,"P",{"data-svelte-h":!0}),k(Ie)!=="svelte-1fx803"&&(Ie.innerHTML=ts),fo=r(e),f(pe.$$.fragment,e),go=r(e),Ze=p(e,"P",{"data-svelte-h":!0}),k(Ze)!=="svelte-nf5ooi"&&(Ze.innerHTML=os),_o=r(e),We=p(e,"P",{"data-svelte-h":!0}),k(We)!=="svelte-w36i1c"&&(We.innerHTML=ns),To=r(e),f(Ne.$$.fragment,e),bo=r(e),f(Ge.$$.fragment,e),yo=r(e),Be=p(e,"UL",{"data-svelte-h":!0}),k(Be)!=="svelte-9oe2yj"&&(Be.innerHTML=ss),ko=r(e),f(He.$$.fragment,e),Mo=r(e),A=p(e,"DIV",{class:!0});var ie=v(A);f(Ve.$$.fragment,ie),Ro=r(ie),Mt=p(ie,"P",{"data-svelte-h":!0}),k(Mt)!=="svelte-ybbdg5"&&(Mt.innerHTML=as),So=r(ie),wt=p(ie,"P",{"data-svelte-h":!0}),k(wt)!=="svelte-1ek1ss9"&&(wt.innerHTML=rs),ie.forEach(d),wo=r(e),f(Re.$$.fragment,e),vo=r(e),U=p(e,"DIV",{class:!0});var x=v(U);f(Se.$$.fragment,x),Xo=r(x),vt=p(x,"P",{"data-svelte-h":!0}),k(vt)!=="svelte-pdnetr"&&(vt.innerHTML=is),Qo=r(x),$t=p(x,"P",{"data-svelte-h":!0}),k($t)!=="svelte-ntrhio"&&($t.innerHTML=ds),Eo=r(x),te=p(x,"DIV",{class:!0});var de=v(te);f(Xe.$$.fragment,de),Lo=r(de),zt=p(de,"P",{"data-svelte-h":!0}),k(zt)!=="svelte-1wjq39d"&&(zt.textContent=ls),Ao=r(de),Jt=p(de,"UL",{"data-svelte-h":!0}),k(Jt)!=="svelte-8gh3n2"&&(Jt.innerHTML=cs),de.forEach(d),Po=r(x),me=p(x,"DIV",{class:!0});var _t=v(me);f(Qe.$$.fragment,_t),Yo=r(_t),Ut=p(_t,"P",{"data-svelte-h":!0}),k(Ut)!=="svelte-1f4f5kp"&&(Ut.innerHTML=ps),_t.forEach(d),Do=r(x),he=p(x,"DIV",{class:!0});var Tt=v(he);f(Ee.$$.fragment,Tt),Oo=r(Tt),xt=p(Tt,"P",{"data-svelte-h":!0}),k(xt)!=="svelte-fl5ab0"&&(xt.textContent=ms),Tt.forEach(d),Ko=r(x),jt=p(x,"DIV",{class:!0});var lo=v(jt);f(Le.$$.fragment,lo),lo.forEach(d),x.forEach(d),$o=r(e),f(Ae.$$.fragment,e),zo=r(e),C=p(e,"DIV",{class:!0});var B=v(C);f(Pe.$$.fragment,B),en=r(B),Ct=p(B,"P",{"data-svelte-h":!0}),k(Ct)!=="svelte-gc3mz8"&&(Ct.innerHTML=hs),tn=r(B),Ft=p(B,"P",{"data-svelte-h":!0}),k(Ft)!=="svelte-gxzj9w"&&(Ft.innerHTML=us),on=r(B),oe=p(B,"DIV",{class:!0});var le=v(oe);f(Ye.$$.fragment,le),nn=r(le),qt=p(le,"P",{"data-svelte-h":!0}),k(qt)!=="svelte-1wjq39d"&&(qt.textContent=fs),sn=r(le),It=p(le,"UL",{"data-svelte-h":!0}),k(It)!=="svelte-8gh3n2"&&(It.innerHTML=gs),le.forEach(d),an=r(B),ue=p(B,"DIV",{class:!0});var bt=v(ue);f(De.$$.fragment,bt),rn=r(bt),Zt=p(bt,"P",{"data-svelte-h":!0}),k(Zt)!=="svelte-fl5ab0"&&(Zt.textContent=_s),bt.forEach(d),B.forEach(d),Jo=r(e),f(Oe.$$.fragment,e),Uo=r(e),F=p(e,"DIV",{class:!0});var H=v(F);f(Ke.$$.fragment,H),dn=r(H),Wt=p(H,"P",{"data-svelte-h":!0}),k(Wt)!=="svelte-1bi4id7"&&(Wt.textContent=Ts),ln=r(H),Nt=p(H,"P",{"data-svelte-h":!0}),k(Nt)!=="svelte-q52n56"&&(Nt.innerHTML=bs),cn=r(H),Gt=p(H,"P",{"data-svelte-h":!0}),k(Gt)!=="svelte-hswkmf"&&(Gt.innerHTML=ys),pn=r(H),S=p(H,"DIV",{class:!0});var P=v(S);f(et.$$.fragment,P),mn=r(P),Bt=p(P,"P",{"data-svelte-h":!0}),k(Bt)!=="svelte-1rqhp68"&&(Bt.innerHTML=ks),hn=r(P),f(fe.$$.fragment,P),un=r(P),f(ge.$$.fragment,P),P.forEach(d),H.forEach(d),xo=r(e),f(tt.$$.fragment,e),jo=r(e),q=p(e,"DIV",{class:!0});var V=v(q);f(ot.$$.fragment,V),fn=r(V),Ht=p(V,"P",{"data-svelte-h":!0}),k(Ht)!=="svelte-6hhvcz"&&(Ht.innerHTML=Ms),gn=r(V),Vt=p(V,"P",{"data-svelte-h":!0}),k(Vt)!=="svelte-q52n56"&&(Vt.innerHTML=ws),_n=r(V),Rt=p(V,"P",{"data-svelte-h":!0}),k(Rt)!=="svelte-hswkmf"&&(Rt.innerHTML=vs),Tn=r(V),X=p(V,"DIV",{class:!0});var Y=v(X);f(nt.$$.fragment,Y),bn=r(Y),St=p(Y,"P",{"data-svelte-h":!0}),k(St)!=="svelte-6i6us4"&&(St.innerHTML=$s),yn=r(Y),f(_e.$$.fragment,Y),kn=r(Y),f(Te.$$.fragment,Y),Y.forEach(d),V.forEach(d),Co=r(e),f(st.$$.fragment,e),Fo=r(e),I=p(e,"DIV",{class:!0});var R=v(I);f(at.$$.fragment,R),Mn=r(R),Xt=p(R,"P",{"data-svelte-h":!0}),k(Xt)!=="svelte-1bi4id7"&&(Xt.textContent=zs),wn=r(R),Qt=p(R,"P",{"data-svelte-h":!0}),k(Qt)!=="svelte-q52n56"&&(Qt.innerHTML=Js),vn=r(R),Et=p(R,"P",{"data-svelte-h":!0}),k(Et)!=="svelte-hswkmf"&&(Et.innerHTML=Us),$n=r(R),Q=p(R,"DIV",{class:!0});var D=v(Q);f(rt.$$.fragment,D),zn=r(D),Lt=p(D,"P",{"data-svelte-h":!0}),k(Lt)!=="svelte-1l2k6mk"&&(Lt.innerHTML=xs),Jn=r(D),f(be.$$.fragment,D),Un=r(D),f(ye.$$.fragment,D),D.forEach(d),R.forEach(d),qo=r(e),f(it.$$.fragment,e),Io=r(e),Z=p(e,"DIV",{class:!0});var ne=v(Z);f(dt.$$.fragment,ne),xn=r(ne),At=p(ne,"P",{"data-svelte-h":!0}),k(At)!=="svelte-14cghox"&&(At.textContent=js),jn=r(ne),Pt=p(ne,"P",{"data-svelte-h":!0}),k(Pt)!=="svelte-q52n56"&&(Pt.innerHTML=Cs),Cn=r(ne),Yt=p(ne,"P",{"data-svelte-h":!0}),k(Yt)!=="svelte-hswkmf"&&(Yt.innerHTML=Fs),Fn=r(ne),G=p(ne,"DIV",{class:!0});var se=v(G);f(lt.$$.fragment,se),qn=r(se),Dt=p(se,"P",{"data-svelte-h":!0}),k(Dt)!=="svelte-12tco70"&&(Dt.innerHTML=qs),In=r(se),f(ke.$$.fragment,se),Zn=r(se),f(Me.$$.fragment,se),Wn=r(se),f(we.$$.fragment,se),se.forEach(d),ne.forEach(d),Zo=r(e),f(ct.$$.fragment,e),Wo=r(e),W=p(e,"DIV",{class:!0});var ae=v(W);f(pt.$$.fragment,ae),Nn=r(ae),Ot=p(ae,"P",{"data-svelte-h":!0}),k(Ot)!=="svelte-19dlshh"&&(Ot.textContent=Is),Gn=r(ae),Kt=p(ae,"P",{"data-svelte-h":!0}),k(Kt)!=="svelte-q52n56"&&(Kt.innerHTML=Zs),Bn=r(ae),eo=p(ae,"P",{"data-svelte-h":!0}),k(eo)!=="svelte-hswkmf"&&(eo.innerHTML=Ws),Hn=r(ae),E=p(ae,"DIV",{class:!0});var Ue=v(E);f(mt.$$.fragment,Ue),Vn=r(Ue),to=p(Ue,"P",{"data-svelte-h":!0}),k(to)!=="svelte-b8ayr2"&&(to.innerHTML=Ns),Rn=r(Ue),f(ve.$$.fragment,Ue),Sn=r(Ue),f($e.$$.fragment,Ue),Ue.forEach(d),ae.forEach(d),No=r(e),f(ht.$$.fragment,e),Go=r(e),N=p(e,"DIV",{class:!0});var re=v(N);f(ut.$$.fragment,re),Xn=r(re),oo=p(re,"P",{"data-svelte-h":!0}),k(oo)!=="svelte-jarhw8"&&(oo.innerHTML=Gs),Qn=r(re),no=p(re,"P",{"data-svelte-h":!0}),k(no)!=="svelte-q52n56"&&(no.innerHTML=Bs),En=r(re),so=p(re,"P",{"data-svelte-h":!0}),k(so)!=="svelte-hswkmf"&&(so.innerHTML=Hs),Ln=r(re),L=p(re,"DIV",{class:!0});var xe=v(L);f(ft.$$.fragment,xe),An=r(xe),ao=p(xe,"P",{"data-svelte-h":!0}),k(ao)!=="svelte-y361yc"&&(ao.innerHTML=Vs),Pn=r(xe),f(ze.$$.fragment,xe),Yn=r(xe),f(Je.$$.fragment,xe),xe.forEach(d),re.forEach(d),Bo=r(e),f(gt.$$.fragment,e),Ho=r(e),io=p(e,"P",{}),v(io).forEach(d),this.h()},h(){$(t,"name","hf:doc:metadata"),$(t,"content",fa),Ls(z,"float","right"),$(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(jt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,i){s(document.head,t),m(e,h,i),m(e,o,i),m(e,l,i),m(e,M,i),m(e,y,i),m(e,z,i),m(e,je,i),g(ee,e,i),m(e,co,i),m(e,Ce,i),m(e,po,i),m(e,Fe,i),m(e,mo,i),m(e,qe,i),m(e,ho,i),g(ce,e,i),m(e,uo,i),m(e,Ie,i),m(e,fo,i),g(pe,e,i),m(e,go,i),m(e,Ze,i),m(e,_o,i),m(e,We,i),m(e,To,i),g(Ne,e,i),m(e,bo,i),g(Ge,e,i),m(e,yo,i),m(e,Be,i),m(e,ko,i),g(He,e,i),m(e,Mo,i),m(e,A,i),g(Ve,A,null),s(A,Ro),s(A,Mt),s(A,So),s(A,wt),m(e,wo,i),g(Re,e,i),m(e,vo,i),m(e,U,i),g(Se,U,null),s(U,Xo),s(U,vt),s(U,Qo),s(U,$t),s(U,Eo),s(U,te),g(Xe,te,null),s(te,Lo),s(te,zt),s(te,Ao),s(te,Jt),s(U,Po),s(U,me),g(Qe,me,null),s(me,Yo),s(me,Ut),s(U,Do),s(U,he),g(Ee,he,null),s(he,Oo),s(he,xt),s(U,Ko),s(U,jt),g(Le,jt,null),m(e,$o,i),g(Ae,e,i),m(e,zo,i),m(e,C,i),g(Pe,C,null),s(C,en),s(C,Ct),s(C,tn),s(C,Ft),s(C,on),s(C,oe),g(Ye,oe,null),s(oe,nn),s(oe,qt),s(oe,sn),s(oe,It),s(C,an),s(C,ue),g(De,ue,null),s(ue,rn),s(ue,Zt),m(e,Jo,i),g(Oe,e,i),m(e,Uo,i),m(e,F,i),g(Ke,F,null),s(F,dn),s(F,Wt),s(F,ln),s(F,Nt),s(F,cn),s(F,Gt),s(F,pn),s(F,S),g(et,S,null),s(S,mn),s(S,Bt),s(S,hn),g(fe,S,null),s(S,un),g(ge,S,null),m(e,xo,i),g(tt,e,i),m(e,jo,i),m(e,q,i),g(ot,q,null),s(q,fn),s(q,Ht),s(q,gn),s(q,Vt),s(q,_n),s(q,Rt),s(q,Tn),s(q,X),g(nt,X,null),s(X,bn),s(X,St),s(X,yn),g(_e,X,null),s(X,kn),g(Te,X,null),m(e,Co,i),g(st,e,i),m(e,Fo,i),m(e,I,i),g(at,I,null),s(I,Mn),s(I,Xt),s(I,wn),s(I,Qt),s(I,vn),s(I,Et),s(I,$n),s(I,Q),g(rt,Q,null),s(Q,zn),s(Q,Lt),s(Q,Jn),g(be,Q,null),s(Q,Un),g(ye,Q,null),m(e,qo,i),g(it,e,i),m(e,Io,i),m(e,Z,i),g(dt,Z,null),s(Z,xn),s(Z,At),s(Z,jn),s(Z,Pt),s(Z,Cn),s(Z,Yt),s(Z,Fn),s(Z,G),g(lt,G,null),s(G,qn),s(G,Dt),s(G,In),g(ke,G,null),s(G,Zn),g(Me,G,null),s(G,Wn),g(we,G,null),m(e,Zo,i),g(ct,e,i),m(e,Wo,i),m(e,W,i),g(pt,W,null),s(W,Nn),s(W,Ot),s(W,Gn),s(W,Kt),s(W,Bn),s(W,eo),s(W,Hn),s(W,E),g(mt,E,null),s(E,Vn),s(E,to),s(E,Rn),g(ve,E,null),s(E,Sn),g($e,E,null),m(e,No,i),g(ht,e,i),m(e,Go,i),m(e,N,i),g(ut,N,null),s(N,Xn),s(N,oo),s(N,Qn),s(N,no),s(N,En),s(N,so),s(N,Ln),s(N,L),g(ft,L,null),s(L,An),s(L,ao),s(L,Pn),g(ze,L,null),s(L,Yn),g(Je,L,null),m(e,Bo,i),g(gt,e,i),m(e,Ho,i),m(e,io,i),Vo=!0},p(e,[i]){const ie={};i&2&&(ie.$$scope={dirty:i,ctx:e}),ce.$set(ie);const x={};i&2&&(x.$$scope={dirty:i,ctx:e}),pe.$set(x);const de={};i&2&&(de.$$scope={dirty:i,ctx:e}),fe.$set(de);const _t={};i&2&&(_t.$$scope={dirty:i,ctx:e}),ge.$set(_t);const Tt={};i&2&&(Tt.$$scope={dirty:i,ctx:e}),_e.$set(Tt);const lo={};i&2&&(lo.$$scope={dirty:i,ctx:e}),Te.$set(lo);const B={};i&2&&(B.$$scope={dirty:i,ctx:e}),be.$set(B);const le={};i&2&&(le.$$scope={dirty:i,ctx:e}),ye.$set(le);const bt={};i&2&&(bt.$$scope={dirty:i,ctx:e}),ke.$set(bt);const H={};i&2&&(H.$$scope={dirty:i,ctx:e}),Me.$set(H);const P={};i&2&&(P.$$scope={dirty:i,ctx:e}),we.$set(P);const V={};i&2&&(V.$$scope={dirty:i,ctx:e}),ve.$set(V);const Y={};i&2&&(Y.$$scope={dirty:i,ctx:e}),$e.$set(Y);const R={};i&2&&(R.$$scope={dirty:i,ctx:e}),ze.$set(R);const D={};i&2&&(D.$$scope={dirty:i,ctx:e}),Je.$set(D)},i(e){Vo||(_(ee.$$.fragment,e),_(ce.$$.fragment,e),_(pe.$$.fragment,e),_(Ne.$$.fragment,e),_(Ge.$$.fragment,e),_(He.$$.fragment,e),_(Ve.$$.fragment,e),_(Re.$$.fragment,e),_(Se.$$.fragment,e),_(Xe.$$.fragment,e),_(Qe.$$.fragment,e),_(Ee.$$.fragment,e),_(Le.$$.fragment,e),_(Ae.$$.fragment,e),_(Pe.$$.fragment,e),_(Ye.$$.fragment,e),_(De.$$.fragment,e),_(Oe.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(tt.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(_e.$$.fragment,e),_(Te.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(be.$$.fragment,e),_(ye.$$.fragment,e),_(it.$$.fragment,e),_(dt.$$.fragment,e),_(lt.$$.fragment,e),_(ke.$$.fragment,e),_(Me.$$.fragment,e),_(we.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(ve.$$.fragment,e),_($e.$$.fragment,e),_(ht.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(ze.$$.fragment,e),_(Je.$$.fragment,e),_(gt.$$.fragment,e),Vo=!0)},o(e){T(ee.$$.fragment,e),T(ce.$$.fragment,e),T(pe.$$.fragment,e),T(Ne.$$.fragment,e),T(Ge.$$.fragment,e),T(He.$$.fragment,e),T(Ve.$$.fragment,e),T(Re.$$.fragment,e),T(Se.$$.fragment,e),T(Xe.$$.fragment,e),T(Qe.$$.fragment,e),T(Ee.$$.fragment,e),T(Le.$$.fragment,e),T(Ae.$$.fragment,e),T(Pe.$$.fragment,e),T(Ye.$$.fragment,e),T(De.$$.fragment,e),T(Oe.$$.fragment,e),T(Ke.$$.fragment,e),T(et.$$.fragment,e),T(fe.$$.fragment,e),T(ge.$$.fragment,e),T(tt.$$.fragment,e),T(ot.$$.fragment,e),T(nt.$$.fragment,e),T(_e.$$.fragment,e),T(Te.$$.fragment,e),T(st.$$.fragment,e),T(at.$$.fragment,e),T(rt.$$.fragment,e),T(be.$$.fragment,e),T(ye.$$.fragment,e),T(it.$$.fragment,e),T(dt.$$.fragment,e),T(lt.$$.fragment,e),T(ke.$$.fragment,e),T(Me.$$.fragment,e),T(we.$$.fragment,e),T(ct.$$.fragment,e),T(pt.$$.fragment,e),T(mt.$$.fragment,e),T(ve.$$.fragment,e),T($e.$$.fragment,e),T(ht.$$.fragment,e),T(ut.$$.fragment,e),T(ft.$$.fragment,e),T(ze.$$.fragment,e),T(Je.$$.fragment,e),T(gt.$$.fragment,e),Vo=!1},d(e){e&&(d(h),d(o),d(l),d(M),d(y),d(z),d(je),d(co),d(Ce),d(po),d(Fe),d(mo),d(qe),d(ho),d(uo),d(Ie),d(fo),d(go),d(Ze),d(_o),d(We),d(To),d(bo),d(yo),d(Be),d(ko),d(Mo),d(A),d(wo),d(vo),d(U),d($o),d(zo),d(C),d(Jo),d(Uo),d(F),d(xo),d(jo),d(q),d(Co),d(Fo),d(I),d(qo),d(Io),d(Z),d(Zo),d(Wo),d(W),d(No),d(Go),d(N),d(Bo),d(Ho),d(io)),d(t),b(ee,e),b(ce,e),b(pe,e),b(Ne,e),b(Ge,e),b(He,e),b(Ve),b(Re,e),b(Se),b(Xe),b(Qe),b(Ee),b(Le),b(Ae,e),b(Pe),b(Ye),b(De),b(Oe,e),b(Ke),b(et),b(fe),b(ge),b(tt,e),b(ot),b(nt),b(_e),b(Te),b(st,e),b(at),b(rt),b(be),b(ye),b(it,e),b(dt),b(lt),b(ke),b(Me),b(we),b(ct,e),b(pt),b(mt),b(ve),b($e),b(ht,e),b(ut),b(ft),b(ze),b(Je),b(gt,e)}}}const fa='{"title":"T5","local":"t5","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"T5Config","local":"transformers.T5Config","sections":[],"depth":2},{"title":"T5Tokenizer","local":"transformers.T5Tokenizer","sections":[],"depth":2},{"title":"T5TokenizerFast","local":"transformers.T5TokenizerFast","sections":[],"depth":2},{"title":"T5Model","local":"transformers.T5Model","sections":[],"depth":2},{"title":"T5ForConditionalGeneration","local":"transformers.T5ForConditionalGeneration","sections":[],"depth":2},{"title":"T5EncoderModel","local":"transformers.T5EncoderModel","sections":[],"depth":2},{"title":"T5ForSequenceClassification","local":"transformers.T5ForSequenceClassification","sections":[],"depth":2},{"title":"T5ForTokenClassification","local":"transformers.T5ForTokenClassification","sections":[],"depth":2},{"title":"T5ForQuestionAnswering","local":"transformers.T5ForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function ga(w){return Ss(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class $a extends Xs{constructor(t){super(),Qs(this,t,ga,ua,Rs,{})}}export{$a as component};
