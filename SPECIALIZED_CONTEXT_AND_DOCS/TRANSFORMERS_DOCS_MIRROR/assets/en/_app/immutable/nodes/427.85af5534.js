import{s as In,o as Zn,n as q}from"../chunks/scheduler.18a86fab.js";import{S as Wn,i as Gn,g as p,s as a,r as m,A as Bn,h as u,f as s,c as r,j as x,x as h,u as f,k as v,y as d,a as c,v as g,d as _,t as T,w as M}from"../chunks/index.98837b22.js";import{T as De}from"../chunks/Tip.77304350.js";import{D as R}from"../chunks/Docstring.a1ef7999.js";import{C as ge}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ke}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as L,E as Nn}from"../chunks/getInferenceSnippets.06c2775f.js";function Vn(k){let t,b='Refer to <a href="t5">T5’s documentation page</a> for more tips, code examples and notebooks.';return{c(){t=p("p"),t.innerHTML=b},l(i){t=u(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1he4oyy"&&(t.innerHTML=b)},m(i,l){c(i,t,l)},p:q,d(i){i&&s(t)}}}function Rn(k){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(i){t=u(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,l){c(i,t,l)},p:q,d(i){i&&s(t)}}}function Xn(k){let t,b="Example:",i,l,y;return l=new ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBVTVQ1TW9kZWwlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ1bXQ1LXNtYWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwVU1UNU1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ1bXQ1LXNtYWxsJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJTdHVkaWVzJTIwaGF2ZSUyMGJlZW4lMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSkuaW5wdXRfaWRzJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQWRlY29kZXJfaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlN0dWRpZXMlMjBzaG93JTIwdGhhdCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEElMEElMjMlMjBwcmVwcm9jZXNzJTNBJTIwUHJlcGVuZCUyMGRlY29kZXJfaW5wdXRfaWRzJTIwd2l0aCUyMHN0YXJ0JTIwdG9rZW4lMjB3aGljaCUyMGlzJTIwcGFkJTIwdG9rZW4lMjBmb3IlMjBVTVQ1TW9kZWwuJTBBJTIzJTIwVGhpcyUyMGlzJTIwbm90JTIwbmVlZGVkJTIwZm9yJTIwdG9yY2gncyUyMFVNVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMjBhcyUyMGl0JTIwZG9lcyUyMHRoaXMlMjBpbnRlcm5hbGx5JTIwdXNpbmclMjBsYWJlbHMlMjBhcmcuJTBBZGVjb2Rlcl9pbnB1dF9pZHMlMjAlM0QlMjBtb2RlbC5fc2hpZnRfcmlnaHQoZGVjb2Rlcl9pbnB1dF9pZHMpJTBBJTBBJTIzJTIwZm9yd2FyZCUyMHBhc3MlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RkZWNvZGVyX2lucHV0X2lkcyklMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, UMT5Model

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5Model.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># preprocess: Prepend decoder_input_ids with start token which is pad token for UMT5Model.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># This is not needed for torch&#x27;s UMT5ForConditionalGeneration as it does this internally using labels arg.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = model._shift_right(decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,i=a(),m(l.$$.fragment)},l(o){t=u(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=b),i=r(o),f(l.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(l,o,w),y=!0},p:q,i(o){y||(_(l.$$.fragment,o),y=!0)},o(o){T(l.$$.fragment,o),y=!1},d(o){o&&(s(t),s(i)),M(l,o)}}}function Hn(k){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(i){t=u(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,l){c(i,t,l)},p:q,d(i){i&&s(t)}}}function Sn(k){let t,b="Examples:",i,l,y;return l=new ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBVTVQ1Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdW10NS1zbWFsbCUyMiklMEFtb2RlbCUyMCUzRCUyMFVNVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnVtdDUtc21hbGwlMjIpJTBBJTBBJTIzJTIwdHJhaW5pbmclMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwJTNDZXh0cmFfaWRfMCUzRSUyMHdhbGtzJTIwaW4lMjAlM0NleHRyYV9pZF8xJTNFJTIwcGFyayUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjIlM0NleHRyYV9pZF8wJTNFJTIwY3V0ZSUyMGRvZyUyMCUzQ2V4dHJhX2lkXzElM0UlMjB0aGUlMjAlM0NleHRyYV9pZF8yJTNFJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikuaW5wdXRfaWRzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUzRGlucHV0X2lkcyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEElMEElMjMlMjBpbmZlcmVuY2UlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyU3R1ZGllcyUyMGhhdmUlMjBzaG93biUyMHRoYXQlMjAlM0NleHRyYV9pZF8wJTNFJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dF9pZHMpJTBBdG9rZW5pemVyLmRlY29kZShvdXRwdXRzJTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, UMT5ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># training</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(<span class="hljs-string">&quot;The &lt;extra_id_0&gt; walks in &lt;extra_id_1&gt; park&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;&lt;extra_id_0&gt; cute dog &lt;extra_id_1&gt; the &lt;extra_id_2&gt;&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># inference</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(<span class="hljs-string">&quot;Studies have shown that &lt;extra_id_0&gt; good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.generate(input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,i=a(),m(l.$$.fragment)},l(o){t=u(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=b),i=r(o),f(l.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(l,o,w),y=!0},p:q,i(o){y||(_(l.$$.fragment,o),y=!0)},o(o){T(l.$$.fragment,o),y=!1},d(o){o&&(s(t),s(i)),M(l,o)}}}function Qn(k){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(i){t=u(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,l){c(i,t,l)},p:q,d(i){i&&s(t)}}}function En(k){let t,b="Example:",i,l,y;return l=new ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBVTVQ1RW5jb2Rlck1vZGVsJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdW10NS1zbWFsbCUyMiklMEFtb2RlbCUyMCUzRCUyMFVNVDVFbmNvZGVyTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnVtdDUtc21hbGwlMjIpJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMlN0dWRpZXMlMjBoYXZlJTIwYmVlbiUyMHNob3duJTIwdGhhdCUyMG93bmluZyUyMGElMjBkb2clMjBpcyUyMGdvb2QlMjBmb3IlMjB5b3UlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUzRGlucHV0X2lkcyklMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, UMT5EncoderModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5EncoderModel.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,i=a(),m(l.$$.fragment)},l(o){t=u(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=b),i=r(o),f(l.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(l,o,w),y=!0},p:q,i(o){y||(_(l.$$.fragment,o),y=!0)},o(o){T(l.$$.fragment,o),y=!1},d(o){o&&(s(t),s(i)),M(l,o)}}}function Ln(k){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(i){t=u(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,l){c(i,t,l)},p:q,d(i){i&&s(t)}}}function An(k){let t,b="Example of single-label classification:",i,l,y;return l=new ge({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFVNVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdW10NS1zbWFsbCUyMiklMEFtb2RlbCUyMCUzRCUyMFVNVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ1bXQ1LXNtYWxsJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBVTVQ1Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdW10NS1zbWFsbCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, UMT5ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,i=a(),m(l.$$.fragment)},l(o){t=u(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=b),i=r(o),f(l.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(l,o,w),y=!0},p:q,i(o){y||(_(l.$$.fragment,o),y=!0)},o(o){T(l.$$.fragment,o),y=!1},d(o){o&&(s(t),s(i)),M(l,o)}}}function Pn(k){let t,b="Example of multi-label classification:",i,l,y;return l=new ge({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFVNVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdW10NS1zbWFsbCUyMiklMEFtb2RlbCUyMCUzRCUyMFVNVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ1bXQ1LXNtYWxsJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBVTVQ1Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGdW10NS1zbWFsbCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, UMT5ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5ForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/umt5-small&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,i=a(),m(l.$$.fragment)},l(o){t=u(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=b),i=r(o),f(l.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(l,o,w),y=!0},p:q,i(o){y||(_(l.$$.fragment,o),y=!0)},o(o){T(l.$$.fragment,o),y=!1},d(o){o&&(s(t),s(i)),M(l,o)}}}function Yn(k){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(i){t=u(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,l){c(i,t,l)},p:q,d(i){i&&s(t)}}}function On(k){let t,b="Example:",i,l,y;return l=new ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBVTVQ1Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdW10NS1zbWFsbCUyMiklMEFtb2RlbCUyMCUzRCUyMFVNVDVGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ1bXQ1LXNtYWxsJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, UMT5ForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5ForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,i=a(),m(l.$$.fragment)},l(o){t=u(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=b),i=r(o),f(l.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(l,o,w),y=!0},p:q,i(o){y||(_(l.$$.fragment,o),y=!0)},o(o){T(l.$$.fragment,o),y=!1},d(o){o&&(s(t),s(i)),M(l,o)}}}function Dn(k){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(i){t=u(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,l){c(i,t,l)},p:q,d(i){i&&s(t)}}}function Kn(k){let t,b="Example:",i,l,y;return l=new ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBVTVQ1Rm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnVtdDUtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBVTVQ1Rm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnVtdDUtc21hbGwlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, UMT5ForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = UMT5ForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,i=a(),m(l.$$.fragment)},l(o){t=u(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=b),i=r(o),f(l.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(l,o,w),y=!0},p:q,i(o){y||(_(l.$$.fragment,o),y=!0)},o(o){T(l.$$.fragment,o),y=!1},d(o){o&&(s(t),s(i)),M(l,o)}}}function es(k){let t,b,i,l,y,o="<em>This model was released on 2023-04-18 and added to Hugging Face Transformers on 2023-07-03.</em>",w,_e,Ct,te,Oo='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Jt,Te,jt,Me,Do='The UMT5 model was proposed in <a href="https://huggingface.co/papers/2304.09151" rel="nofollow">UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining</a> by Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, Noah Constant.',zt,be,Ko="The abstract from the paper is the following:",xt,ye,en="<em>Pretrained multilingual large language models have typically used heuristic temperature-based sampling to balance between different languages. However previous work has not systematically evaluated the efficacy of different pretraining language distributions across model scales. In this paper, we propose a new sampling method, UniMax, that delivers more uniform coverage of head languages while mitigating overfitting on tail languages by explicitly capping the number of repeats over each language’s corpus. We perform an extensive series of ablations testing a range of sampling strategies on a suite of multilingual benchmarks, while varying model scale. We find that UniMax outperforms standard temperature-based sampling, and the benefits persist as scale increases. As part of our contribution, we release: (i) an improved and refreshed mC4 multilingual corpus consisting of 29 trillion characters across 107 languages, and (ii) a suite of pretrained umT5 model checkpoints trained with UniMax sampling.</em>",Ft,we,tn="Google has released the following variants:",qt,ke,on='<li><a href="https://huggingface.co/google/umt5-small" rel="nofollow">google/umt5-small</a></li> <li><a href="https://huggingface.co/google/umt5-base" rel="nofollow">google/umt5-base</a></li> <li><a href="https://huggingface.co/google/umt5-xl" rel="nofollow">google/umt5-xl</a></li> <li><a href="https://huggingface.co/google/umt5-xxl" rel="nofollow">google/umt5-xxl</a>.</li>',It,ve,nn=`This model was contributed by <a href="https://huggingface.co/agemagician" rel="nofollow">agemagician</a> and <a href="https://huggingface.co/stefan-it" rel="nofollow">stefan-it</a>. The original code can be
found <a href="https://github.com/google-research/t5x" rel="nofollow">here</a>.`,Zt,Ue,Wt,$e,sn=`<li>UMT5 was only pre-trained on <a href="https://huggingface.co/datasets/mc4" rel="nofollow">mC4</a> excluding any supervised training.
Therefore, this model has to be fine-tuned before it is usable on a downstream task, unlike the original T5 model.</li> <li>Since umT5 was pre-trained in an unsupervised manner, there’s no real advantage to using a task prefix during single-task
fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.</li>`,Gt,Ce,Bt,Je,an=`<code>UmT5</code> is based on mT5, with a non-shared relative positional bias that is computed for each layer. This means that the model set <code>has_relative_bias</code> for each layer.
The conversion script is also different because the model was saved in t5x’s latest checkpointing format.`,Nt,je,Vt,ze,Rt,oe,Xt,xe,Ht,A,Fe,ao,et,rn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Model">UMT5Model</a>. It is used to instantiate a UMT5
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the UMT5
<a href="https://huggingface.co/google/umt5-small" rel="nofollow">google/umt5-small</a> architecture.`,ro,tt,dn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,St,qe,Qt,U,Ie,io,ot,ln="The bare Umt5 Model outputting raw hidden-states without any specific head on top.",lo,nt,cn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,co,st,pn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,po,X,Ze,uo,at,un='The <a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Model">UMT5Model</a> forward method, overrides the <code>__call__</code> special method.',ho,ne,mo,se,Et,We,Lt,$,Ge,fo,rt,hn="UMT5 Model with a <code>language modeling</code> head on top.",go,it,mn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_o,dt,fn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,To,H,Be,Mo,lt,gn='The <a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForConditionalGeneration">UMT5ForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',bo,ae,yo,re,At,Ne,Pt,C,Ve,wo,ct,_n="The bare Umt5 Model outputting raw hidden-states without any specific head on top.",ko,pt,Tn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vo,ut,Mn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Uo,S,Re,$o,ht,bn='The <a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5EncoderModel">UMT5EncoderModel</a> forward method, overrides the <code>__call__</code> special method.',Co,ie,Jo,de,Yt,Xe,Ot,J,He,jo,mt,yn=`UMT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.`,zo,ft,wn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xo,gt,kn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fo,F,Se,qo,_t,vn='The <a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForSequenceClassification">UMT5ForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Io,le,Zo,ce,Wo,pe,Dt,Qe,Kt,j,Ee,Go,Tt,Un=`The Umt5 transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Bo,Mt,$n=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,No,bt,Cn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Vo,Q,Le,Ro,yt,Jn='The <a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForTokenClassification">UMT5ForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Xo,ue,Ho,he,eo,Ae,to,z,Pe,So,wt,jn=`The Umt5 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Qo,kt,zn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Eo,vt,xn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Lo,E,Ye,Ao,Ut,Fn='The <a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForQuestionAnswering">UMT5ForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Po,me,Yo,fe,oo,Oe,no,$t,so;return _e=new L({props:{title:"UMT5",local:"umt5",headingTag:"h1"}}),Te=new L({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ue=new L({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Ce=new L({props:{title:"Differences with mT5?",local:"differences-with-mt5",headingTag:"h2"}}),je=new L({props:{title:"Sample usage",local:"sample-usage",headingTag:"h1"}}),ze=new ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvclNlcTJTZXFMTSUyQyUyMEF1dG9Ub2tlbml6ZXIlMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvclNlcTJTZXFMTS5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdW10NS1zbWFsbCUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ1bXQ1LXNtYWxsJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJBJTIwJTNDZXh0cmFfaWRfMCUzRSUyMHdhbGtzJTIwaW50byUyMGElMjBiYXIlMjBhbmQlMjBvcmRlcnMlMjBhJTIwJTNDZXh0cmFfaWRfMSUzRSUyMHdpdGglMjAlM0NleHRyYV9pZF8yJTNFJTIwcGluY2glMjBvZiUyMCUzQ2V4dHJhX2lkXzMlM0UuJTIyJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUwQSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMpJTBBcHJpbnQodG9rZW5pemVyLmJhdGNoX2RlY29kZShvdXRwdXRzKSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForSeq2SeqLM.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/umt5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;A &lt;extra_id_0&gt; walks into a bar and orders a &lt;extra_id_1&gt; with &lt;extra_id_2&gt; pinch of &lt;extra_id_3&gt;.&quot;</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(tokenizer.batch_decode(outputs))
[<span class="hljs-string">&#x27;&lt;pad&gt;&lt;extra_id_0&gt;nyone who&lt;extra_id_1&gt; drink&lt;extra_id_2&gt; a&lt;extra_id_3&gt; alcohol&lt;extra_id_4&gt; A&lt;extra_id_5&gt; A. This&lt;extra_id_6&gt; I&lt;extra_id_7&gt;&lt;extra_id_52&gt;&lt;extra_id_53&gt;&lt;/s&gt;&#x27;</span>]`,wrap:!1}}),oe=new De({props:{$$slots:{default:[Vn]},$$scope:{ctx:k}}}),xe=new L({props:{title:"UMT5Config",local:"transformers.UMT5Config",headingTag:"h2"}}),Fe=new R({props:{name:"class transformers.UMT5Config",anchor:"transformers.UMT5Config",parameters:[{name:"vocab_size",val:" = 250112"},{name:"d_model",val:" = 512"},{name:"d_kv",val:" = 64"},{name:"d_ff",val:" = 1024"},{name:"num_layers",val:" = 8"},{name:"num_decoder_layers",val:" = None"},{name:"num_heads",val:" = 6"},{name:"relative_attention_num_buckets",val:" = 32"},{name:"relative_attention_max_distance",val:" = 128"},{name:"dropout_rate",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-06"},{name:"initializer_factor",val:" = 1.0"},{name:"feed_forward_proj",val:" = 'gated-gelu'"},{name:"is_encoder_decoder",val:" = True"},{name:"use_cache",val:" = True"},{name:"tokenizer_class",val:" = 'T5Tokenizer'"},{name:"tie_word_embeddings",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"decoder_start_token_id",val:" = 0"},{name:"classifier_dropout",val:" = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.UMT5Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 250112) &#x2014;
Vocabulary size of the UMT5 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Model">UMT5Model</a> or <code>TFUMT5Model</code>.`,name:"vocab_size"},{anchor:"transformers.UMT5Config.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Size of the encoder layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.UMT5Config.d_kv",description:`<strong>d_kv</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Size of the key, query, value projections per attention head. <code>d_kv</code> has to be equal to <code>d_model // num_heads</code>.`,name:"d_kv"},{anchor:"transformers.UMT5Config.d_ff",description:`<strong>d_ff</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Size of the intermediate feed forward layer in each <code>UMT5Block</code>.`,name:"d_ff"},{anchor:"transformers.UMT5Config.num_layers",description:`<strong>num_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_layers"},{anchor:"transformers.UMT5Config.num_decoder_layers",description:`<strong>num_decoder_layers</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Number of hidden layers in the Transformer decoder. Will use the same value as <code>num_layers</code> if not set.`,name:"num_decoder_layers"},{anchor:"transformers.UMT5Config.num_heads",description:`<strong>num_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_heads"},{anchor:"transformers.UMT5Config.relative_attention_num_buckets",description:`<strong>relative_attention_num_buckets</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of buckets to use for each attention layer.`,name:"relative_attention_num_buckets"},{anchor:"transformers.UMT5Config.relative_attention_max_distance",description:`<strong>relative_attention_max_distance</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The maximum distance of the longer sequences for the bucket separation.`,name:"relative_attention_max_distance"},{anchor:"transformers.UMT5Config.dropout_rate",description:`<strong>dropout_rate</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The ratio for all dropout layers.`,name:"dropout_rate"},{anchor:"transformers.UMT5Config.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.UMT5Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-6) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.UMT5Config.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"},{anchor:"transformers.UMT5Config.feed_forward_proj",description:`<strong>feed_forward_proj</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;gated-gelu&quot;</code>) &#x2014;
Type of feed forward layer to be used. Should be one of <code>&quot;relu&quot;</code> or <code>&quot;gated-gelu&quot;</code>.`,name:"feed_forward_proj"},{anchor:"transformers.UMT5Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/configuration_umt5.py#L27"}}),qe=new L({props:{title:"UMT5Model",local:"transformers.UMT5Model",headingTag:"h2"}}),Ie=new R({props:{name:"class transformers.UMT5Model",anchor:"transformers.UMT5Model",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.UMT5Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Model">UMT5Model</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L939"}}),Ze=new R({props:{name:"forward",anchor:"transformers.UMT5Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.UMT5Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so
you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./umt5#training">UMT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.UMT5Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.UMT5Model.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>UMT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./umt5#training">UMT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.UMT5Model.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.UMT5Model.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.UMT5Model.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.UMT5Model.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.UMT5Model.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.UMT5Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.UMT5Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.UMT5Model.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.UMT5Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.UMT5Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.UMT5Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.UMT5Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.UMT5Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1009",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config"
>UMT5Config</a>) and inputs.</p>
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
`}}),ne=new De({props:{$$slots:{default:[Rn]},$$scope:{ctx:k}}}),se=new Ke({props:{anchor:"transformers.UMT5Model.forward.example",$$slots:{default:[Xn]},$$scope:{ctx:k}}}),We=new L({props:{title:"UMT5ForConditionalGeneration",local:"transformers.UMT5ForConditionalGeneration",headingTag:"h2"}}),Ge=new R({props:{name:"class transformers.UMT5ForConditionalGeneration",anchor:"transformers.UMT5ForConditionalGeneration",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.UMT5ForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForConditionalGeneration">UMT5ForConditionalGeneration</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1150"}}),Be=new R({props:{name:"forward",anchor:"transformers.UMT5ForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.UMT5ForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so
you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./umt5#training">UMT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>UMT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./umt5#training">UMT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[-100, 0, ..., config.vocab_size - 1]</code>. All labels set to <code>-100</code> are ignored (masked), the loss is only computed for
labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.UMT5ForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1213",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config"
>UMT5Config</a>) and inputs.</p>
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
`}}),ae=new De({props:{$$slots:{default:[Hn]},$$scope:{ctx:k}}}),re=new Ke({props:{anchor:"transformers.UMT5ForConditionalGeneration.forward.example",$$slots:{default:[Sn]},$$scope:{ctx:k}}}),Ne=new L({props:{title:"UMT5EncoderModel",local:"transformers.UMT5EncoderModel",headingTag:"h2"}}),Ve=new R({props:{name:"class transformers.UMT5EncoderModel",anchor:"transformers.UMT5EncoderModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.UMT5EncoderModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5EncoderModel">UMT5EncoderModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1381"}}),Re=new R({props:{name:"forward",anchor:"transformers.UMT5EncoderModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.UMT5EncoderModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./umt5#training">UMT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.UMT5EncoderModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.UMT5EncoderModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.UMT5EncoderModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.UMT5EncoderModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.UMT5EncoderModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.UMT5EncoderModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1439",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config"
>UMT5Config</a>) and inputs.</p>
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
`}}),ie=new De({props:{$$slots:{default:[Qn]},$$scope:{ctx:k}}}),de=new Ke({props:{anchor:"transformers.UMT5EncoderModel.forward.example",$$slots:{default:[En]},$$scope:{ctx:k}}}),Xe=new L({props:{title:"UMT5ForSequenceClassification",local:"transformers.UMT5ForSequenceClassification",headingTag:"h2"}}),He=new R({props:{name:"class transformers.UMT5ForSequenceClassification",anchor:"transformers.UMT5ForSequenceClassification",parameters:[{name:"config",val:": UMT5Config"}],parametersDescription:[{anchor:"transformers.UMT5ForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config">UMT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1495"}}),Se=new R({props:{name:"forward",anchor:"transformers.UMT5ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.UMT5ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so
you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./umt5#training">UMT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.UMT5ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.UMT5ForSequenceClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>UMT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./umt5#training">UMT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.UMT5ForSequenceClassification.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.UMT5ForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.UMT5ForSequenceClassification.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.UMT5ForSequenceClassification.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.UMT5ForSequenceClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.UMT5ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.UMT5ForSequenceClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.UMT5ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.UMT5ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.UMT5ForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.UMT5ForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.UMT5ForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1510",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config"
>UMT5Config</a>) and inputs.</p>
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
`}}),le=new De({props:{$$slots:{default:[Ln]},$$scope:{ctx:k}}}),ce=new Ke({props:{anchor:"transformers.UMT5ForSequenceClassification.forward.example",$$slots:{default:[An]},$$scope:{ctx:k}}}),pe=new Ke({props:{anchor:"transformers.UMT5ForSequenceClassification.forward.example-2",$$slots:{default:[Pn]},$$scope:{ctx:k}}}),Qe=new L({props:{title:"UMT5ForTokenClassification",local:"transformers.UMT5ForTokenClassification",headingTag:"h2"}}),Ee=new R({props:{name:"class transformers.UMT5ForTokenClassification",anchor:"transformers.UMT5ForTokenClassification",parameters:[{name:"config",val:": UMT5Config"}],parametersDescription:[{anchor:"transformers.UMT5ForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config">UMT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1659"}}),Le=new R({props:{name:"forward",anchor:"transformers.UMT5ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.UMT5ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./umt5#training">UMT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.UMT5ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.UMT5ForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.UMT5ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.UMT5ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.UMT5ForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.UMT5ForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.UMT5ForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1675",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config"
>UMT5Config</a>) and inputs.</p>
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
`}}),ue=new De({props:{$$slots:{default:[Yn]},$$scope:{ctx:k}}}),he=new Ke({props:{anchor:"transformers.UMT5ForTokenClassification.forward.example",$$slots:{default:[On]},$$scope:{ctx:k}}}),Ae=new L({props:{title:"UMT5ForQuestionAnswering",local:"transformers.UMT5ForQuestionAnswering",headingTag:"h2"}}),Pe=new R({props:{name:"class transformers.UMT5ForQuestionAnswering",anchor:"transformers.UMT5ForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.UMT5ForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5ForQuestionAnswering">UMT5ForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1736"}}),Ye=new R({props:{name:"forward",anchor:"transformers.UMT5ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.UMT5ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. UMT5 is a model with relative position embeddings so
you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./umt5#training">UMT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>UMT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./umt5#training">UMT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.UMT5ForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/umt5/modeling_umt5.py#L1783",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput"
>transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/umt5#transformers.UMT5Config"
>UMT5Config</a>) and inputs.</p>
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
`}}),me=new De({props:{$$slots:{default:[Dn]},$$scope:{ctx:k}}}),fe=new Ke({props:{anchor:"transformers.UMT5ForQuestionAnswering.forward.example",$$slots:{default:[Kn]},$$scope:{ctx:k}}}),Oe=new Nn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/umt5.md"}}),{c(){t=p("meta"),b=a(),i=p("p"),l=a(),y=p("p"),y.innerHTML=o,w=a(),m(_e.$$.fragment),Ct=a(),te=p("div"),te.innerHTML=Oo,Jt=a(),m(Te.$$.fragment),jt=a(),Me=p("p"),Me.innerHTML=Do,zt=a(),be=p("p"),be.textContent=Ko,xt=a(),ye=p("p"),ye.innerHTML=en,Ft=a(),we=p("p"),we.textContent=tn,qt=a(),ke=p("ul"),ke.innerHTML=on,It=a(),ve=p("p"),ve.innerHTML=nn,Zt=a(),m(Ue.$$.fragment),Wt=a(),$e=p("ul"),$e.innerHTML=sn,Gt=a(),m(Ce.$$.fragment),Bt=a(),Je=p("p"),Je.innerHTML=an,Nt=a(),m(je.$$.fragment),Vt=a(),m(ze.$$.fragment),Rt=a(),m(oe.$$.fragment),Xt=a(),m(xe.$$.fragment),Ht=a(),A=p("div"),m(Fe.$$.fragment),ao=a(),et=p("p"),et.innerHTML=rn,ro=a(),tt=p("p"),tt.innerHTML=dn,St=a(),m(qe.$$.fragment),Qt=a(),U=p("div"),m(Ie.$$.fragment),io=a(),ot=p("p"),ot.textContent=ln,lo=a(),nt=p("p"),nt.innerHTML=cn,co=a(),st=p("p"),st.innerHTML=pn,po=a(),X=p("div"),m(Ze.$$.fragment),uo=a(),at=p("p"),at.innerHTML=un,ho=a(),m(ne.$$.fragment),mo=a(),m(se.$$.fragment),Et=a(),m(We.$$.fragment),Lt=a(),$=p("div"),m(Ge.$$.fragment),fo=a(),rt=p("p"),rt.innerHTML=hn,go=a(),it=p("p"),it.innerHTML=mn,_o=a(),dt=p("p"),dt.innerHTML=fn,To=a(),H=p("div"),m(Be.$$.fragment),Mo=a(),lt=p("p"),lt.innerHTML=gn,bo=a(),m(ae.$$.fragment),yo=a(),m(re.$$.fragment),At=a(),m(Ne.$$.fragment),Pt=a(),C=p("div"),m(Ve.$$.fragment),wo=a(),ct=p("p"),ct.textContent=_n,ko=a(),pt=p("p"),pt.innerHTML=Tn,vo=a(),ut=p("p"),ut.innerHTML=Mn,Uo=a(),S=p("div"),m(Re.$$.fragment),$o=a(),ht=p("p"),ht.innerHTML=bn,Co=a(),m(ie.$$.fragment),Jo=a(),m(de.$$.fragment),Yt=a(),m(Xe.$$.fragment),Ot=a(),J=p("div"),m(He.$$.fragment),jo=a(),mt=p("p"),mt.textContent=yn,zo=a(),ft=p("p"),ft.innerHTML=wn,xo=a(),gt=p("p"),gt.innerHTML=kn,Fo=a(),F=p("div"),m(Se.$$.fragment),qo=a(),_t=p("p"),_t.innerHTML=vn,Io=a(),m(le.$$.fragment),Zo=a(),m(ce.$$.fragment),Wo=a(),m(pe.$$.fragment),Dt=a(),m(Qe.$$.fragment),Kt=a(),j=p("div"),m(Ee.$$.fragment),Go=a(),Tt=p("p"),Tt.textContent=Un,Bo=a(),Mt=p("p"),Mt.innerHTML=$n,No=a(),bt=p("p"),bt.innerHTML=Cn,Vo=a(),Q=p("div"),m(Le.$$.fragment),Ro=a(),yt=p("p"),yt.innerHTML=Jn,Xo=a(),m(ue.$$.fragment),Ho=a(),m(he.$$.fragment),eo=a(),m(Ae.$$.fragment),to=a(),z=p("div"),m(Pe.$$.fragment),So=a(),wt=p("p"),wt.innerHTML=jn,Qo=a(),kt=p("p"),kt.innerHTML=zn,Eo=a(),vt=p("p"),vt.innerHTML=xn,Lo=a(),E=p("div"),m(Ye.$$.fragment),Ao=a(),Ut=p("p"),Ut.innerHTML=Fn,Po=a(),m(me.$$.fragment),Yo=a(),m(fe.$$.fragment),oo=a(),m(Oe.$$.fragment),no=a(),$t=p("p"),this.h()},l(e){const n=Bn("svelte-u9bgzb",document.head);t=u(n,"META",{name:!0,content:!0}),n.forEach(s),b=r(e),i=u(e,"P",{}),x(i).forEach(s),l=r(e),y=u(e,"P",{"data-svelte-h":!0}),h(y)!=="svelte-w4zlgr"&&(y.innerHTML=o),w=r(e),f(_e.$$.fragment,e),Ct=r(e),te=u(e,"DIV",{class:!0,"data-svelte-h":!0}),h(te)!=="svelte-13t8s2t"&&(te.innerHTML=Oo),Jt=r(e),f(Te.$$.fragment,e),jt=r(e),Me=u(e,"P",{"data-svelte-h":!0}),h(Me)!=="svelte-1vvbxpc"&&(Me.innerHTML=Do),zt=r(e),be=u(e,"P",{"data-svelte-h":!0}),h(be)!=="svelte-vfdo9a"&&(be.textContent=Ko),xt=r(e),ye=u(e,"P",{"data-svelte-h":!0}),h(ye)!=="svelte-10kmzrd"&&(ye.innerHTML=en),Ft=r(e),we=u(e,"P",{"data-svelte-h":!0}),h(we)!=="svelte-1p0jqca"&&(we.textContent=tn),qt=r(e),ke=u(e,"UL",{"data-svelte-h":!0}),h(ke)!=="svelte-coj30a"&&(ke.innerHTML=on),It=r(e),ve=u(e,"P",{"data-svelte-h":!0}),h(ve)!=="svelte-1orfdft"&&(ve.innerHTML=nn),Zt=r(e),f(Ue.$$.fragment,e),Wt=r(e),$e=u(e,"UL",{"data-svelte-h":!0}),h($e)!=="svelte-gb6546"&&($e.innerHTML=sn),Gt=r(e),f(Ce.$$.fragment,e),Bt=r(e),Je=u(e,"P",{"data-svelte-h":!0}),h(Je)!=="svelte-4mli02"&&(Je.innerHTML=an),Nt=r(e),f(je.$$.fragment,e),Vt=r(e),f(ze.$$.fragment,e),Rt=r(e),f(oe.$$.fragment,e),Xt=r(e),f(xe.$$.fragment,e),Ht=r(e),A=u(e,"DIV",{class:!0});var ee=x(A);f(Fe.$$.fragment,ee),ao=r(ee),et=u(ee,"P",{"data-svelte-h":!0}),h(et)!=="svelte-185rr3w"&&(et.innerHTML=rn),ro=r(ee),tt=u(ee,"P",{"data-svelte-h":!0}),h(tt)!=="svelte-1ek1ss9"&&(tt.innerHTML=dn),ee.forEach(s),St=r(e),f(qe.$$.fragment,e),Qt=r(e),U=u(e,"DIV",{class:!0});var I=x(U);f(Ie.$$.fragment,I),io=r(I),ot=u(I,"P",{"data-svelte-h":!0}),h(ot)!=="svelte-1r92lz9"&&(ot.textContent=ln),lo=r(I),nt=u(I,"P",{"data-svelte-h":!0}),h(nt)!=="svelte-q52n56"&&(nt.innerHTML=cn),co=r(I),st=u(I,"P",{"data-svelte-h":!0}),h(st)!=="svelte-hswkmf"&&(st.innerHTML=pn),po=r(I),X=u(I,"DIV",{class:!0});var P=x(X);f(Ze.$$.fragment,P),uo=r(P),at=u(P,"P",{"data-svelte-h":!0}),h(at)!=="svelte-1e85e86"&&(at.innerHTML=un),ho=r(P),f(ne.$$.fragment,P),mo=r(P),f(se.$$.fragment,P),P.forEach(s),I.forEach(s),Et=r(e),f(We.$$.fragment,e),Lt=r(e),$=u(e,"DIV",{class:!0});var Z=x($);f(Ge.$$.fragment,Z),fo=r(Z),rt=u(Z,"P",{"data-svelte-h":!0}),h(rt)!=="svelte-1h0jipp"&&(rt.innerHTML=hn),go=r(Z),it=u(Z,"P",{"data-svelte-h":!0}),h(it)!=="svelte-q52n56"&&(it.innerHTML=mn),_o=r(Z),dt=u(Z,"P",{"data-svelte-h":!0}),h(dt)!=="svelte-hswkmf"&&(dt.innerHTML=fn),To=r(Z),H=u(Z,"DIV",{class:!0});var Y=x(H);f(Be.$$.fragment,Y),Mo=r(Y),lt=u(Y,"P",{"data-svelte-h":!0}),h(lt)!=="svelte-1i4ge4m"&&(lt.innerHTML=gn),bo=r(Y),f(ae.$$.fragment,Y),yo=r(Y),f(re.$$.fragment,Y),Y.forEach(s),Z.forEach(s),At=r(e),f(Ne.$$.fragment,e),Pt=r(e),C=u(e,"DIV",{class:!0});var W=x(C);f(Ve.$$.fragment,W),wo=r(W),ct=u(W,"P",{"data-svelte-h":!0}),h(ct)!=="svelte-1r92lz9"&&(ct.textContent=_n),ko=r(W),pt=u(W,"P",{"data-svelte-h":!0}),h(pt)!=="svelte-q52n56"&&(pt.innerHTML=Tn),vo=r(W),ut=u(W,"P",{"data-svelte-h":!0}),h(ut)!=="svelte-hswkmf"&&(ut.innerHTML=Mn),Uo=r(W),S=u(W,"DIV",{class:!0});var O=x(S);f(Re.$$.fragment,O),$o=r(O),ht=u(O,"P",{"data-svelte-h":!0}),h(ht)!=="svelte-avrbku"&&(ht.innerHTML=bn),Co=r(O),f(ie.$$.fragment,O),Jo=r(O),f(de.$$.fragment,O),O.forEach(s),W.forEach(s),Yt=r(e),f(Xe.$$.fragment,e),Ot=r(e),J=u(e,"DIV",{class:!0});var G=x(J);f(He.$$.fragment,G),jo=r(G),mt=u(G,"P",{"data-svelte-h":!0}),h(mt)!=="svelte-1o18zcb"&&(mt.textContent=yn),zo=r(G),ft=u(G,"P",{"data-svelte-h":!0}),h(ft)!=="svelte-q52n56"&&(ft.innerHTML=wn),xo=r(G),gt=u(G,"P",{"data-svelte-h":!0}),h(gt)!=="svelte-hswkmf"&&(gt.innerHTML=kn),Fo=r(G),F=u(G,"DIV",{class:!0});var B=x(F);f(Se.$$.fragment,B),qo=r(B),_t=u(B,"P",{"data-svelte-h":!0}),h(_t)!=="svelte-1n0zlye"&&(_t.innerHTML=vn),Io=r(B),f(le.$$.fragment,B),Zo=r(B),f(ce.$$.fragment,B),Wo=r(B),f(pe.$$.fragment,B),B.forEach(s),G.forEach(s),Dt=r(e),f(Qe.$$.fragment,e),Kt=r(e),j=u(e,"DIV",{class:!0});var N=x(j);f(Ee.$$.fragment,N),Go=r(N),Tt=u(N,"P",{"data-svelte-h":!0}),h(Tt)!=="svelte-1tffayb"&&(Tt.textContent=Un),Bo=r(N),Mt=u(N,"P",{"data-svelte-h":!0}),h(Mt)!=="svelte-q52n56"&&(Mt.innerHTML=$n),No=r(N),bt=u(N,"P",{"data-svelte-h":!0}),h(bt)!=="svelte-hswkmf"&&(bt.innerHTML=Cn),Vo=r(N),Q=u(N,"DIV",{class:!0});var D=x(Q);f(Le.$$.fragment,D),Ro=r(D),yt=u(D,"P",{"data-svelte-h":!0}),h(yt)!=="svelte-nxq37w"&&(yt.innerHTML=Jn),Xo=r(D),f(ue.$$.fragment,D),Ho=r(D),f(he.$$.fragment,D),D.forEach(s),N.forEach(s),eo=r(e),f(Ae.$$.fragment,e),to=r(e),z=u(e,"DIV",{class:!0});var V=x(z);f(Pe.$$.fragment,V),So=r(V),wt=u(V,"P",{"data-svelte-h":!0}),h(wt)!=="svelte-7n8lnu"&&(wt.innerHTML=jn),Qo=r(V),kt=u(V,"P",{"data-svelte-h":!0}),h(kt)!=="svelte-q52n56"&&(kt.innerHTML=zn),Eo=r(V),vt=u(V,"P",{"data-svelte-h":!0}),h(vt)!=="svelte-hswkmf"&&(vt.innerHTML=xn),Lo=r(V),E=u(V,"DIV",{class:!0});var K=x(E);f(Ye.$$.fragment,K),Ao=r(K),Ut=u(K,"P",{"data-svelte-h":!0}),h(Ut)!=="svelte-109n9n6"&&(Ut.innerHTML=Fn),Po=r(K),f(me.$$.fragment,K),Yo=r(K),f(fe.$$.fragment,K),K.forEach(s),V.forEach(s),oo=r(e),f(Oe.$$.fragment,e),no=r(e),$t=u(e,"P",{}),x($t).forEach(s),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",ts),v(te,"class","flex flex-wrap space-x-1"),v(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){d(document.head,t),c(e,b,n),c(e,i,n),c(e,l,n),c(e,y,n),c(e,w,n),g(_e,e,n),c(e,Ct,n),c(e,te,n),c(e,Jt,n),g(Te,e,n),c(e,jt,n),c(e,Me,n),c(e,zt,n),c(e,be,n),c(e,xt,n),c(e,ye,n),c(e,Ft,n),c(e,we,n),c(e,qt,n),c(e,ke,n),c(e,It,n),c(e,ve,n),c(e,Zt,n),g(Ue,e,n),c(e,Wt,n),c(e,$e,n),c(e,Gt,n),g(Ce,e,n),c(e,Bt,n),c(e,Je,n),c(e,Nt,n),g(je,e,n),c(e,Vt,n),g(ze,e,n),c(e,Rt,n),g(oe,e,n),c(e,Xt,n),g(xe,e,n),c(e,Ht,n),c(e,A,n),g(Fe,A,null),d(A,ao),d(A,et),d(A,ro),d(A,tt),c(e,St,n),g(qe,e,n),c(e,Qt,n),c(e,U,n),g(Ie,U,null),d(U,io),d(U,ot),d(U,lo),d(U,nt),d(U,co),d(U,st),d(U,po),d(U,X),g(Ze,X,null),d(X,uo),d(X,at),d(X,ho),g(ne,X,null),d(X,mo),g(se,X,null),c(e,Et,n),g(We,e,n),c(e,Lt,n),c(e,$,n),g(Ge,$,null),d($,fo),d($,rt),d($,go),d($,it),d($,_o),d($,dt),d($,To),d($,H),g(Be,H,null),d(H,Mo),d(H,lt),d(H,bo),g(ae,H,null),d(H,yo),g(re,H,null),c(e,At,n),g(Ne,e,n),c(e,Pt,n),c(e,C,n),g(Ve,C,null),d(C,wo),d(C,ct),d(C,ko),d(C,pt),d(C,vo),d(C,ut),d(C,Uo),d(C,S),g(Re,S,null),d(S,$o),d(S,ht),d(S,Co),g(ie,S,null),d(S,Jo),g(de,S,null),c(e,Yt,n),g(Xe,e,n),c(e,Ot,n),c(e,J,n),g(He,J,null),d(J,jo),d(J,mt),d(J,zo),d(J,ft),d(J,xo),d(J,gt),d(J,Fo),d(J,F),g(Se,F,null),d(F,qo),d(F,_t),d(F,Io),g(le,F,null),d(F,Zo),g(ce,F,null),d(F,Wo),g(pe,F,null),c(e,Dt,n),g(Qe,e,n),c(e,Kt,n),c(e,j,n),g(Ee,j,null),d(j,Go),d(j,Tt),d(j,Bo),d(j,Mt),d(j,No),d(j,bt),d(j,Vo),d(j,Q),g(Le,Q,null),d(Q,Ro),d(Q,yt),d(Q,Xo),g(ue,Q,null),d(Q,Ho),g(he,Q,null),c(e,eo,n),g(Ae,e,n),c(e,to,n),c(e,z,n),g(Pe,z,null),d(z,So),d(z,wt),d(z,Qo),d(z,kt),d(z,Eo),d(z,vt),d(z,Lo),d(z,E),g(Ye,E,null),d(E,Ao),d(E,Ut),d(E,Po),g(me,E,null),d(E,Yo),g(fe,E,null),c(e,oo,n),g(Oe,e,n),c(e,no,n),c(e,$t,n),so=!0},p(e,[n]){const ee={};n&2&&(ee.$$scope={dirty:n,ctx:e}),oe.$set(ee);const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),ne.$set(I);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),se.$set(P);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),ae.$set(Z);const Y={};n&2&&(Y.$$scope={dirty:n,ctx:e}),re.$set(Y);const W={};n&2&&(W.$$scope={dirty:n,ctx:e}),ie.$set(W);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),de.$set(O);const G={};n&2&&(G.$$scope={dirty:n,ctx:e}),le.$set(G);const B={};n&2&&(B.$$scope={dirty:n,ctx:e}),ce.$set(B);const N={};n&2&&(N.$$scope={dirty:n,ctx:e}),pe.$set(N);const D={};n&2&&(D.$$scope={dirty:n,ctx:e}),ue.$set(D);const V={};n&2&&(V.$$scope={dirty:n,ctx:e}),he.$set(V);const K={};n&2&&(K.$$scope={dirty:n,ctx:e}),me.$set(K);const qn={};n&2&&(qn.$$scope={dirty:n,ctx:e}),fe.$set(qn)},i(e){so||(_(_e.$$.fragment,e),_(Te.$$.fragment,e),_(Ue.$$.fragment,e),_(Ce.$$.fragment,e),_(je.$$.fragment,e),_(ze.$$.fragment,e),_(oe.$$.fragment,e),_(xe.$$.fragment,e),_(Fe.$$.fragment,e),_(qe.$$.fragment,e),_(Ie.$$.fragment,e),_(Ze.$$.fragment,e),_(ne.$$.fragment,e),_(se.$$.fragment,e),_(We.$$.fragment,e),_(Ge.$$.fragment,e),_(Be.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(Ne.$$.fragment,e),_(Ve.$$.fragment,e),_(Re.$$.fragment,e),_(ie.$$.fragment,e),_(de.$$.fragment,e),_(Xe.$$.fragment,e),_(He.$$.fragment,e),_(Se.$$.fragment,e),_(le.$$.fragment,e),_(ce.$$.fragment,e),_(pe.$$.fragment,e),_(Qe.$$.fragment,e),_(Ee.$$.fragment,e),_(Le.$$.fragment,e),_(ue.$$.fragment,e),_(he.$$.fragment,e),_(Ae.$$.fragment,e),_(Pe.$$.fragment,e),_(Ye.$$.fragment,e),_(me.$$.fragment,e),_(fe.$$.fragment,e),_(Oe.$$.fragment,e),so=!0)},o(e){T(_e.$$.fragment,e),T(Te.$$.fragment,e),T(Ue.$$.fragment,e),T(Ce.$$.fragment,e),T(je.$$.fragment,e),T(ze.$$.fragment,e),T(oe.$$.fragment,e),T(xe.$$.fragment,e),T(Fe.$$.fragment,e),T(qe.$$.fragment,e),T(Ie.$$.fragment,e),T(Ze.$$.fragment,e),T(ne.$$.fragment,e),T(se.$$.fragment,e),T(We.$$.fragment,e),T(Ge.$$.fragment,e),T(Be.$$.fragment,e),T(ae.$$.fragment,e),T(re.$$.fragment,e),T(Ne.$$.fragment,e),T(Ve.$$.fragment,e),T(Re.$$.fragment,e),T(ie.$$.fragment,e),T(de.$$.fragment,e),T(Xe.$$.fragment,e),T(He.$$.fragment,e),T(Se.$$.fragment,e),T(le.$$.fragment,e),T(ce.$$.fragment,e),T(pe.$$.fragment,e),T(Qe.$$.fragment,e),T(Ee.$$.fragment,e),T(Le.$$.fragment,e),T(ue.$$.fragment,e),T(he.$$.fragment,e),T(Ae.$$.fragment,e),T(Pe.$$.fragment,e),T(Ye.$$.fragment,e),T(me.$$.fragment,e),T(fe.$$.fragment,e),T(Oe.$$.fragment,e),so=!1},d(e){e&&(s(b),s(i),s(l),s(y),s(w),s(Ct),s(te),s(Jt),s(jt),s(Me),s(zt),s(be),s(xt),s(ye),s(Ft),s(we),s(qt),s(ke),s(It),s(ve),s(Zt),s(Wt),s($e),s(Gt),s(Bt),s(Je),s(Nt),s(Vt),s(Rt),s(Xt),s(Ht),s(A),s(St),s(Qt),s(U),s(Et),s(Lt),s($),s(At),s(Pt),s(C),s(Yt),s(Ot),s(J),s(Dt),s(Kt),s(j),s(eo),s(to),s(z),s(oo),s(no),s($t)),s(t),M(_e,e),M(Te,e),M(Ue,e),M(Ce,e),M(je,e),M(ze,e),M(oe,e),M(xe,e),M(Fe),M(qe,e),M(Ie),M(Ze),M(ne),M(se),M(We,e),M(Ge),M(Be),M(ae),M(re),M(Ne,e),M(Ve),M(Re),M(ie),M(de),M(Xe,e),M(He),M(Se),M(le),M(ce),M(pe),M(Qe,e),M(Ee),M(Le),M(ue),M(he),M(Ae,e),M(Pe),M(Ye),M(me),M(fe),M(Oe,e)}}}const ts='{"title":"UMT5","local":"umt5","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Differences with mT5?","local":"differences-with-mt5","sections":[],"depth":2}],"depth":1}';function os(k){return Zn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class cs extends Wn{constructor(t){super(),Gn(this,t,os,es,In,{})}}export{cs as component};
