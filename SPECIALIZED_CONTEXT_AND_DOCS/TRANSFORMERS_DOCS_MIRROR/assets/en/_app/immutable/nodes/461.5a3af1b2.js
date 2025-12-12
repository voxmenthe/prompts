import{s as Dn,o as Sn,n as V}from"../chunks/scheduler.18a86fab.js";import{S as An,i as Yn,g as i,s,r as u,A as Qn,h as l,f as a,c as r,j as N,x as m,u as f,k as w,y as o,a as p,v as g,d as _,t as M,w as b}from"../chunks/index.98837b22.js";import{T as zt}from"../chunks/Tip.77304350.js";import{D as P}from"../chunks/Docstring.a1ef7999.js";import{C as jt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Jt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as S,E as Kn}from"../chunks/getInferenceSnippets.06c2775f.js";function es(y){let t,T=`This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: <code>pip install -U transformers==4.40.2</code>.`;return{c(){t=i("p"),t.innerHTML=T},l(c){t=l(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-1sq0hrb"&&(t.innerHTML=T)},m(c,h){p(c,t,h)},p:V,d(c){c&&a(t)}}}function ts(y){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=i("p"),t.innerHTML=T},l(c){t=l(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,h){p(c,t,h)},p:V,d(c){c&&a(t)}}}function os(y){let t,T="Example:",c,h,k;return h=new jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Qcm9waGV0TmV0TW9kZWwlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJwYXRyaWNrdm9ucGxhdGVuJTJGeHByb3BoZXRuZXQtbGFyZ2UtdW5jYXNlZC1zdGFuZGFsb25lJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExNUHJvcGhldE5ldE1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJwYXRyaWNrdm9ucGxhdGVuJTJGeHByb3BoZXRuZXQtbGFyZ2UtdW5jYXNlZC1zdGFuZGFsb25lJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJTdHVkaWVzJTIwaGF2ZSUyMGJlZW4lMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSkuaW5wdXRfaWRzJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQWRlY29kZXJfaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlN0dWRpZXMlMjBzaG93JTIwdGhhdCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RkZWNvZGVyX2lucHV0X2lkcyklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTIwJTIwJTIzJTIwbWFpbiUyMHN0cmVhbSUyMGhpZGRlbiUyMHN0YXRlcyUwQWxhc3RfaGlkZGVuX3N0YXRlc19uZ3JhbSUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGVfbmdyYW0lMjAlMjAlMjMlMjBwcmVkaWN0JTIwaGlkZGVuJTIwc3RhdGVz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMProphetNetModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMProphetNetModel.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state  <span class="hljs-comment"># main stream hidden states</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states_ngram = outputs.last_hidden_state_ngram  <span class="hljs-comment"># predict hidden states</span>`,wrap:!1}}),{c(){t=i("p"),t.textContent=T,c=s(),u(h.$$.fragment)},l(d){t=l(d,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(d),f(h.$$.fragment,d)},m(d,v){p(d,t,v),p(d,c,v),g(h,d,v),k=!0},p:V,i(d){k||(_(h.$$.fragment,d),k=!0)},o(d){M(h.$$.fragment,d),k=!1},d(d){d&&(a(t),a(c)),b(h,d)}}}function ns(y){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=i("p"),t.innerHTML=T},l(c){t=l(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,h){p(c,t,h)},p:V,d(c){c&&a(t)}}}function ss(y){let t,T="Example:",c,h,k;return h=new jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Qcm9waGV0TmV0RW5jb2RlciUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIycGF0cmlja3ZvbnBsYXRlbiUyRnhwcm9waGV0bmV0LWxhcmdlLXVuY2FzZWQtc3RhbmRhbG9uZSUyMiklMEFtb2RlbCUyMCUzRCUyMFhMTVByb3BoZXROZXRFbmNvZGVyLmZyb21fcHJldHJhaW5lZCglMjJwYXRyaWNrdm9ucGxhdGVuJTJGcHJvcGhldG5ldC1sYXJnZS11bmNhc2VkLXN0YW5kYWxvbmUlMjIpJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMProphetNetEncoder
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMProphetNetEncoder.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/prophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=i("p"),t.textContent=T,c=s(),u(h.$$.fragment)},l(d){t=l(d,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(d),f(h.$$.fragment,d)},m(d,v){p(d,t,v),p(d,c,v),g(h,d,v),k=!0},p:V,i(d){k||(_(h.$$.fragment,d),k=!0)},o(d){M(h.$$.fragment,d),k=!1},d(d){d&&(a(t),a(c)),b(h,d)}}}function rs(y){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=i("p"),t.innerHTML=T},l(c){t=l(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,h){p(c,t,h)},p:V,d(c){c&&a(t)}}}function as(y){let t,T="Example:",c,h,k;return h=new jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Qcm9waGV0TmV0RGVjb2RlciUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIycGF0cmlja3ZvbnBsYXRlbiUyRnhwcm9waGV0bmV0LWxhcmdlLXVuY2FzZWQtc3RhbmRhbG9uZSUyMiklMEFtb2RlbCUyMCUzRCUyMFhMTVByb3BoZXROZXREZWNvZGVyLmZyb21fcHJldHJhaW5lZCglMjJwYXRyaWNrdm9ucGxhdGVuJTJGeHByb3BoZXRuZXQtbGFyZ2UtdW5jYXNlZC1zdGFuZGFsb25lJTIyJTJDJTIwYWRkX2Nyb3NzX2F0dGVudGlvbiUzREZhbHNlKSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBbGFzdF9oaWRkZW5fc3RhdGVzJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMProphetNetDecoder
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMProphetNetDecoder.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=i("p"),t.textContent=T,c=s(),u(h.$$.fragment)},l(d){t=l(d,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(d),f(h.$$.fragment,d)},m(d,v){p(d,t,v),p(d,c,v),g(h,d,v),k=!0},p:V,i(d){k||(_(h.$$.fragment,d),k=!0)},o(d){M(h.$$.fragment,d),k=!1},d(d){d&&(a(t),a(c)),b(h,d)}}}function ds(y){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=i("p"),t.innerHTML=T},l(c){t=l(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,h){p(c,t,h)},p:V,d(c){c&&a(t)}}}function is(y){let t,T="Example:",c,h,k;return h=new jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Qcm9waGV0TmV0Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIycGF0cmlja3ZvbnBsYXRlbiUyRnhwcm9waGV0bmV0LWxhcmdlLXVuY2FzZWQtc3RhbmRhbG9uZSUyMiklMEFtb2RlbCUyMCUzRCUyMFhMTVByb3BoZXROZXRGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnBhdHJpY2t2b25wbGF0ZW4lMkZ4cHJvcGhldG5ldC1sYXJnZS11bmNhc2VkLXN0YW5kYWxvbmUlMjIpJTBBJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMlN0dWRpZXMlMjBoYXZlJTIwYmVlbiUyMHNob3duJTIwdGhhdCUyMG93bmluZyUyMGElMjBkb2clMjBpcyUyMGdvb2QlMjBmb3IlMjB5b3UlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBZGVjb2Rlcl9pbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyU3R1ZGllcyUyMHNob3clMjB0aGF0JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikuaW5wdXRfaWRzJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dF9pZHMlMkMlMjBkZWNvZGVyX2lucHV0X2lkcyUzRGRlY29kZXJfaW5wdXRfaWRzKSUwQSUwQWxvZ2l0c19uZXh0X3Rva2VuJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMjAlMjAlMjMlMjBsb2dpdHMlMjB0byUyMHByZWRpY3QlMjBuZXh0JTIwdG9rZW4lMjBhcyUyMHVzdWFsJTBBbG9naXRzX25ncmFtX25leHRfdG9rZW5zJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHNfbmdyYW0lMjAlMjAlMjMlMjBsb2dpdHMlMjB0byUyMHByZWRpY3QlMjAybmQlMkMlMjAzcmQlMkMlMjAuLi4lMjBuZXh0JTIwdG9rZW5z",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMProphetNetForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMProphetNetForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits_next_token = outputs.logits  <span class="hljs-comment"># logits to predict next token as usual</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>logits_ngram_next_tokens = outputs.logits_ngram  <span class="hljs-comment"># logits to predict 2nd, 3rd, ... next tokens</span>`,wrap:!1}}),{c(){t=i("p"),t.textContent=T,c=s(),u(h.$$.fragment)},l(d){t=l(d,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(d),f(h.$$.fragment,d)},m(d,v){p(d,t,v),p(d,c,v),g(h,d,v),k=!0},p:V,i(d){k||(_(h.$$.fragment,d),k=!0)},o(d){M(h.$$.fragment,d),k=!1},d(d){d&&(a(t),a(c)),b(h,d)}}}function ls(y){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=i("p"),t.innerHTML=T},l(c){t=l(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,h){p(c,t,h)},p:V,d(c){c&&a(t)}}}function cs(y){let t,T="Example:",c,h,k;return h=new jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Qcm9waGV0TmV0Rm9yQ2F1c2FsTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnBhdHJpY2t2b25wbGF0ZW4lMkZ4cHJvcGhldG5ldC1sYXJnZS11bmNhc2VkLXN0YW5kYWxvbmUlMjIpJTBBbW9kZWwlMjAlM0QlMjBYTE1Qcm9waGV0TmV0Rm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMnBhdHJpY2t2b25wbGF0ZW4lMkZ4cHJvcGhldG5ldC1sYXJnZS11bmNhc2VkLXN0YW5kYWxvbmUlMjIpJTBBYXNzZXJ0JTIwbW9kZWwuY29uZmlnLmlzX2RlY29kZXIlMkMlMjBmJTIyJTdCbW9kZWwuX19jbGFzc19fJTdEJTIwaGFzJTIwdG8lMjBiZSUyMGNvbmZpZ3VyZWQlMjBhcyUyMGElMjBkZWNvZGVyLiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEElMEElMjMlMjBNb2RlbCUyMGNhbiUyMGFsc28lMjBiZSUyMHVzZWQlMjB3aXRoJTIwRW5jb2RlckRlY29kZXIlMjBmcmFtZXdvcmslMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQmVydFRva2VuaXplciUyQyUyMEVuY29kZXJEZWNvZGVyTW9kZWwlMkMlMjBBdXRvVG9rZW5pemVyJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXJfZW5jJTIwJTNEJTIwQmVydFRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWxhcmdlLXVuY2FzZWQlMjIpJTBBdG9rZW5pemVyX2RlYyUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnBhdHJpY2t2b25wbGF0ZW4lMkZ4cHJvcGhldG5ldC1sYXJnZS11bmNhc2VkLXN0YW5kYWxvbmUlMjIpJTBBbW9kZWwlMjAlM0QlMjBFbmNvZGVyRGVjb2Rlck1vZGVsLmZyb21fZW5jb2Rlcl9kZWNvZGVyX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWxhcmdlLXVuY2FzZWQlMjIlMkMlMjAlMjJwYXRyaWNrdm9ucGxhdGVuJTJGeHByb3BoZXRuZXQtbGFyZ2UtdW5jYXNlZC1zdGFuZGFsb25lJTIyJTBBKSUwQSUwQUFSVElDTEUlMjAlM0QlMjAoJTBBJTIwJTIwJTIwJTIwJTIydGhlJTIwdXMlMjBzdGF0ZSUyMGRlcGFydG1lbnQlMjBzYWlkJTIwd2VkbmVzZGF5JTIwaXQlMjBoYWQlMjByZWNlaXZlZCUyMG5vJTIwJTIyJTBBJTIwJTIwJTIwJTIwJTIyZm9ybWFsJTIwd29yZCUyMGZyb20lMjBib2xpdmlhJTIwdGhhdCUyMGl0JTIwd2FzJTIwZXhwZWxsaW5nJTIwdGhlJTIwdXMlMjBhbWJhc3NhZG9yJTIwdGhlcmUlMjAlMjIlMEElMjAlMjAlMjAlMjAlMjJidXQlMjBzYWlkJTIwdGhlJTIwY2hhcmdlcyUyMG1hZGUlMjBhZ2FpbnN0JTIwaGltJTIwYXJlJTIwJTYwJTYwJTIwYmFzZWxlc3MlMjAuJTIyJTBBKSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcl9lbmMoQVJUSUNMRSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplcl9kZWMoJTBBJTIwJTIwJTIwJTIwJTIydXMlMjByZWplY3RzJTIwY2hhcmdlcyUyMGFnYWluc3QlMjBpdHMlMjBhbWJhc3NhZG9yJTIwaW4lMjBib2xpdmlhJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSkuaW5wdXRfaWRzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUzRGlucHV0X2lkcyUyQyUyMGRlY29kZXJfaW5wdXRfaWRzJTNEbGFiZWxzJTVCJTNBJTJDJTIwJTNBLTElNUQlMkMlMjBsYWJlbHMlM0RsYWJlbHMlNUIlM0ElMkMlMjAxJTNBJTVEKSUwQSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3M=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMProphetNetForCausalLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMProphetNetForCausalLM.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Model can also be used with EncoderDecoder framework</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, EncoderDecoderModel, AutoTokenizer
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer_enc = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer_dec = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EncoderDecoderModel.from_encoder_decoder_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google-bert/bert-large-uncased&quot;</span>, <span class="hljs-string">&quot;patrickvonplaten/xprophetnet-large-uncased-standalone&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ARTICLE = (
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;the us state department said wednesday it had received no &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;formal word from bolivia that it was expelling the us ambassador there &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;but said the charges made against him are \`\` baseless .&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer_enc(ARTICLE, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer_dec(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;us rejects charges against its ambassador in bolivia&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=labels[:, :-<span class="hljs-number">1</span>], labels=labels[:, <span class="hljs-number">1</span>:])

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`,wrap:!1}}),{c(){t=i("p"),t.textContent=T,c=s(),u(h.$$.fragment)},l(d){t=l(d,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(d),f(h.$$.fragment,d)},m(d,v){p(d,t,v),p(d,c,v),g(h,d,v),k=!0},p:V,i(d){k||(_(h.$$.fragment,d),k=!0)},o(d){M(h.$$.fragment,d),k=!1},d(d){d&&(a(t),a(c)),b(h,d)}}}function ps(y){let t,T,c,h,k,d="<em>This model was released on 2020-01-13 and added to Hugging Face Transformers on 2023-06-20.</em>",v,_e,qt,Q,dn='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Ft,K,It,ee,ln='<a href="https://huggingface.co/models?filter=xprophetnet"><img alt="Models" src="https://img.shields.io/badge/All_model_pages-xprophetnet-blueviolet"/></a> <a href="https://huggingface.co/spaces/docs-demos/xprophetnet-large-wiki100-cased-xglue-ntg"><img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>',Zt,Me,cn=`<strong>DISCLAIMER:</strong> If you see something strange, file a <a href="https://github.com/huggingface/transformers/issues/new?assignees=&amp;labels=&amp;template=bug-report.md&amp;title" rel="nofollow">Github Issue</a> and assign
@patrickvonplaten`,Gt,be,Wt,Te,pn=`The XLM-ProphetNet model was proposed in <a href="https://huggingface.co/papers/2001.04063" rel="nofollow">ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,</a> by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.`,Bt,ke,hn=`XLM-ProphetNet is an encoder-decoder model and can predict n-future tokens for “ngram” language modeling instead of
just the next token. Its architecture is identical to ProhpetNet, but the model was trained on the multi-lingual
“wiki100” Wikipedia dump. XLM-ProphetNet’s model architecture and pretraining objective is same as ProphetNet, but XLM-ProphetNet was pre-trained on the cross-lingual dataset XGLUE.`,Ut,ye,mn="The abstract from the paper is the following:",Ht,ve,un=`<em>In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel
self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of
the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by
n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time
step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent
overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale
dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for
abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new
state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus.</em>`,Rt,we,fn='The Authors’ code can be found <a href="https://github.com/microsoft/ProphetNet" rel="nofollow">here</a>.',Vt,Le,Et,Ne,gn='<li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/translation">Translation task guide</a></li> <li><a href="../tasks/summarization">Summarization task guide</a></li>',Ot,Xe,Dt,U,xe,po,et,_n=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel">XLMProphetNetModel</a>. It is used to instantiate a
XLMProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the XLMProphetNet
<a href="https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased" rel="nofollow">microsoft/xprophetnet-large-wiki100-cased</a>
architecture.`,ho,tt,Mn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,St,$e,At,L,Pe,mo,ot,bn=`Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.`,uo,nt,Tn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,fo,E,ze,go,st,kn=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A XLMProphetNet sequence has the following format:`,_o,rt,yn="<li>single sequence: <code>X [SEP]</code></li> <li>pair of sequences: <code>A [SEP] B [SEP]</code></li>",Mo,te,Ce,bo,at,vn="Converts a sequence of tokens (strings for sub-words) in a single string.",To,oe,Je,ko,dt,wn=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLMProphetNet
does not make use of token type ids, therefore a list of zeros is returned.`,yo,ne,je,vo,it,Ln=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Yt,qe,Qt,z,Fe,wo,lt,Nn=`The bare XLMProphetNet Model outputting raw hidden-states without any specific head on top.
This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Lo,ct,Xn=`Original ProphetNet code can be found <a href="https://github.com/microsoft/ProphetNet" rel="nofollow">here</a>. Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file <code>convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py</code>.`,No,pt,xn=`This model is a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.`,Xo,I,Ie,xo,ht,$n='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel">XLMProphetNetModel</a> forward method, overrides the <code>__call__</code> special method.',$o,se,Po,re,Kt,Ze,eo,X,Ge,zo,mt,Pn=`The standalone encoder part of the XLMProphetNetModel.
This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Co,ut,zn=`Original ProphetNet code can be found <a href="https://github.com/microsoft/ProphetNet" rel="nofollow">here</a>. Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file <code>convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py</code>.`,Jo,ft,Cn=`This model is a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.`,jo,gt,Jn=`word_embeddings  (<code>torch.nn.Embeddings</code> of shape <code>(config.vocab_size, config.hidden_size)</code>, <em>optional</em>):
The word embedding parameters. This can be used to initialize <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder">XLMProphetNetEncoder</a> with pre-defined word
embeddings instead of randomly initialized word embeddings.`,qo,Z,We,Fo,_t,jn='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder">XLMProphetNetEncoder</a> forward method, overrides the <code>__call__</code> special method.',Io,ae,Zo,de,to,Be,oo,x,Ue,Go,Mt,qn=`The standalone decoder part of the XLMProphetNetModel.
This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Wo,bt,Fn=`Original ProphetNet code can be found <a href="https://github.com/microsoft/ProphetNet" rel="nofollow">here</a>. Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file <code>convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py</code>.`,Bo,Tt,In=`This model is a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.`,Uo,kt,Zn=`word_embeddings  (<code>torch.nn.Embeddings</code> of shape <code>(config.vocab_size, config.hidden_size)</code>, <em>optional</em>):
The word embedding parameters. This can be used to initialize <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetEncoder">XLMProphetNetEncoder</a> with pre-defined word
embeddings instead of randomly initialized word embeddings.`,Ho,G,He,Ro,yt,Gn='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetDecoder">XLMProphetNetDecoder</a> forward method, overrides the <code>__call__</code> special method.',Vo,ie,Eo,le,no,Re,so,C,Ve,Oo,vt,Wn=`The XLMProphetNet Model with a language modeling head. Can be used for sequence generation tasks.
This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Do,wt,Bn=`Original ProphetNet code can be found <a href="https://github.com/microsoft/ProphetNet" rel="nofollow">here</a>. Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file <code>convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py</code>.`,So,Lt,Un=`This model is a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.`,Ao,W,Ee,Yo,Nt,Hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForConditionalGeneration">XLMProphetNetForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Qo,ce,Ko,pe,ro,Oe,ao,J,De,en,Xt,Rn=`The standalone decoder part of the XLMProphetNetModel with a lm head on top. The model can be used for causal language modeling.
This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,tn,xt,Vn=`Original ProphetNet code can be found <a href="https://github.com/microsoft/ProphetNet" rel="nofollow">here</a>. Checkpoints were converted
from original Fairseq checkpoints. For more information on the checkpoint conversion, please take a look at the
file <code>convert_prophetnet_original_pytorch_checkpoint_to_pytorch.py</code>.`,on,$t,En=`This model is a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
behavior.`,nn,B,Se,sn,Pt,On='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetForCausalLM">XLMProphetNetForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',rn,he,an,me,io,Ae,lo,Ct,co;return _e=new S({props:{title:"XLM-ProphetNet",local:"xlm-prophetnet",headingTag:"h1"}}),K=new zt({props:{warning:!0,$$slots:{default:[es]},$$scope:{ctx:y}}}),be=new S({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Le=new S({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Xe=new S({props:{title:"XLMProphetNetConfig",local:"transformers.XLMProphetNetConfig",headingTag:"h2"}}),xe=new P({props:{name:"class transformers.XLMProphetNetConfig",anchor:"transformers.XLMProphetNetConfig",parameters:[{name:"activation_dropout",val:": typing.Optional[float] = 0.1"},{name:"activation_function",val:": typing.Union[str, typing.Callable, NoneType] = 'gelu'"},{name:"vocab_size",val:": typing.Optional[int] = 30522"},{name:"hidden_size",val:": typing.Optional[int] = 1024"},{name:"encoder_ffn_dim",val:": typing.Optional[int] = 4096"},{name:"num_encoder_layers",val:": typing.Optional[int] = 12"},{name:"num_encoder_attention_heads",val:": typing.Optional[int] = 16"},{name:"decoder_ffn_dim",val:": typing.Optional[int] = 4096"},{name:"num_decoder_layers",val:": typing.Optional[int] = 12"},{name:"num_decoder_attention_heads",val:": typing.Optional[int] = 16"},{name:"attention_dropout",val:": typing.Optional[float] = 0.1"},{name:"dropout",val:": typing.Optional[float] = 0.1"},{name:"max_position_embeddings",val:": typing.Optional[int] = 512"},{name:"init_std",val:": typing.Optional[float] = 0.02"},{name:"is_encoder_decoder",val:": typing.Optional[bool] = True"},{name:"add_cross_attention",val:": typing.Optional[bool] = True"},{name:"decoder_start_token_id",val:": typing.Optional[int] = 0"},{name:"ngram",val:": typing.Optional[int] = 2"},{name:"num_buckets",val:": typing.Optional[int] = 32"},{name:"relative_max_distance",val:": typing.Optional[int] = 128"},{name:"disable_ngram_loss",val:": typing.Optional[bool] = False"},{name:"eps",val:": typing.Optional[float] = 0.0"},{name:"use_cache",val:": typing.Optional[bool] = True"},{name:"pad_token_id",val:": typing.Optional[int] = 0"},{name:"bos_token_id",val:": typing.Optional[int] = 1"},{name:"eos_token_id",val:": typing.Optional[int] = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLMProphetNetConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.XLMProphetNetConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.XLMProphetNetConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetModel">XLMProphetNetModel</a>.`,name:"vocab_size"},{anchor:"transformers.XLMProphetNetConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.XLMProphetNetConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.XLMProphetNetConfig.num_encoder_layers",description:`<strong>num_encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"num_encoder_layers"},{anchor:"transformers.XLMProphetNetConfig.num_encoder_attention_heads",description:`<strong>num_encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_encoder_attention_heads"},{anchor:"transformers.XLMProphetNetConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the <code>intermediate</code> (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.XLMProphetNetConfig.num_decoder_layers",description:`<strong>num_decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"num_decoder_layers"},{anchor:"transformers.XLMProphetNetConfig.num_decoder_attention_heads",description:`<strong>num_decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_decoder_attention_heads"},{anchor:"transformers.XLMProphetNetConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.XLMProphetNetConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.XLMProphetNetConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.XLMProphetNetConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.XLMProphetNetConfig.add_cross_attention",description:`<strong>add_cross_attention</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether cross-attention layers should be added to the model.`,name:"add_cross_attention"},{anchor:"transformers.XLMProphetNetConfig.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether this is an encoder/decoder model.`,name:"is_encoder_decoder"},{anchor:"transformers.XLMProphetNetConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.XLMProphetNetConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.XLMProphetNetConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.XLMProphetNetConfig.ngram",description:`<strong>ngram</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first
token.`,name:"ngram"},{anchor:"transformers.XLMProphetNetConfig.num_buckets",description:`<strong>num_buckets</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of buckets to use for each attention layer. This is for relative position calculation. See the
[T5 paper](see <a href="https://huggingface.co/papers/1910.10683" rel="nofollow">https://huggingface.co/papers/1910.10683</a>) for more details.`,name:"num_buckets"},{anchor:"transformers.XLMProphetNetConfig.relative_max_distance",description:`<strong>relative_max_distance</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Relative distances greater than this number will be put into the last same bucket. This is for relative
position calculation. See the [T5 paper](see <a href="https://huggingface.co/papers/1910.10683" rel="nofollow">https://huggingface.co/papers/1910.10683</a>) for more details.`,name:"relative_max_distance"},{anchor:"transformers.XLMProphetNetConfig.disable_ngram_loss",description:`<strong>disable_ngram_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether be trained predicting only the next first token.`,name:"disable_ngram_loss"},{anchor:"transformers.XLMProphetNetConfig.eps",description:`<strong>eps</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Controls the <code>epsilon</code> parameter value for label smoothing in the loss calculation. If set to 0, no label
smoothing is performed.`,name:"eps"},{anchor:"transformers.XLMProphetNetConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/configuration_xlm_prophetnet.py#L26"}}),$e=new S({props:{title:"XLMProphetNetTokenizer",local:"transformers.XLMProphetNetTokenizer",headingTag:"h2"}}),Pe=new P({props:{name:"class transformers.XLMProphetNetTokenizer",anchor:"transformers.XLMProphetNetTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '[SEP]'"},{name:"eos_token",val:" = '[SEP]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"unk_token",val:" = '[UNK]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLMProphetNetTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.XLMProphetNetTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.XLMProphetNetTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.XLMProphetNetTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.XLMProphetNetTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.XLMProphetNetTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.XLMProphetNetTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.XLMProphetNetTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.XLMProphetNetTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.XLMProphetNetTokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L43"}}),ze=new P({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.XLMProphetNetTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added`,name:"token_ids_0"},{anchor:"transformers.XLMProphetNetTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L296",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ce=new P({props:{name:"convert_tokens_to_string",anchor:"transformers.XLMProphetNetTokenizer.convert_tokens_to_string",parameters:[{name:"tokens",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L274"}}),Je=new P({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.XLMProphetNetTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XLMProphetNetTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L223",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),je=new P({props:{name:"get_special_tokens_mask",anchor:"transformers.XLMProphetNetTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.XLMProphetNetTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XLMProphetNetTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.XLMProphetNetTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/tokenization_xlm_prophetnet.py#L195",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),qe=new S({props:{title:"XLMProphetNetModel",local:"transformers.XLMProphetNetModel",headingTag:"h2"}}),Fe=new P({props:{name:"class transformers.XLMProphetNetModel",anchor:"transformers.XLMProphetNetModel",parameters:[{name:"config",val:": XLMProphetNetConfig"}],parametersDescription:[{anchor:"transformers.XLMProphetNetModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig">XLMProphetNetConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1700"}}),Ie=new P({props:{name:"forward",anchor:"transformers.XLMProphetNetModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMProphetNetModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMProphetNetModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>XLMProphetNet uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.XLMProphetNetModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.XLMProphetNetModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMProphetNetModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.XLMProphetNetModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.XLMProphetNetModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.XLMProphetNetModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XLMProphetNetModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XLMProphetNetModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMProphetNetModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMProphetNetModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1736",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig"
>XLMProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>) — Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>last_hidden_state_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,ngram * decoder_sequence_length, config.vocab_size)</code>, <em>optional</em>) — Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_ngram_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>decoder_ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
compute the weighted average in the</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, encoder_sequence_length, encoder_sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),se=new zt({props:{$$slots:{default:[ts]},$$scope:{ctx:y}}}),re=new Jt({props:{anchor:"transformers.XLMProphetNetModel.forward.example",$$slots:{default:[os]},$$scope:{ctx:y}}}),Ze=new S({props:{title:"XLMProphetNetEncoder",local:"transformers.XLMProphetNetEncoder",headingTag:"h2"}}),Ge=new P({props:{name:"class transformers.XLMProphetNetEncoder",anchor:"transformers.XLMProphetNetEncoder",parameters:[{name:"config",val:": XLMProphetNetConfig"},{name:"word_embeddings",val:": Embedding = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetEncoder.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig">XLMProphetNetConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1229"}}),We=new P({props:{name:"forward",anchor:"transformers.XLMProphetNetEncoder.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetEncoder.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMProphetNetEncoder.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMProphetNetEncoder.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMProphetNetEncoder.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMProphetNetEncoder.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMProphetNetEncoder.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1259",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig"
>XLMProphetNetConfig</a>) and inputs.</p>
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
`}}),ae=new zt({props:{$$slots:{default:[ns]},$$scope:{ctx:y}}}),de=new Jt({props:{anchor:"transformers.XLMProphetNetEncoder.forward.example",$$slots:{default:[ss]},$$scope:{ctx:y}}}),Be=new S({props:{title:"XLMProphetNetDecoder",local:"transformers.XLMProphetNetDecoder",headingTag:"h2"}}),Ue=new P({props:{name:"class transformers.XLMProphetNetDecoder",anchor:"transformers.XLMProphetNetDecoder",parameters:[{name:"config",val:": XLMProphetNetConfig"},{name:"word_embeddings",val:": typing.Optional[torch.nn.modules.sparse.Embedding] = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetDecoder.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig">XLMProphetNetConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1354"}}),He=new P({props:{name:"forward",anchor:"transformers.XLMProphetNetDecoder.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetDecoder.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMProphetNetDecoder.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMProphetNetDecoder.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMProphetNetDecoder.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMProphetNetDecoder.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMProphetNetDecoder.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XLMProphetNetDecoder.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XLMProphetNetDecoder.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:`,name:"encoder_attention_mask"},{anchor:"transformers.XLMProphetNetDecoder.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.XLMProphetNetDecoder.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XLMProphetNetDecoder.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1391",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig"
>XLMProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>) — Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>last_hidden_state_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) — Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>ngram_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
compute the weighted average in the</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ie=new zt({props:{$$slots:{default:[rs]},$$scope:{ctx:y}}}),le=new Jt({props:{anchor:"transformers.XLMProphetNetDecoder.forward.example",$$slots:{default:[as]},$$scope:{ctx:y}}}),Re=new S({props:{title:"XLMProphetNetForConditionalGeneration",local:"transformers.XLMProphetNetForConditionalGeneration",headingTag:"h2"}}),Ve=new P({props:{name:"class transformers.XLMProphetNetForConditionalGeneration",anchor:"transformers.XLMProphetNetForConditionalGeneration",parameters:[{name:"config",val:": XLMProphetNetConfig"}],parametersDescription:[{anchor:"transformers.XLMProphetNetForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig">XLMProphetNetConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1831"}}),Ee=new P({props:{name:"forward",anchor:"transformers.XLMProphetNetForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>XLMProphetNet uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[-100, 0, ..., config.vocab_size - 1]</code>. All labels set to <code>-100</code> are ignored (masked), the loss is only computed for
labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L1852",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig"
>XLMProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, config.vocab_size)</code>) — Prediction scores of the main stream language modeling head (scores for each vocabulary token before
SoftMax).</p>
</li>
<li>
<p><strong>logits_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) — Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_ngram_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>decoder_ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
compute the weighted average in the</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, encoder_sequence_length, encoder_sequence_length)</code>. Attentions weights of the encoder, after the attention
softmax, used to compute the weighted average in the self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetSeq2SeqLMOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ce=new zt({props:{$$slots:{default:[ds]},$$scope:{ctx:y}}}),pe=new Jt({props:{anchor:"transformers.XLMProphetNetForConditionalGeneration.forward.example",$$slots:{default:[is]},$$scope:{ctx:y}}}),Oe=new S({props:{title:"XLMProphetNetForCausalLM",local:"transformers.XLMProphetNetForCausalLM",headingTag:"h2"}}),De=new P({props:{name:"class transformers.XLMProphetNetForCausalLM",anchor:"transformers.XLMProphetNetForCausalLM",parameters:[{name:"config",val:": XLMProphetNetConfig"}],parametersDescription:[{anchor:"transformers.XLMProphetNetForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig">XLMProphetNetConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L2041"}}),Se=new P({props:{name:"forward",anchor:"transformers.XLMProphetNetForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMProphetNetForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
it.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(encoder_layers, encoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:`,name:"encoder_attention_mask"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"use_cache"},{anchor:"transformers.XLMProphetNetForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/xlm_prophetnet/modeling_xlm_prophetnet.py#L2080",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-prophetnet#transformers.XLMProphetNetConfig"
>XLMProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, config.vocab_size)</code>) — Prediction scores of the main stream language modeling head (scores for each vocabulary token before
SoftMax).</p>
</li>
<li>
<p><strong>logits_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) — Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of main stream of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>ngram_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, encoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the cross-attention layer of the decoder, after the attention softmax, used to
compute the weighted average in the</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.deprecated.xlm_prophetnet.modeling_xlm_prophetnet.XLMProphetNetDecoderLMOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),he=new zt({props:{$$slots:{default:[ls]},$$scope:{ctx:y}}}),me=new Jt({props:{anchor:"transformers.XLMProphetNetForCausalLM.forward.example",$$slots:{default:[cs]},$$scope:{ctx:y}}}),Ae=new Kn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xlm-prophetnet.md"}}),{c(){t=i("meta"),T=s(),c=i("p"),h=s(),k=i("p"),k.innerHTML=d,v=s(),u(_e.$$.fragment),qt=s(),Q=i("div"),Q.innerHTML=dn,Ft=s(),u(K.$$.fragment),It=s(),ee=i("div"),ee.innerHTML=ln,Zt=s(),Me=i("p"),Me.innerHTML=cn,Gt=s(),u(be.$$.fragment),Wt=s(),Te=i("p"),Te.innerHTML=pn,Bt=s(),ke=i("p"),ke.textContent=hn,Ut=s(),ye=i("p"),ye.textContent=mn,Ht=s(),ve=i("p"),ve.innerHTML=un,Rt=s(),we=i("p"),we.innerHTML=fn,Vt=s(),u(Le.$$.fragment),Et=s(),Ne=i("ul"),Ne.innerHTML=gn,Ot=s(),u(Xe.$$.fragment),Dt=s(),U=i("div"),u(xe.$$.fragment),po=s(),et=i("p"),et.innerHTML=_n,ho=s(),tt=i("p"),tt.innerHTML=Mn,St=s(),u($e.$$.fragment),At=s(),L=i("div"),u(Pe.$$.fragment),mo=s(),ot=i("p"),ot.innerHTML=bn,uo=s(),nt=i("p"),nt.innerHTML=Tn,fo=s(),E=i("div"),u(ze.$$.fragment),go=s(),st=i("p"),st.textContent=kn,_o=s(),rt=i("ul"),rt.innerHTML=yn,Mo=s(),te=i("div"),u(Ce.$$.fragment),bo=s(),at=i("p"),at.textContent=vn,To=s(),oe=i("div"),u(Je.$$.fragment),ko=s(),dt=i("p"),dt.textContent=wn,yo=s(),ne=i("div"),u(je.$$.fragment),vo=s(),it=i("p"),it.innerHTML=Ln,Yt=s(),u(qe.$$.fragment),Qt=s(),z=i("div"),u(Fe.$$.fragment),wo=s(),lt=i("p"),lt.innerHTML=Nn,Lo=s(),ct=i("p"),ct.innerHTML=Xn,No=s(),pt=i("p"),pt.innerHTML=xn,Xo=s(),I=i("div"),u(Ie.$$.fragment),xo=s(),ht=i("p"),ht.innerHTML=$n,$o=s(),u(se.$$.fragment),Po=s(),u(re.$$.fragment),Kt=s(),u(Ze.$$.fragment),eo=s(),X=i("div"),u(Ge.$$.fragment),zo=s(),mt=i("p"),mt.innerHTML=Pn,Co=s(),ut=i("p"),ut.innerHTML=zn,Jo=s(),ft=i("p"),ft.innerHTML=Cn,jo=s(),gt=i("p"),gt.innerHTML=Jn,qo=s(),Z=i("div"),u(We.$$.fragment),Fo=s(),_t=i("p"),_t.innerHTML=jn,Io=s(),u(ae.$$.fragment),Zo=s(),u(de.$$.fragment),to=s(),u(Be.$$.fragment),oo=s(),x=i("div"),u(Ue.$$.fragment),Go=s(),Mt=i("p"),Mt.innerHTML=qn,Wo=s(),bt=i("p"),bt.innerHTML=Fn,Bo=s(),Tt=i("p"),Tt.innerHTML=In,Uo=s(),kt=i("p"),kt.innerHTML=Zn,Ho=s(),G=i("div"),u(He.$$.fragment),Ro=s(),yt=i("p"),yt.innerHTML=Gn,Vo=s(),u(ie.$$.fragment),Eo=s(),u(le.$$.fragment),no=s(),u(Re.$$.fragment),so=s(),C=i("div"),u(Ve.$$.fragment),Oo=s(),vt=i("p"),vt.innerHTML=Wn,Do=s(),wt=i("p"),wt.innerHTML=Bn,So=s(),Lt=i("p"),Lt.innerHTML=Un,Ao=s(),W=i("div"),u(Ee.$$.fragment),Yo=s(),Nt=i("p"),Nt.innerHTML=Hn,Qo=s(),u(ce.$$.fragment),Ko=s(),u(pe.$$.fragment),ro=s(),u(Oe.$$.fragment),ao=s(),J=i("div"),u(De.$$.fragment),en=s(),Xt=i("p"),Xt.innerHTML=Rn,tn=s(),xt=i("p"),xt.innerHTML=Vn,on=s(),$t=i("p"),$t.innerHTML=En,nn=s(),B=i("div"),u(Se.$$.fragment),sn=s(),Pt=i("p"),Pt.innerHTML=On,rn=s(),u(he.$$.fragment),an=s(),u(me.$$.fragment),io=s(),u(Ae.$$.fragment),lo=s(),Ct=i("p"),this.h()},l(e){const n=Qn("svelte-u9bgzb",document.head);t=l(n,"META",{name:!0,content:!0}),n.forEach(a),T=r(e),c=l(e,"P",{}),N(c).forEach(a),h=r(e),k=l(e,"P",{"data-svelte-h":!0}),m(k)!=="svelte-1k0h2bc"&&(k.innerHTML=d),v=r(e),f(_e.$$.fragment,e),qt=r(e),Q=l(e,"DIV",{class:!0,"data-svelte-h":!0}),m(Q)!=="svelte-13t8s2t"&&(Q.innerHTML=dn),Ft=r(e),f(K.$$.fragment,e),It=r(e),ee=l(e,"DIV",{class:!0,"data-svelte-h":!0}),m(ee)!=="svelte-u6l7ab"&&(ee.innerHTML=ln),Zt=r(e),Me=l(e,"P",{"data-svelte-h":!0}),m(Me)!=="svelte-s752y"&&(Me.innerHTML=cn),Gt=r(e),f(be.$$.fragment,e),Wt=r(e),Te=l(e,"P",{"data-svelte-h":!0}),m(Te)!=="svelte-j59vg9"&&(Te.innerHTML=pn),Bt=r(e),ke=l(e,"P",{"data-svelte-h":!0}),m(ke)!=="svelte-u8v14v"&&(ke.textContent=hn),Ut=r(e),ye=l(e,"P",{"data-svelte-h":!0}),m(ye)!=="svelte-vfdo9a"&&(ye.textContent=mn),Ht=r(e),ve=l(e,"P",{"data-svelte-h":!0}),m(ve)!=="svelte-1jvtdli"&&(ve.innerHTML=un),Rt=r(e),we=l(e,"P",{"data-svelte-h":!0}),m(we)!=="svelte-mvxxnf"&&(we.innerHTML=fn),Vt=r(e),f(Le.$$.fragment,e),Et=r(e),Ne=l(e,"UL",{"data-svelte-h":!0}),m(Ne)!=="svelte-jwyjs9"&&(Ne.innerHTML=gn),Ot=r(e),f(Xe.$$.fragment,e),Dt=r(e),U=l(e,"DIV",{class:!0});var A=N(U);f(xe.$$.fragment,A),po=r(A),et=l(A,"P",{"data-svelte-h":!0}),m(et)!=="svelte-13qeqsq"&&(et.innerHTML=_n),ho=r(A),tt=l(A,"P",{"data-svelte-h":!0}),m(tt)!=="svelte-1ek1ss9"&&(tt.innerHTML=Mn),A.forEach(a),St=r(e),f($e.$$.fragment,e),At=r(e),L=l(e,"DIV",{class:!0});var $=N(L);f(Pe.$$.fragment,$),mo=r($),ot=l($,"P",{"data-svelte-h":!0}),m(ot)!=="svelte-19vr0qz"&&(ot.innerHTML=bn),uo=r($),nt=l($,"P",{"data-svelte-h":!0}),m(nt)!=="svelte-ntrhio"&&(nt.innerHTML=Tn),fo=r($),E=l($,"DIV",{class:!0});var Y=N(E);f(ze.$$.fragment,Y),go=r(Y),st=l(Y,"P",{"data-svelte-h":!0}),m(st)!=="svelte-c34cyj"&&(st.textContent=kn),_o=r(Y),rt=l(Y,"UL",{"data-svelte-h":!0}),m(rt)!=="svelte-rua507"&&(rt.innerHTML=yn),Y.forEach(a),Mo=r($),te=l($,"DIV",{class:!0});var Ye=N(te);f(Ce.$$.fragment,Ye),bo=r(Ye),at=l(Ye,"P",{"data-svelte-h":!0}),m(at)!=="svelte-1ne8awa"&&(at.textContent=vn),Ye.forEach(a),To=r($),oe=l($,"DIV",{class:!0});var Qe=N(oe);f(Je.$$.fragment,Qe),ko=r(Qe),dt=l(Qe,"P",{"data-svelte-h":!0}),m(dt)!=="svelte-194ygpb"&&(dt.textContent=wn),Qe.forEach(a),yo=r($),ne=l($,"DIV",{class:!0});var Ke=N(ne);f(je.$$.fragment,Ke),vo=r(Ke),it=l(Ke,"P",{"data-svelte-h":!0}),m(it)!=="svelte-1f4f5kp"&&(it.innerHTML=Ln),Ke.forEach(a),$.forEach(a),Yt=r(e),f(qe.$$.fragment,e),Qt=r(e),z=l(e,"DIV",{class:!0});var F=N(z);f(Fe.$$.fragment,F),wo=r(F),lt=l(F,"P",{"data-svelte-h":!0}),m(lt)!=="svelte-1vhmtog"&&(lt.innerHTML=Nn),Lo=r(F),ct=l(F,"P",{"data-svelte-h":!0}),m(ct)!=="svelte-jbq9y7"&&(ct.innerHTML=Xn),No=r(F),pt=l(F,"P",{"data-svelte-h":!0}),m(pt)!=="svelte-1707pv8"&&(pt.innerHTML=xn),Xo=r(F),I=l(F,"DIV",{class:!0});var H=N(I);f(Ie.$$.fragment,H),xo=r(H),ht=l(H,"P",{"data-svelte-h":!0}),m(ht)!=="svelte-158rsce"&&(ht.innerHTML=$n),$o=r(H),f(se.$$.fragment,H),Po=r(H),f(re.$$.fragment,H),H.forEach(a),F.forEach(a),Kt=r(e),f(Ze.$$.fragment,e),eo=r(e),X=l(e,"DIV",{class:!0});var j=N(X);f(Ge.$$.fragment,j),zo=r(j),mt=l(j,"P",{"data-svelte-h":!0}),m(mt)!=="svelte-5e2bpk"&&(mt.innerHTML=Pn),Co=r(j),ut=l(j,"P",{"data-svelte-h":!0}),m(ut)!=="svelte-jbq9y7"&&(ut.innerHTML=zn),Jo=r(j),ft=l(j,"P",{"data-svelte-h":!0}),m(ft)!=="svelte-1707pv8"&&(ft.innerHTML=Cn),jo=r(j),gt=l(j,"P",{"data-svelte-h":!0}),m(gt)!=="svelte-1vlrknu"&&(gt.innerHTML=Jn),qo=r(j),Z=l(j,"DIV",{class:!0});var R=N(Z);f(We.$$.fragment,R),Fo=r(R),_t=l(R,"P",{"data-svelte-h":!0}),m(_t)!=="svelte-226c4c"&&(_t.innerHTML=jn),Io=r(R),f(ae.$$.fragment,R),Zo=r(R),f(de.$$.fragment,R),R.forEach(a),j.forEach(a),to=r(e),f(Be.$$.fragment,e),oo=r(e),x=l(e,"DIV",{class:!0});var q=N(x);f(Ue.$$.fragment,q),Go=r(q),Mt=l(q,"P",{"data-svelte-h":!0}),m(Mt)!=="svelte-yzg110"&&(Mt.innerHTML=qn),Wo=r(q),bt=l(q,"P",{"data-svelte-h":!0}),m(bt)!=="svelte-jbq9y7"&&(bt.innerHTML=Fn),Bo=r(q),Tt=l(q,"P",{"data-svelte-h":!0}),m(Tt)!=="svelte-1707pv8"&&(Tt.innerHTML=In),Uo=r(q),kt=l(q,"P",{"data-svelte-h":!0}),m(kt)!=="svelte-1vlrknu"&&(kt.innerHTML=Zn),Ho=r(q),G=l(q,"DIV",{class:!0});var ue=N(G);f(He.$$.fragment,ue),Ro=r(ue),yt=l(ue,"P",{"data-svelte-h":!0}),m(yt)!=="svelte-1ds9vs0"&&(yt.innerHTML=Gn),Vo=r(ue),f(ie.$$.fragment,ue),Eo=r(ue),f(le.$$.fragment,ue),ue.forEach(a),q.forEach(a),no=r(e),f(Re.$$.fragment,e),so=r(e),C=l(e,"DIV",{class:!0});var O=N(C);f(Ve.$$.fragment,O),Oo=r(O),vt=l(O,"P",{"data-svelte-h":!0}),m(vt)!=="svelte-7baz44"&&(vt.innerHTML=Wn),Do=r(O),wt=l(O,"P",{"data-svelte-h":!0}),m(wt)!=="svelte-jbq9y7"&&(wt.innerHTML=Bn),So=r(O),Lt=l(O,"P",{"data-svelte-h":!0}),m(Lt)!=="svelte-1707pv8"&&(Lt.innerHTML=Un),Ao=r(O),W=l(O,"DIV",{class:!0});var fe=N(W);f(Ee.$$.fragment,fe),Yo=r(fe),Nt=l(fe,"P",{"data-svelte-h":!0}),m(Nt)!=="svelte-1gclu44"&&(Nt.innerHTML=Hn),Qo=r(fe),f(ce.$$.fragment,fe),Ko=r(fe),f(pe.$$.fragment,fe),fe.forEach(a),O.forEach(a),ro=r(e),f(Oe.$$.fragment,e),ao=r(e),J=l(e,"DIV",{class:!0});var D=N(J);f(De.$$.fragment,D),en=r(D),Xt=l(D,"P",{"data-svelte-h":!0}),m(Xt)!=="svelte-1uqo3zv"&&(Xt.innerHTML=Rn),tn=r(D),xt=l(D,"P",{"data-svelte-h":!0}),m(xt)!=="svelte-jbq9y7"&&(xt.innerHTML=Vn),on=r(D),$t=l(D,"P",{"data-svelte-h":!0}),m($t)!=="svelte-1707pv8"&&($t.innerHTML=En),nn=r(D),B=l(D,"DIV",{class:!0});var ge=N(B);f(Se.$$.fragment,ge),sn=r(ge),Pt=l(ge,"P",{"data-svelte-h":!0}),m(Pt)!=="svelte-kx4yoa"&&(Pt.innerHTML=On),rn=r(ge),f(he.$$.fragment,ge),an=r(ge),f(me.$$.fragment,ge),ge.forEach(a),D.forEach(a),io=r(e),f(Ae.$$.fragment,e),lo=r(e),Ct=l(e,"P",{}),N(Ct).forEach(a),this.h()},h(){w(t,"name","hf:doc:metadata"),w(t,"content",hs),w(Q,"class","flex flex-wrap space-x-1"),w(ee,"class","flex flex-wrap space-x-1"),w(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){o(document.head,t),p(e,T,n),p(e,c,n),p(e,h,n),p(e,k,n),p(e,v,n),g(_e,e,n),p(e,qt,n),p(e,Q,n),p(e,Ft,n),g(K,e,n),p(e,It,n),p(e,ee,n),p(e,Zt,n),p(e,Me,n),p(e,Gt,n),g(be,e,n),p(e,Wt,n),p(e,Te,n),p(e,Bt,n),p(e,ke,n),p(e,Ut,n),p(e,ye,n),p(e,Ht,n),p(e,ve,n),p(e,Rt,n),p(e,we,n),p(e,Vt,n),g(Le,e,n),p(e,Et,n),p(e,Ne,n),p(e,Ot,n),g(Xe,e,n),p(e,Dt,n),p(e,U,n),g(xe,U,null),o(U,po),o(U,et),o(U,ho),o(U,tt),p(e,St,n),g($e,e,n),p(e,At,n),p(e,L,n),g(Pe,L,null),o(L,mo),o(L,ot),o(L,uo),o(L,nt),o(L,fo),o(L,E),g(ze,E,null),o(E,go),o(E,st),o(E,_o),o(E,rt),o(L,Mo),o(L,te),g(Ce,te,null),o(te,bo),o(te,at),o(L,To),o(L,oe),g(Je,oe,null),o(oe,ko),o(oe,dt),o(L,yo),o(L,ne),g(je,ne,null),o(ne,vo),o(ne,it),p(e,Yt,n),g(qe,e,n),p(e,Qt,n),p(e,z,n),g(Fe,z,null),o(z,wo),o(z,lt),o(z,Lo),o(z,ct),o(z,No),o(z,pt),o(z,Xo),o(z,I),g(Ie,I,null),o(I,xo),o(I,ht),o(I,$o),g(se,I,null),o(I,Po),g(re,I,null),p(e,Kt,n),g(Ze,e,n),p(e,eo,n),p(e,X,n),g(Ge,X,null),o(X,zo),o(X,mt),o(X,Co),o(X,ut),o(X,Jo),o(X,ft),o(X,jo),o(X,gt),o(X,qo),o(X,Z),g(We,Z,null),o(Z,Fo),o(Z,_t),o(Z,Io),g(ae,Z,null),o(Z,Zo),g(de,Z,null),p(e,to,n),g(Be,e,n),p(e,oo,n),p(e,x,n),g(Ue,x,null),o(x,Go),o(x,Mt),o(x,Wo),o(x,bt),o(x,Bo),o(x,Tt),o(x,Uo),o(x,kt),o(x,Ho),o(x,G),g(He,G,null),o(G,Ro),o(G,yt),o(G,Vo),g(ie,G,null),o(G,Eo),g(le,G,null),p(e,no,n),g(Re,e,n),p(e,so,n),p(e,C,n),g(Ve,C,null),o(C,Oo),o(C,vt),o(C,Do),o(C,wt),o(C,So),o(C,Lt),o(C,Ao),o(C,W),g(Ee,W,null),o(W,Yo),o(W,Nt),o(W,Qo),g(ce,W,null),o(W,Ko),g(pe,W,null),p(e,ro,n),g(Oe,e,n),p(e,ao,n),p(e,J,n),g(De,J,null),o(J,en),o(J,Xt),o(J,tn),o(J,xt),o(J,on),o(J,$t),o(J,nn),o(J,B),g(Se,B,null),o(B,sn),o(B,Pt),o(B,rn),g(he,B,null),o(B,an),g(me,B,null),p(e,io,n),g(Ae,e,n),p(e,lo,n),p(e,Ct,n),co=!0},p(e,[n]){const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),K.$set(A);const $={};n&2&&($.$$scope={dirty:n,ctx:e}),se.$set($);const Y={};n&2&&(Y.$$scope={dirty:n,ctx:e}),re.$set(Y);const Ye={};n&2&&(Ye.$$scope={dirty:n,ctx:e}),ae.$set(Ye);const Qe={};n&2&&(Qe.$$scope={dirty:n,ctx:e}),de.$set(Qe);const Ke={};n&2&&(Ke.$$scope={dirty:n,ctx:e}),ie.$set(Ke);const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),le.$set(F);const H={};n&2&&(H.$$scope={dirty:n,ctx:e}),ce.$set(H);const j={};n&2&&(j.$$scope={dirty:n,ctx:e}),pe.$set(j);const R={};n&2&&(R.$$scope={dirty:n,ctx:e}),he.$set(R);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),me.$set(q)},i(e){co||(_(_e.$$.fragment,e),_(K.$$.fragment,e),_(be.$$.fragment,e),_(Le.$$.fragment,e),_(Xe.$$.fragment,e),_(xe.$$.fragment,e),_($e.$$.fragment,e),_(Pe.$$.fragment,e),_(ze.$$.fragment,e),_(Ce.$$.fragment,e),_(Je.$$.fragment,e),_(je.$$.fragment,e),_(qe.$$.fragment,e),_(Fe.$$.fragment,e),_(Ie.$$.fragment,e),_(se.$$.fragment,e),_(re.$$.fragment,e),_(Ze.$$.fragment,e),_(Ge.$$.fragment,e),_(We.$$.fragment,e),_(ae.$$.fragment,e),_(de.$$.fragment,e),_(Be.$$.fragment,e),_(Ue.$$.fragment,e),_(He.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),_(Re.$$.fragment,e),_(Ve.$$.fragment,e),_(Ee.$$.fragment,e),_(ce.$$.fragment,e),_(pe.$$.fragment,e),_(Oe.$$.fragment,e),_(De.$$.fragment,e),_(Se.$$.fragment,e),_(he.$$.fragment,e),_(me.$$.fragment,e),_(Ae.$$.fragment,e),co=!0)},o(e){M(_e.$$.fragment,e),M(K.$$.fragment,e),M(be.$$.fragment,e),M(Le.$$.fragment,e),M(Xe.$$.fragment,e),M(xe.$$.fragment,e),M($e.$$.fragment,e),M(Pe.$$.fragment,e),M(ze.$$.fragment,e),M(Ce.$$.fragment,e),M(Je.$$.fragment,e),M(je.$$.fragment,e),M(qe.$$.fragment,e),M(Fe.$$.fragment,e),M(Ie.$$.fragment,e),M(se.$$.fragment,e),M(re.$$.fragment,e),M(Ze.$$.fragment,e),M(Ge.$$.fragment,e),M(We.$$.fragment,e),M(ae.$$.fragment,e),M(de.$$.fragment,e),M(Be.$$.fragment,e),M(Ue.$$.fragment,e),M(He.$$.fragment,e),M(ie.$$.fragment,e),M(le.$$.fragment,e),M(Re.$$.fragment,e),M(Ve.$$.fragment,e),M(Ee.$$.fragment,e),M(ce.$$.fragment,e),M(pe.$$.fragment,e),M(Oe.$$.fragment,e),M(De.$$.fragment,e),M(Se.$$.fragment,e),M(he.$$.fragment,e),M(me.$$.fragment,e),M(Ae.$$.fragment,e),co=!1},d(e){e&&(a(T),a(c),a(h),a(k),a(v),a(qt),a(Q),a(Ft),a(It),a(ee),a(Zt),a(Me),a(Gt),a(Wt),a(Te),a(Bt),a(ke),a(Ut),a(ye),a(Ht),a(ve),a(Rt),a(we),a(Vt),a(Et),a(Ne),a(Ot),a(Dt),a(U),a(St),a(At),a(L),a(Yt),a(Qt),a(z),a(Kt),a(eo),a(X),a(to),a(oo),a(x),a(no),a(so),a(C),a(ro),a(ao),a(J),a(io),a(lo),a(Ct)),a(t),b(_e,e),b(K,e),b(be,e),b(Le,e),b(Xe,e),b(xe),b($e,e),b(Pe),b(ze),b(Ce),b(Je),b(je),b(qe,e),b(Fe),b(Ie),b(se),b(re),b(Ze,e),b(Ge),b(We),b(ae),b(de),b(Be,e),b(Ue),b(He),b(ie),b(le),b(Re,e),b(Ve),b(Ee),b(ce),b(pe),b(Oe,e),b(De),b(Se),b(he),b(me),b(Ae,e)}}}const hs='{"title":"XLM-ProphetNet","local":"xlm-prophetnet","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"XLMProphetNetConfig","local":"transformers.XLMProphetNetConfig","sections":[],"depth":2},{"title":"XLMProphetNetTokenizer","local":"transformers.XLMProphetNetTokenizer","sections":[],"depth":2},{"title":"XLMProphetNetModel","local":"transformers.XLMProphetNetModel","sections":[],"depth":2},{"title":"XLMProphetNetEncoder","local":"transformers.XLMProphetNetEncoder","sections":[],"depth":2},{"title":"XLMProphetNetDecoder","local":"transformers.XLMProphetNetDecoder","sections":[],"depth":2},{"title":"XLMProphetNetForConditionalGeneration","local":"transformers.XLMProphetNetForConditionalGeneration","sections":[],"depth":2},{"title":"XLMProphetNetForCausalLM","local":"transformers.XLMProphetNetForCausalLM","sections":[],"depth":2}],"depth":1}';function ms(y){return Sn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ks extends An{constructor(t){super(),Yn(this,t,ms,ps,Dn,{})}}export{ks as component};
