import{s as Sn,o as Dn,n as B}from"../chunks/scheduler.18a86fab.js";import{S as Yn,i as Pn,g as c,s as a,r as h,A as On,h as p,f as s,c as r,j as v,x as b,u as f,k as L,l as Kn,y as i,a as d,v as g,d as _,t as y,w as M}from"../chunks/index.98837b22.js";import{T as $t}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as te}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as It}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as K,E as es}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as ts,a as os}from"../chunks/HfOption.6641485e.js";function ns(k){let t,m="Click on the LayoutLM models in the right sidebar for more examples of how to apply LayoutLM to different vision and language tasks.";return{c(){t=c("p"),t.textContent=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-1px1x7o"&&(t.textContent=m)},m(n,u){d(n,t,u)},p:B,d(n){n&&s(t)}}}function ss(k){let t,m;return t=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMExheW91dExNRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJpbXBpcmElMkZsYXlvdXRsbS1kb2N1bWVudC1xYSUyMiUyQyUyMGFkZF9wcmVmaXhfc3BhY2UlM0RUcnVlKSUwQW1vZGVsJTIwJTNEJTIwTGF5b3V0TE1Gb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyaW1waXJhJTJGbGF5b3V0bG0tZG9jdW1lbnQtcWElMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYpJTBBJTBBZGF0YXNldCUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJuaWVsc3IlMkZmdW5zZCUyMiUyQyUyMHNwbGl0JTNEJTIydHJhaW4lMjIpJTBBZXhhbXBsZSUyMCUzRCUyMGRhdGFzZXQlNUIwJTVEJTBBcXVlc3Rpb24lMjAlM0QlMjAlMjJ3aGF0J3MlMjBoaXMlMjBuYW1lJTNGJTIyJTBBd29yZHMlMjAlM0QlMjBleGFtcGxlJTVCJTIyd29yZHMlMjIlNUQlMEFib3hlcyUyMCUzRCUyMGV4YW1wbGUlNUIlMjJiYm94ZXMlMjIlNUQlMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjBxdWVzdGlvbi5zcGxpdCgpJTJDJTBBJTIwJTIwJTIwJTIwd29yZHMlMkMlMEElMjAlMjAlMjAlMjBpc19zcGxpdF9pbnRvX3dvcmRzJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMHJldHVybl90b2tlbl90eXBlX2lkcyUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQWJib3glMjAlM0QlMjAlNUIlNUQlMEFmb3IlMjBpJTJDJTIwcyUyQyUyMHclMjBpbiUyMHppcChlbmNvZGluZy5pbnB1dF9pZHMlNUIwJTVEJTJDJTIwZW5jb2Rpbmcuc2VxdWVuY2VfaWRzKDApJTJDJTIwZW5jb2Rpbmcud29yZF9pZHMoMCkpJTNBJTBBJTIwJTIwJTIwJTIwaWYlMjBzJTIwJTNEJTNEJTIwMSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGJib3guYXBwZW5kKGJveGVzJTVCdyU1RCklMEElMjAlMjAlMjAlMjBlbGlmJTIwaSUyMCUzRCUzRCUyMHRva2VuaXplci5zZXBfdG9rZW5faWQlM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBiYm94LmFwcGVuZCglNUIxMDAwJTVEJTIwKiUyMDQpJTBBJTIwJTIwJTIwJTIwZWxzZSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGJib3guYXBwZW5kKCU1QjAlNUQlMjAqJTIwNCklMEFlbmNvZGluZyU1QiUyMmJib3glMjIlNUQlMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCYmJveCU1RCklMEElMEF3b3JkX2lkcyUyMCUzRCUyMGVuY29kaW5nLndvcmRfaWRzKDApJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmcpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXN0YXJ0X3Njb3JlcyUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzJTBBZW5kX3Njb3JlcyUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cyUwQXN0YXJ0JTJDJTIwZW5kJTIwJTNEJTIwd29yZF9pZHMlNUJzdGFydF9zY29yZXMuYXJnbWF4KC0xKSU1RCUyQyUyMHdvcmRfaWRzJTVCZW5kX3Njb3Jlcy5hcmdtYXgoLTEpJTVEJTBBcHJpbnQoJTIyJTIwJTIyLmpvaW4od29yZHMlNUJzdGFydCUyMCUzQSUyMGVuZCUyMCUyQiUyMDElNUQpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LayoutLMForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;impira/layoutlm-document-qa&quot;</span>, add_prefix_space=<span class="hljs-literal">True</span>)
model = LayoutLMForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;impira/layoutlm-document-qa&quot;</span>, dtype=torch.float16)

dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
example = dataset[<span class="hljs-number">0</span>]
question = <span class="hljs-string">&quot;what&#x27;s his name?&quot;</span>
words = example[<span class="hljs-string">&quot;words&quot;</span>]
boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

encoding = tokenizer(
    question.split(),
    words,
    is_split_into_words=<span class="hljs-literal">True</span>,
    return_token_type_ids=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
)
bbox = []
<span class="hljs-keyword">for</span> i, s, w <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(encoding.input_ids[<span class="hljs-number">0</span>], encoding.sequence_ids(<span class="hljs-number">0</span>), encoding.word_ids(<span class="hljs-number">0</span>)):
    <span class="hljs-keyword">if</span> s == <span class="hljs-number">1</span>:
        bbox.append(boxes[w])
    <span class="hljs-keyword">elif</span> i == tokenizer.sep_token_id:
        bbox.append([<span class="hljs-number">1000</span>] * <span class="hljs-number">4</span>)
    <span class="hljs-keyword">else</span>:
        bbox.append([<span class="hljs-number">0</span>] * <span class="hljs-number">4</span>)
encoding[<span class="hljs-string">&quot;bbox&quot;</span>] = torch.tensor([bbox])

word_ids = encoding.word_ids(<span class="hljs-number">0</span>)
outputs = model(**encoding)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
start, end = word_ids[start_scores.argmax(-<span class="hljs-number">1</span>)], word_ids[end_scores.argmax(-<span class="hljs-number">1</span>)]
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot; &quot;</span>.join(words[start : end + <span class="hljs-number">1</span>]))`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,u){g(t,n,u),m=!0},p:B,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){y(t.$$.fragment,n),m=!1},d(n){M(t,n)}}}function as(k){let t,m;return t=new os({props:{id:"usage",option:"AutoModel",$$slots:{default:[ss]},$$scope:{ctx:k}}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,u){g(t,n,u),m=!0},p(n,u){const T={};u&2&&(T.$$scope={dirty:u,ctx:n}),t.$set(T)},i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){y(t.$$.fragment,n),m=!1},d(n){M(t,n)}}}function rs(k){let t,m="Examples:",n,u,T;return u=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExheW91dExNQ29uZmlnJTJDJTIwTGF5b3V0TE1Nb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBMYXlvdXRMTSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwTGF5b3V0TE1Db25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwTGF5b3V0TE1Nb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LayoutLMConfig, LayoutLMModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a LayoutLM configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = LayoutLMConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),h(u.$$.fragment)},l(l){t=p(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-kvfsh7"&&(t.textContent=m),n=r(l),f(u.$$.fragment,l)},m(l,w){d(l,t,w),d(l,n,w),g(u,l,w),T=!0},p:B,i(l){T||(_(u.$$.fragment,l),T=!0)},o(l){y(u.$$.fragment,l),T=!1},d(l){l&&(s(t),s(n)),M(u,l)}}}function is(k){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,u){d(n,t,u)},p:B,d(n){n&&s(t)}}}function ls(k){let t,m="Examples:",n,u,T;return u=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMYXlvdXRMTU1vZGVsJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZsYXlvdXRsbS1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBMYXlvdXRMTU1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZsYXlvdXRsbS1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBd29yZHMlMjAlM0QlMjAlNUIlMjJIZWxsbyUyMiUyQyUyMCUyMndvcmxkJTIyJTVEJTBBbm9ybWFsaXplZF93b3JkX2JveGVzJTIwJTNEJTIwJTVCNjM3JTJDJTIwNzczJTJDJTIwNjkzJTJDJTIwNzgyJTVEJTJDJTIwJTVCNjk4JTJDJTIwNzczJTJDJTIwNzMzJTJDJTIwNzgyJTVEJTBBJTBBdG9rZW5fYm94ZXMlMjAlM0QlMjAlNUIlNUQlMEFmb3IlMjB3b3JkJTJDJTIwYm94JTIwaW4lMjB6aXAod29yZHMlMkMlMjBub3JtYWxpemVkX3dvcmRfYm94ZXMpJTNBJTBBJTIwJTIwJTIwJTIwd29yZF90b2tlbnMlMjAlM0QlMjB0b2tlbml6ZXIudG9rZW5pemUod29yZCklMEElMjAlMjAlMjAlMjB0b2tlbl9ib3hlcy5leHRlbmQoJTVCYm94JTVEJTIwKiUyMGxlbih3b3JkX3Rva2VucykpJTBBJTIzJTIwYWRkJTIwYm91bmRpbmclMjBib3hlcyUyMG9mJTIwY2xzJTIwJTJCJTIwc2VwJTIwdG9rZW5zJTBBdG9rZW5fYm94ZXMlMjAlM0QlMjAlNUIlNUIwJTJDJTIwMCUyQyUyMDAlMkMlMjAwJTVEJTVEJTIwJTJCJTIwdG9rZW5fYm94ZXMlMjAlMkIlMjAlNUIlNUIxMDAwJTJDJTIwMTAwMCUyQyUyMDEwMDAlMkMlMjAxMDAwJTVEJTVEJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTIyJTIwJTIyLmpvaW4od29yZHMpJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFpbnB1dF9pZHMlMjAlM0QlMjBlbmNvZGluZyU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQWF0dGVudGlvbl9tYXNrJTIwJTNEJTIwZW5jb2RpbmclNUIlMjJhdHRlbnRpb25fbWFzayUyMiU1RCUwQXRva2VuX3R5cGVfaWRzJTIwJTNEJTIwZW5jb2RpbmclNUIlMjJ0b2tlbl90eXBlX2lkcyUyMiU1RCUwQWJib3glMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCdG9rZW5fYm94ZXMlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMGlucHV0X2lkcyUzRGlucHV0X2lkcyUyQyUyMGJib3glM0RiYm94JTJDJTIwYXR0ZW50aW9uX21hc2slM0RhdHRlbnRpb25fbWFzayUyQyUyMHRva2VuX3R5cGVfaWRzJTNEdG9rZW5fdHlwZV9pZHMlMEEpJTBBJTBBbGFzdF9oaWRkZW5fc3RhdGVzJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LayoutLMModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMModel.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = torch.tensor([token_boxes])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),h(u.$$.fragment)},l(l){t=p(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-kvfsh7"&&(t.textContent=m),n=r(l),f(u.$$.fragment,l)},m(l,w){d(l,t,w),d(l,n,w),g(u,l,w),T=!0},p:B,i(l){T||(_(u.$$.fragment,l),T=!0)},o(l){y(u.$$.fragment,l),T=!1},d(l){l&&(s(t),s(n)),M(u,l)}}}function ds(k){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,u){d(n,t,u)},p:B,d(n){n&&s(t)}}}function cs(k){let t,m="Examples:",n,u,T;return u=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMYXlvdXRMTUZvck1hc2tlZExNJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZsYXlvdXRsbS1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBMYXlvdXRMTUZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZsYXlvdXRsbS1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBd29yZHMlMjAlM0QlMjAlNUIlMjJIZWxsbyUyMiUyQyUyMCUyMiU1Qk1BU0slNUQlMjIlNUQlMEFub3JtYWxpemVkX3dvcmRfYm94ZXMlMjAlM0QlMjAlNUI2MzclMkMlMjA3NzMlMkMlMjA2OTMlMkMlMjA3ODIlNUQlMkMlMjAlNUI2OTglMkMlMjA3NzMlMkMlMjA3MzMlMkMlMjA3ODIlNUQlMEElMEF0b2tlbl9ib3hlcyUyMCUzRCUyMCU1QiU1RCUwQWZvciUyMHdvcmQlMkMlMjBib3glMjBpbiUyMHppcCh3b3JkcyUyQyUyMG5vcm1hbGl6ZWRfd29yZF9ib3hlcyklM0ElMEElMjAlMjAlMjAlMjB3b3JkX3Rva2VucyUyMCUzRCUyMHRva2VuaXplci50b2tlbml6ZSh3b3JkKSUwQSUyMCUyMCUyMCUyMHRva2VuX2JveGVzLmV4dGVuZCglNUJib3glNUQlMjAqJTIwbGVuKHdvcmRfdG9rZW5zKSklMEElMjMlMjBhZGQlMjBib3VuZGluZyUyMGJveGVzJTIwb2YlMjBjbHMlMjAlMkIlMjBzZXAlMjB0b2tlbnMlMEF0b2tlbl9ib3hlcyUyMCUzRCUyMCU1QiU1QjAlMkMlMjAwJTJDJTIwMCUyQyUyMDAlNUQlNUQlMjAlMkIlMjB0b2tlbl9ib3hlcyUyMCUyQiUyMCU1QiU1QjEwMDAlMkMlMjAxMDAwJTJDJTIwMTAwMCUyQyUyMDEwMDAlNUQlNUQlMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglMjIlMjAlMjIuam9pbih3b3JkcyklMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWlucHV0X2lkcyUyMCUzRCUyMGVuY29kaW5nJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBYXR0ZW50aW9uX21hc2slMjAlM0QlMjBlbmNvZGluZyU1QiUyMmF0dGVudGlvbl9tYXNrJTIyJTVEJTBBdG9rZW5fdHlwZV9pZHMlMjAlM0QlMjBlbmNvZGluZyU1QiUyMnRva2VuX3R5cGVfaWRzJTIyJTVEJTBBYmJveCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUJ0b2tlbl9ib3hlcyU1RCklMEElMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMjB3b3JsZCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMGlucHV0X2lkcyUzRGlucHV0X2lkcyUyQyUwQSUyMCUyMCUyMCUyMGJib3glM0RiYm94JTJDJTBBJTIwJTIwJTIwJTIwYXR0ZW50aW9uX21hc2slM0RhdHRlbnRpb25fbWFzayUyQyUwQSUyMCUyMCUyMCUyMHRva2VuX3R5cGVfaWRzJTNEdG9rZW5fdHlwZV9pZHMlMkMlMEElMjAlMjAlMjAlMjBsYWJlbHMlM0RsYWJlbHMlMkMlMEEpJTBBJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LayoutLMForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMForMaskedLM.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;[MASK]&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = torch.tensor([token_boxes])

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=labels,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),h(u.$$.fragment)},l(l){t=p(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-kvfsh7"&&(t.textContent=m),n=r(l),f(u.$$.fragment,l)},m(l,w){d(l,t,w),d(l,n,w),g(u,l,w),T=!0},p:B,i(l){T||(_(u.$$.fragment,l),T=!0)},o(l){y(u.$$.fragment,l),T=!1},d(l){l&&(s(t),s(n)),M(u,l)}}}function ps(k){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,u){d(n,t,u)},p:B,d(n){n&&s(t)}}}function us(k){let t,m="Examples:",n,u,T;return u=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMYXlvdXRMTUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRmxheW91dGxtLWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMExheW91dExNRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbGF5b3V0bG0tYmFzZS11bmNhc2VkJTIyKSUwQSUwQXdvcmRzJTIwJTNEJTIwJTVCJTIySGVsbG8lMjIlMkMlMjAlMjJ3b3JsZCUyMiU1RCUwQW5vcm1hbGl6ZWRfd29yZF9ib3hlcyUyMCUzRCUyMCU1QjYzNyUyQyUyMDc3MyUyQyUyMDY5MyUyQyUyMDc4MiU1RCUyQyUyMCU1QjY5OCUyQyUyMDc3MyUyQyUyMDczMyUyQyUyMDc4MiU1RCUwQSUwQXRva2VuX2JveGVzJTIwJTNEJTIwJTVCJTVEJTBBZm9yJTIwd29yZCUyQyUyMGJveCUyMGluJTIwemlwKHdvcmRzJTJDJTIwbm9ybWFsaXplZF93b3JkX2JveGVzKSUzQSUwQSUyMCUyMCUyMCUyMHdvcmRfdG9rZW5zJTIwJTNEJTIwdG9rZW5pemVyLnRva2VuaXplKHdvcmQpJTBBJTIwJTIwJTIwJTIwdG9rZW5fYm94ZXMuZXh0ZW5kKCU1QmJveCU1RCUyMColMjBsZW4od29yZF90b2tlbnMpKSUwQSUyMyUyMGFkZCUyMGJvdW5kaW5nJTIwYm94ZXMlMjBvZiUyMGNscyUyMCUyQiUyMHNlcCUyMHRva2VucyUwQXRva2VuX2JveGVzJTIwJTNEJTIwJTVCJTVCMCUyQyUyMDAlMkMlMjAwJTJDJTIwMCU1RCU1RCUyMCUyQiUyMHRva2VuX2JveGVzJTIwJTJCJTIwJTVCJTVCMTAwMCUyQyUyMDEwMDAlMkMlMjAxMDAwJTJDJTIwMTAwMCU1RCU1RCUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCUyMiUyMCUyMi5qb2luKHdvcmRzKSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBaW5wdXRfaWRzJTIwJTNEJTIwZW5jb2RpbmclNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEFhdHRlbnRpb25fbWFzayUyMCUzRCUyMGVuY29kaW5nJTVCJTIyYXR0ZW50aW9uX21hc2slMjIlNUQlMEF0b2tlbl90eXBlX2lkcyUyMCUzRCUyMGVuY29kaW5nJTVCJTIydG9rZW5fdHlwZV9pZHMlMjIlNUQlMEFiYm94JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QnRva2VuX2JveGVzJTVEKSUwQXNlcXVlbmNlX2xhYmVsJTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjElNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMGlucHV0X2lkcyUzRGlucHV0X2lkcyUyQyUwQSUyMCUyMCUyMCUyMGJib3glM0RiYm94JTJDJTBBJTIwJTIwJTIwJTIwYXR0ZW50aW9uX21hc2slM0RhdHRlbnRpb25fbWFzayUyQyUwQSUyMCUyMCUyMCUyMHRva2VuX3R5cGVfaWRzJTNEdG9rZW5fdHlwZV9pZHMlMkMlMEElMjAlMjAlMjAlMjBsYWJlbHMlM0RzZXF1ZW5jZV9sYWJlbCUyQyUwQSklMEElMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LayoutLMForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = torch.tensor([token_boxes])
<span class="hljs-meta">&gt;&gt;&gt; </span>sequence_label = torch.tensor([<span class="hljs-number">1</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=sequence_label,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),h(u.$$.fragment)},l(l){t=p(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-kvfsh7"&&(t.textContent=m),n=r(l),f(u.$$.fragment,l)},m(l,w){d(l,t,w),d(l,n,w),g(u,l,w),T=!0},p:B,i(l){T||(_(u.$$.fragment,l),T=!0)},o(l){y(u.$$.fragment,l),T=!1},d(l){l&&(s(t),s(n)),M(u,l)}}}function ms(k){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,u){d(n,t,u)},p:B,d(n){n&&s(t)}}}function hs(k){let t,m="Examples:",n,u,T;return u=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMYXlvdXRMTUZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRmxheW91dGxtLWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMExheW91dExNRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbGF5b3V0bG0tYmFzZS11bmNhc2VkJTIyKSUwQSUwQXdvcmRzJTIwJTNEJTIwJTVCJTIySGVsbG8lMjIlMkMlMjAlMjJ3b3JsZCUyMiU1RCUwQW5vcm1hbGl6ZWRfd29yZF9ib3hlcyUyMCUzRCUyMCU1QjYzNyUyQyUyMDc3MyUyQyUyMDY5MyUyQyUyMDc4MiU1RCUyQyUyMCU1QjY5OCUyQyUyMDc3MyUyQyUyMDczMyUyQyUyMDc4MiU1RCUwQSUwQXRva2VuX2JveGVzJTIwJTNEJTIwJTVCJTVEJTBBZm9yJTIwd29yZCUyQyUyMGJveCUyMGluJTIwemlwKHdvcmRzJTJDJTIwbm9ybWFsaXplZF93b3JkX2JveGVzKSUzQSUwQSUyMCUyMCUyMCUyMHdvcmRfdG9rZW5zJTIwJTNEJTIwdG9rZW5pemVyLnRva2VuaXplKHdvcmQpJTBBJTIwJTIwJTIwJTIwdG9rZW5fYm94ZXMuZXh0ZW5kKCU1QmJveCU1RCUyMColMjBsZW4od29yZF90b2tlbnMpKSUwQSUyMyUyMGFkZCUyMGJvdW5kaW5nJTIwYm94ZXMlMjBvZiUyMGNscyUyMCUyQiUyMHNlcCUyMHRva2VucyUwQXRva2VuX2JveGVzJTIwJTNEJTIwJTVCJTVCMCUyQyUyMDAlMkMlMjAwJTJDJTIwMCU1RCU1RCUyMCUyQiUyMHRva2VuX2JveGVzJTIwJTJCJTIwJTVCJTVCMTAwMCUyQyUyMDEwMDAlMkMlMjAxMDAwJTJDJTIwMTAwMCU1RCU1RCUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCUyMiUyMCUyMi5qb2luKHdvcmRzKSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBaW5wdXRfaWRzJTIwJTNEJTIwZW5jb2RpbmclNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEFhdHRlbnRpb25fbWFzayUyMCUzRCUyMGVuY29kaW5nJTVCJTIyYXR0ZW50aW9uX21hc2slMjIlNUQlMEF0b2tlbl90eXBlX2lkcyUyMCUzRCUyMGVuY29kaW5nJTVCJTIydG9rZW5fdHlwZV9pZHMlMjIlNUQlMEFiYm94JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QnRva2VuX2JveGVzJTVEKSUwQXRva2VuX2xhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTJDJTIwMSUyQyUyMDAlMkMlMjAwJTVEKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBvZiUyMDElMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoJTBBJTIwJTIwJTIwJTIwaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTBBJTIwJTIwJTIwJTIwYmJveCUzRGJib3glMkMlMEElMjAlMjAlMjAlMjBhdHRlbnRpb25fbWFzayUzRGF0dGVudGlvbl9tYXNrJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5fdHlwZV9pZHMlM0R0b2tlbl90eXBlX2lkcyUyQyUwQSUyMCUyMCUyMCUyMGxhYmVscyUzRHRva2VuX2xhYmVscyUyQyUwQSklMEElMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LayoutLMForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMForTokenClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/layoutlm-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>words = [<span class="hljs-string">&quot;Hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>normalized_word_boxes = [<span class="hljs-number">637</span>, <span class="hljs-number">773</span>, <span class="hljs-number">693</span>, <span class="hljs-number">782</span>], [<span class="hljs-number">698</span>, <span class="hljs-number">773</span>, <span class="hljs-number">733</span>, <span class="hljs-number">782</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> word, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(words, normalized_word_boxes):
<span class="hljs-meta">... </span>    word_tokens = tokenizer.tokenize(word)
<span class="hljs-meta">... </span>    token_boxes.extend([box] * <span class="hljs-built_in">len</span>(word_tokens))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add bounding boxes of cls + sep tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>token_boxes = [[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]] + token_boxes + [[<span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>, <span class="hljs-number">1000</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(<span class="hljs-string">&quot; &quot;</span>.join(words), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = encoding[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>attention_mask = encoding[<span class="hljs-string">&quot;attention_mask&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>token_type_ids = encoding[<span class="hljs-string">&quot;token_type_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = torch.tensor([token_boxes])
<span class="hljs-meta">&gt;&gt;&gt; </span>token_labels = torch.tensor([<span class="hljs-number">1</span>, <span class="hljs-number">1</span>, <span class="hljs-number">0</span>, <span class="hljs-number">0</span>]).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># batch size of 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    input_ids=input_ids,
<span class="hljs-meta">... </span>    bbox=bbox,
<span class="hljs-meta">... </span>    attention_mask=attention_mask,
<span class="hljs-meta">... </span>    token_type_ids=token_type_ids,
<span class="hljs-meta">... </span>    labels=token_labels,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),h(u.$$.fragment)},l(l){t=p(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-kvfsh7"&&(t.textContent=m),n=r(l),f(u.$$.fragment,l)},m(l,w){d(l,t,w),d(l,n,w),g(u,l,w),T=!0},p:B,i(l){T||(_(u.$$.fragment,l),T=!0)},o(l){y(u.$$.fragment,l),T=!1},d(l){l&&(s(t),s(n)),M(u,l)}}}function fs(k){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,u){d(n,t,u)},p:B,d(n){n&&s(t)}}}function gs(k){let t,m;return t=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMYXlvdXRMTUZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJpbXBpcmElMkZsYXlvdXRsbS1kb2N1bWVudC1xYSUyMiUyQyUyMGFkZF9wcmVmaXhfc3BhY2UlM0RUcnVlKSUwQW1vZGVsJTIwJTNEJTIwTGF5b3V0TE1Gb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyaW1waXJhJTJGbGF5b3V0bG0tZG9jdW1lbnQtcWElMjIlMkMlMjByZXZpc2lvbiUzRCUyMjFlM2ViYWMlMjIpJTBBJTBBZGF0YXNldCUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJuaWVsc3IlMkZmdW5zZCUyMiUyQyUyMHNwbGl0JTNEJTIydHJhaW4lMjIpJTBBZXhhbXBsZSUyMCUzRCUyMGRhdGFzZXQlNUIwJTVEJTBBcXVlc3Rpb24lMjAlM0QlMjAlMjJ3aGF0J3MlMjBoaXMlMjBuYW1lJTNGJTIyJTBBd29yZHMlMjAlM0QlMjBleGFtcGxlJTVCJTIyd29yZHMlMjIlNUQlMEFib3hlcyUyMCUzRCUyMGV4YW1wbGUlNUIlMjJiYm94ZXMlMjIlNUQlMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjBxdWVzdGlvbi5zcGxpdCgpJTJDJTIwd29yZHMlMkMlMjBpc19zcGxpdF9pbnRvX3dvcmRzJTNEVHJ1ZSUyQyUyMHJldHVybl90b2tlbl90eXBlX2lkcyUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQWJib3glMjAlM0QlMjAlNUIlNUQlMEFmb3IlMjBpJTJDJTIwcyUyQyUyMHclMjBpbiUyMHppcChlbmNvZGluZy5pbnB1dF9pZHMlNUIwJTVEJTJDJTIwZW5jb2Rpbmcuc2VxdWVuY2VfaWRzKDApJTJDJTIwZW5jb2Rpbmcud29yZF9pZHMoMCkpJTNBJTBBJTIwJTIwJTIwJTIwaWYlMjBzJTIwJTNEJTNEJTIwMSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGJib3guYXBwZW5kKGJveGVzJTVCdyU1RCklMEElMjAlMjAlMjAlMjBlbGlmJTIwaSUyMCUzRCUzRCUyMHRva2VuaXplci5zZXBfdG9rZW5faWQlM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBiYm94LmFwcGVuZCglNUIxMDAwJTVEJTIwKiUyMDQpJTBBJTIwJTIwJTIwJTIwZWxzZSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGJib3guYXBwZW5kKCU1QjAlNUQlMjAqJTIwNCklMEFlbmNvZGluZyU1QiUyMmJib3glMjIlNUQlMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCYmJveCU1RCklMEElMEF3b3JkX2lkcyUyMCUzRCUyMGVuY29kaW5nLndvcmRfaWRzKDApJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmcpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXN0YXJ0X3Njb3JlcyUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzJTBBZW5kX3Njb3JlcyUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cyUwQXN0YXJ0JTJDJTIwZW5kJTIwJTNEJTIwd29yZF9pZHMlNUJzdGFydF9zY29yZXMuYXJnbWF4KC0xKSU1RCUyQyUyMHdvcmRfaWRzJTVCZW5kX3Njb3Jlcy5hcmdtYXgoLTEpJTVEJTBBcHJpbnQoJTIyJTIwJTIyLmpvaW4od29yZHMlNUJzdGFydCUyMCUzQSUyMGVuZCUyMCUyQiUyMDElNUQpKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LayoutLMForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;impira/layoutlm-document-qa&quot;</span>, add_prefix_space=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LayoutLMForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;impira/layoutlm-document-qa&quot;</span>, revision=<span class="hljs-string">&quot;1e3ebac&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>question = <span class="hljs-string">&quot;what&#x27;s his name?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;words&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(
<span class="hljs-meta">... </span>    question.split(), words, is_split_into_words=<span class="hljs-literal">True</span>, return_token_type_ids=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>bbox = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> i, s, w <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(encoding.input_ids[<span class="hljs-number">0</span>], encoding.sequence_ids(<span class="hljs-number">0</span>), encoding.word_ids(<span class="hljs-number">0</span>)):
<span class="hljs-meta">... </span>    <span class="hljs-keyword">if</span> s == <span class="hljs-number">1</span>:
<span class="hljs-meta">... </span>        bbox.append(boxes[w])
<span class="hljs-meta">... </span>    <span class="hljs-keyword">elif</span> i == tokenizer.sep_token_id:
<span class="hljs-meta">... </span>        bbox.append([<span class="hljs-number">1000</span>] * <span class="hljs-number">4</span>)
<span class="hljs-meta">... </span>    <span class="hljs-keyword">else</span>:
<span class="hljs-meta">... </span>        bbox.append([<span class="hljs-number">0</span>] * <span class="hljs-number">4</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding[<span class="hljs-string">&quot;bbox&quot;</span>] = torch.tensor([bbox])

<span class="hljs-meta">&gt;&gt;&gt; </span>word_ids = encoding.word_ids(<span class="hljs-number">0</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>start_scores = outputs.start_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>end_scores = outputs.end_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>start, end = word_ids[start_scores.argmax(-<span class="hljs-number">1</span>)], word_ids[end_scores.argmax(-<span class="hljs-number">1</span>)]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot; &quot;</span>.join(words[start : end + <span class="hljs-number">1</span>]))
M. Hamann P. Harper, P. Martinez`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,u){g(t,n,u),m=!0},p:B,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){y(t.$$.fragment,n),m=!1},d(n){M(t,n)}}}function _s(k){let t,m,n,u,T,l="<em>This model was released on 2019-12-31 and added to Hugging Face Transformers on 2020-11-16.</em>",w,oe,cn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',qt,ye,Wt,Me,pn='<a href="https://huggingface.co/papers/1912.13318" rel="nofollow">LayoutLM</a> jointly learns text and the document layout rather than focusing only on text. It incorporates positional layout information and visual features of words from the document images.',Zt,be,un='You can find all the original LayoutLM checkpoints under the <a href="https://huggingface.co/collections/microsoft/layoutlm-6564539601de72cb631d0902" rel="nofollow">LayoutLM</a> collection.',Nt,ne,Rt,Te,mn='The example below demonstrates question answering with the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',Bt,se,Qt,ke,Vt,we,hn='<li><p>The original LayoutLM was not designed with a unified processing workflow. Instead, it expects preprocessed text (<code>words</code>) and bounding boxes (<code>boxes</code>) from an external OCR engine (like <a href="https://pypi.org/project/pytesseract/" rel="nofollow">Pytesseract</a>) and provide them as additional inputs to the tokenizer.</p></li> <li><p>The <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel.forward">forward()</a> method expects the input <code>bbox</code> (bounding boxes of the input tokens). Each bounding box should be in the format <code>(x0, y0, x1, y1)</code>.  <code>(x0, y0)</code> corresponds to the upper left corner of the bounding box and <code>{x1, y1)</code> corresponds to the lower right corner. The bounding boxes need to be normalized on a 0-1000 scale as shown below.</p></li>',Et,ve,Gt,Le,fn="<li><code>width</code> and <code>height</code> correspond to the width and height of the original document in which the token occurs. These values can be obtained as shown below.</li>",At,Ue,Ht,je,Xt,Je,gn="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLM. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",St,xe,_n='<li>Read <a href="https://www.philschmid.de/fine-tuning-layoutlm-keras" rel="nofollow">fine-tuning LayoutLM for document-understanding using Keras &amp; Hugging Face Transformers</a> to learn more.</li> <li>Read <a href="https://www.philschmid.de/fine-tuning-layoutlm" rel="nofollow">fine-tune LayoutLM for document-understanding using only Hugging Face Transformers</a> for more information.</li> <li>Refer to this <a href="https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb" rel="nofollow">notebook</a> for a practical example of how to fine-tune LayoutLM.</li> <li>Refer to this <a href="https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb" rel="nofollow">notebook</a> for an example of how to fine-tune LayoutLM for sequence classification.</li> <li>Refer to this <a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb" rel="nofollow">notebook</a> for an example of how to fine-tune LayoutLM for token classification.</li> <li>Read <a href="https://www.philschmid.de/inference-endpoints-layoutlm" rel="nofollow">Deploy LayoutLM with Hugging Face Inference Endpoints</a> to learn how to deploy LayoutLM.</li>',Dt,Ce,Yt,I,ze,fo,ot,yn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel">LayoutLMModel</a>. It is used to instantiate a
LayoutLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the LayoutLM
<a href="https://huggingface.co/microsoft/layoutlm-base-uncased" rel="nofollow">microsoft/layoutlm-base-uncased</a> architecture.`,go,nt,Mn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig">BertConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig">BertConfig</a> for more information.`,_o,ae,Pt,$e,Ot,F,Ie,yo,st,bn="Construct a LayoutLM tokenizer. Based on WordPiece.",Mo,at,Tn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,bo,re,Fe,To,rt,kn=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`,Kt,qe,eo,q,We,ko,it,wn="Construct a “fast” LayoutLM tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on WordPiece.",wo,lt,vn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,vo,ie,Ze,Lo,dt,Ln=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`,to,Ne,oo,j,Re,Uo,ct,Un="The bare Layoutlm Model outputting raw hidden-states without any specific head on top.",jo,pt,jn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Jo,ut,Jn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xo,Q,Be,Co,mt,xn='The <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel">LayoutLMModel</a> forward method, overrides the <code>__call__</code> special method.',zo,le,$o,de,no,Qe,so,J,Ve,Io,ht,Cn="The Layoutlm Model with a <code>language modeling</code> head on top.”",Fo,ft,zn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,qo,gt,$n=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wo,V,Ee,Zo,_t,In='The <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForMaskedLM">LayoutLMForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',No,ce,Ro,pe,ao,Ge,ro,x,Ae,Bo,yt,Fn=`LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
document image classification tasks such as the <a href="https://www.cs.cmu.edu/~aharley/rvl-cdip/" rel="nofollow">RVL-CDIP</a> dataset.`,Qo,Mt,qn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Vo,bt,Wn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Eo,E,He,Go,Tt,Zn='The <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForSequenceClassification">LayoutLMForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Ao,ue,Ho,me,io,Xe,lo,C,Se,Xo,kt,Nn=`LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
sequence labeling (information extraction) tasks such as the <a href="https://guillaumejaume.github.io/FUNSD/" rel="nofollow">FUNSD</a>
dataset and the <a href="https://rrc.cvc.uab.es/?ch=13" rel="nofollow">SROIE</a> dataset.`,So,wt,Rn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Do,vt,Bn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Yo,G,De,Po,Lt,Qn='The <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification">LayoutLMForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Oo,he,Ko,fe,co,Ye,po,z,Pe,en,Ut,Vn=`The Layoutlm transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,tn,jt,En=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,on,Jt,Gn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,nn,U,Oe,sn,xt,An='The <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForQuestionAnswering">LayoutLMForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',an,ge,rn,Ct,Hn="Example:",ln,zt,Xn=`In the example below, we prepare a question + context pair for the LayoutLM model. It will give us a prediction
of what it thinks the answer is (the span of the answer within the texts parsed from the image).`,dn,_e,uo,Ke,mo,Ft,ho;return ye=new K({props:{title:"LayoutLM",local:"layoutlm",headingTag:"h1"}}),ne=new $t({props:{warning:!1,$$slots:{default:[ns]},$$scope:{ctx:k}}}),se=new ts({props:{id:"usage",options:["AutoModel"],$$slots:{default:[as]},$$scope:{ctx:k}}}),ke=new K({props:{title:"Notes",local:"notes",headingTag:"h2"}}),ve=new te({props:{code:"ZGVmJTIwbm9ybWFsaXplX2Jib3goYmJveCUyQyUyMHdpZHRoJTJDJTIwaGVpZ2h0KSUzQSUwQSUyMCUyMCUyMCUyMHJldHVybiUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGludCgxMDAwJTIwKiUyMChiYm94JTVCMCU1RCUyMCUyRiUyMHdpZHRoKSklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBpbnQoMTAwMCUyMColMjAoYmJveCU1QjElNUQlMjAlMkYlMjBoZWlnaHQpKSUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGludCgxMDAwJTIwKiUyMChiYm94JTVCMiU1RCUyMCUyRiUyMHdpZHRoKSklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBpbnQoMTAwMCUyMColMjAoYmJveCU1QjMlNUQlMjAlMkYlMjBoZWlnaHQpKSUyQyUwQSUyMCUyMCUyMCUyMCU1RA==",highlighted:`<span class="hljs-keyword">def</span> <span class="hljs-title function_">normalize_bbox</span>(<span class="hljs-params">bbox, width, height</span>):
    <span class="hljs-keyword">return</span> [
        <span class="hljs-built_in">int</span>(<span class="hljs-number">1000</span> * (bbox[<span class="hljs-number">0</span>] / width)),
        <span class="hljs-built_in">int</span>(<span class="hljs-number">1000</span> * (bbox[<span class="hljs-number">1</span>] / height)),
        <span class="hljs-built_in">int</span>(<span class="hljs-number">1000</span> * (bbox[<span class="hljs-number">2</span>] / width)),
        <span class="hljs-built_in">int</span>(<span class="hljs-number">1000</span> * (bbox[<span class="hljs-number">3</span>] / height)),
    ]`,wrap:!1}}),Ue=new te({props:{code:"ZnJvbSUyMFBJTCUyMGltcG9ydCUyMEltYWdlJTBBJTBBJTIzJTIwRG9jdW1lbnQlMjBjYW4lMjBiZSUyMGElMjBwbmclMkMlMjBqcGclMkMlMjBldGMuJTIwUERGcyUyMG11c3QlMjBiZSUyMGNvbnZlcnRlZCUyMHRvJTIwaW1hZ2VzLiUwQWltYWdlJTIwJTNEJTIwSW1hZ2Uub3BlbihuYW1lX29mX3lvdXJfZG9jdW1lbnQpLmNvbnZlcnQoJTIyUkdCJTIyKSUwQSUwQXdpZHRoJTJDJTIwaGVpZ2h0JTIwJTNEJTIwaW1hZ2Uuc2l6ZQ==",highlighted:`<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-comment"># Document can be a png, jpg, etc. PDFs must be converted to images.</span>
image = Image.<span class="hljs-built_in">open</span>(name_of_your_document).convert(<span class="hljs-string">&quot;RGB&quot;</span>)

width, height = image.size`,wrap:!1}}),je=new K({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Ce=new K({props:{title:"LayoutLMConfig",local:"transformers.LayoutLMConfig",headingTag:"h2"}}),ze=new $({props:{name:"class transformers.LayoutLMConfig",anchor:"transformers.LayoutLMConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"max_2d_position_embeddings",val:" = 1024"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the LayoutLM model. Defines the different tokens that can be represented by the
<em>inputs_ids</em> passed to the forward method of <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel">LayoutLMModel</a>.`,name:"vocab_size"},{anchor:"transformers.LayoutLMConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.LayoutLMConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.LayoutLMConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.LayoutLMConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.LayoutLMConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.LayoutLMConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.LayoutLMConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.LayoutLMConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.LayoutLMConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed into <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel">LayoutLMModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.LayoutLMConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.LayoutLMConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.LayoutLMConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The value used to pad input_ids.`,name:"pad_token_id"},{anchor:"transformers.LayoutLMConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.LayoutLMConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.LayoutLMConfig.max_2d_position_embeddings",description:`<strong>max_2d_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum value that the 2D position embedding might ever used. Typically set this to something large
just in case (e.g., 1024).`,name:"max_2d_position_embeddings"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/configuration_layoutlm.py#L30"}}),ae=new It({props:{anchor:"transformers.LayoutLMConfig.example",$$slots:{default:[rs]},$$scope:{ctx:k}}}),$e=new K({props:{title:"LayoutLMTokenizer",local:"transformers.LayoutLMTokenizer",headingTag:"h2"}}),Ie=new $({props:{name:"class transformers.LayoutLMTokenizer",anchor:"transformers.LayoutLMTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.LayoutLMTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.LayoutLMTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.LayoutLMTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.LayoutLMTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LayoutLMTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.LayoutLMTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.LayoutLMTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.LayoutLMTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.LayoutLMTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.LayoutLMTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original LayoutLM).`,name:"strip_accents"},{anchor:"transformers.LayoutLMTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/tokenization_layoutlm.py#L54"}}),Fe=new $({props:{name:"__call__",anchor:"transformers.LayoutLMTokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMTokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.LayoutLMTokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.LayoutLMTokenizer.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.LayoutLMTokenizer.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.LayoutLMTokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.LayoutLMTokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LayoutLMTokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.LayoutLMTokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LayoutLMTokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LayoutLMTokenizer.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.LayoutLMTokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LayoutLMTokenizer.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.LayoutLMTokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.LayoutLMTokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.LayoutLMTokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.LayoutLMTokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.LayoutLMTokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.LayoutLMTokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.LayoutLMTokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.LayoutLMTokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.LayoutLMTokenizer.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>token_type_ids</strong> — List of token type ids to be fed to a model (when <code>return_token_type_ids=True</code> or
if <em>“token_type_ids”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a></p>
</li>
<li>
<p><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>overflowing_tokens</strong> — List of overflowing tokens sequences (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>num_truncated_tokens</strong> — Number of tokens truncated (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>special_tokens_mask</strong> — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when <code>add_special_tokens=True</code> and <code>return_special_tokens_mask=True</code>).</p>
</li>
<li>
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>)</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),qe=new K({props:{title:"LayoutLMTokenizerFast",local:"transformers.LayoutLMTokenizerFast",headingTag:"h2"}}),We=new $({props:{name:"class transformers.LayoutLMTokenizerFast",anchor:"transformers.LayoutLMTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.LayoutLMTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.LayoutLMTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LayoutLMTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.LayoutLMTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.LayoutLMTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.LayoutLMTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.LayoutLMTokenizerFast.clean_text",description:`<strong>clean_text</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to clean the text before tokenization by removing any control characters and replacing all
whitespaces by the classic one.`,name:"clean_text"},{anchor:"transformers.LayoutLMTokenizerFast.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see <a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">this
issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.LayoutLMTokenizerFast.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original LayoutLM).`,name:"strip_accents"},{anchor:"transformers.LayoutLMTokenizerFast.wordpieces_prefix",description:`<strong>wordpieces_prefix</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;##&quot;</code>) &#x2014;
The prefix for subwords.`,name:"wordpieces_prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/tokenization_layoutlm_fast.py#L33"}}),Ze=new $({props:{name:"__call__",anchor:"transformers.LayoutLMTokenizerFast.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMTokenizerFast.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.LayoutLMTokenizerFast.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>token_type_ids</strong> — List of token type ids to be fed to a model (when <code>return_token_type_ids=True</code> or
if <em>“token_type_ids”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a></p>
</li>
<li>
<p><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>overflowing_tokens</strong> — List of overflowing tokens sequences (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>num_truncated_tokens</strong> — Number of tokens truncated (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>special_tokens_mask</strong> — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when <code>add_special_tokens=True</code> and <code>return_special_tokens_mask=True</code>).</p>
</li>
<li>
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>)</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),Ne=new K({props:{title:"LayoutLMModel",local:"transformers.LayoutLMModel",headingTag:"h2"}}),Re=new $({props:{name:"class transformers.LayoutLMModel",anchor:"transformers.LayoutLMModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel">LayoutLMModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L487"}}),Be=new $({props:{name:"forward",anchor:"transformers.LayoutLMModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMModel.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L513",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),le=new $t({props:{$$slots:{default:[is]},$$scope:{ctx:k}}}),de=new It({props:{anchor:"transformers.LayoutLMModel.forward.example",$$slots:{default:[ls]},$$scope:{ctx:k}}}),Qe=new K({props:{title:"LayoutLMForMaskedLM",local:"transformers.LayoutLMForMaskedLM",headingTag:"h2"}}),Ve=new $({props:{name:"class transformers.LayoutLMForMaskedLM",anchor:"transformers.LayoutLMForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForMaskedLM">LayoutLMForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L634"}}),Ee=new $({props:{name:"forward",anchor:"transformers.LayoutLMForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMForMaskedLM.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.LayoutLMForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L656",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
`}}),ce=new $t({props:{$$slots:{default:[ds]},$$scope:{ctx:k}}}),pe=new It({props:{anchor:"transformers.LayoutLMForMaskedLM.forward.example",$$slots:{default:[cs]},$$scope:{ctx:k}}}),Ge=new K({props:{title:"LayoutLMForSequenceClassification",local:"transformers.LayoutLMForSequenceClassification",headingTag:"h2"}}),Ae=new $({props:{name:"class transformers.LayoutLMForSequenceClassification",anchor:"transformers.LayoutLMForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForSequenceClassification">LayoutLMForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L760"}}),He=new $({props:{name:"forward",anchor:"transformers.LayoutLMForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L774",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
`}}),ue=new $t({props:{$$slots:{default:[ps]},$$scope:{ctx:k}}}),me=new It({props:{anchor:"transformers.LayoutLMForSequenceClassification.forward.example",$$slots:{default:[us]},$$scope:{ctx:k}}}),Xe=new K({props:{title:"LayoutLMForTokenClassification",local:"transformers.LayoutLMForTokenClassification",headingTag:"h2"}}),Se=new $({props:{name:"class transformers.LayoutLMForTokenClassification",anchor:"transformers.LayoutLMForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LayoutLMForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification">LayoutLMForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L896"}}),De=new $({props:{name:"forward",anchor:"transformers.LayoutLMForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMForTokenClassification.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.LayoutLMForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L910",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
`}}),he=new $t({props:{$$slots:{default:[ms]},$$scope:{ctx:k}}}),fe=new It({props:{anchor:"transformers.LayoutLMForTokenClassification.forward.example",$$slots:{default:[hs]},$$scope:{ctx:k}}}),Ye=new K({props:{title:"LayoutLMForQuestionAnswering",local:"transformers.LayoutLMForQuestionAnswering",headingTag:"h2"}}),Pe=new $({props:{name:"class transformers.LayoutLMForQuestionAnswering",anchor:"transformers.LayoutLMForQuestionAnswering",parameters:[{name:"config",val:""},{name:"has_visual_segment_embedding",val:" = True"}],parametersDescription:[{anchor:"transformers.LayoutLMForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMForQuestionAnswering">LayoutLMForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.LayoutLMForQuestionAnswering.has_visual_segment_embedding",description:`<strong>has_visual_segment_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add visual segment embeddings.`,name:"has_visual_segment_embedding"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L1006"}}),Oe=new $({props:{name:"forward",anchor:"transformers.LayoutLMForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LayoutLMForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LayoutLMForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlm/modeling_layoutlm.py#L1024",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMConfig"
>LayoutLMConfig</a>) and inputs.</p>
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
`}}),ge=new $t({props:{$$slots:{default:[fs]},$$scope:{ctx:k}}}),_e=new It({props:{anchor:"transformers.LayoutLMForQuestionAnswering.forward.example",$$slots:{default:[gs]},$$scope:{ctx:k}}}),Ke=new es({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/layoutlm.md"}}),{c(){t=c("meta"),m=a(),n=c("p"),u=a(),T=c("p"),T.innerHTML=l,w=a(),oe=c("div"),oe.innerHTML=cn,qt=a(),h(ye.$$.fragment),Wt=a(),Me=c("p"),Me.innerHTML=pn,Zt=a(),be=c("p"),be.innerHTML=un,Nt=a(),h(ne.$$.fragment),Rt=a(),Te=c("p"),Te.innerHTML=mn,Bt=a(),h(se.$$.fragment),Qt=a(),h(ke.$$.fragment),Vt=a(),we=c("ul"),we.innerHTML=hn,Et=a(),h(ve.$$.fragment),Gt=a(),Le=c("ul"),Le.innerHTML=fn,At=a(),h(Ue.$$.fragment),Ht=a(),h(je.$$.fragment),Xt=a(),Je=c("p"),Je.textContent=gn,St=a(),xe=c("ul"),xe.innerHTML=_n,Dt=a(),h(Ce.$$.fragment),Yt=a(),I=c("div"),h(ze.$$.fragment),fo=a(),ot=c("p"),ot.innerHTML=yn,go=a(),nt=c("p"),nt.innerHTML=Mn,_o=a(),h(ae.$$.fragment),Pt=a(),h($e.$$.fragment),Ot=a(),F=c("div"),h(Ie.$$.fragment),yo=a(),st=c("p"),st.textContent=bn,Mo=a(),at=c("p"),at.innerHTML=Tn,bo=a(),re=c("div"),h(Fe.$$.fragment),To=a(),rt=c("p"),rt.textContent=kn,Kt=a(),h(qe.$$.fragment),eo=a(),q=c("div"),h(We.$$.fragment),ko=a(),it=c("p"),it.innerHTML=wn,wo=a(),lt=c("p"),lt.innerHTML=vn,vo=a(),ie=c("div"),h(Ze.$$.fragment),Lo=a(),dt=c("p"),dt.textContent=Ln,to=a(),h(Ne.$$.fragment),oo=a(),j=c("div"),h(Re.$$.fragment),Uo=a(),ct=c("p"),ct.textContent=Un,jo=a(),pt=c("p"),pt.innerHTML=jn,Jo=a(),ut=c("p"),ut.innerHTML=Jn,xo=a(),Q=c("div"),h(Be.$$.fragment),Co=a(),mt=c("p"),mt.innerHTML=xn,zo=a(),h(le.$$.fragment),$o=a(),h(de.$$.fragment),no=a(),h(Qe.$$.fragment),so=a(),J=c("div"),h(Ve.$$.fragment),Io=a(),ht=c("p"),ht.innerHTML=Cn,Fo=a(),ft=c("p"),ft.innerHTML=zn,qo=a(),gt=c("p"),gt.innerHTML=$n,Wo=a(),V=c("div"),h(Ee.$$.fragment),Zo=a(),_t=c("p"),_t.innerHTML=In,No=a(),h(ce.$$.fragment),Ro=a(),h(pe.$$.fragment),ao=a(),h(Ge.$$.fragment),ro=a(),x=c("div"),h(Ae.$$.fragment),Bo=a(),yt=c("p"),yt.innerHTML=Fn,Qo=a(),Mt=c("p"),Mt.innerHTML=qn,Vo=a(),bt=c("p"),bt.innerHTML=Wn,Eo=a(),E=c("div"),h(He.$$.fragment),Go=a(),Tt=c("p"),Tt.innerHTML=Zn,Ao=a(),h(ue.$$.fragment),Ho=a(),h(me.$$.fragment),io=a(),h(Xe.$$.fragment),lo=a(),C=c("div"),h(Se.$$.fragment),Xo=a(),kt=c("p"),kt.innerHTML=Nn,So=a(),wt=c("p"),wt.innerHTML=Rn,Do=a(),vt=c("p"),vt.innerHTML=Bn,Yo=a(),G=c("div"),h(De.$$.fragment),Po=a(),Lt=c("p"),Lt.innerHTML=Qn,Oo=a(),h(he.$$.fragment),Ko=a(),h(fe.$$.fragment),co=a(),h(Ye.$$.fragment),po=a(),z=c("div"),h(Pe.$$.fragment),en=a(),Ut=c("p"),Ut.innerHTML=Vn,tn=a(),jt=c("p"),jt.innerHTML=En,on=a(),Jt=c("p"),Jt.innerHTML=Gn,nn=a(),U=c("div"),h(Oe.$$.fragment),sn=a(),xt=c("p"),xt.innerHTML=An,an=a(),h(ge.$$.fragment),rn=a(),Ct=c("p"),Ct.textContent=Hn,ln=a(),zt=c("p"),zt.textContent=Xn,dn=a(),h(_e.$$.fragment),uo=a(),h(Ke.$$.fragment),mo=a(),Ft=c("p"),this.h()},l(e){const o=On("svelte-u9bgzb",document.head);t=p(o,"META",{name:!0,content:!0}),o.forEach(s),m=r(e),n=p(e,"P",{}),v(n).forEach(s),u=r(e),T=p(e,"P",{"data-svelte-h":!0}),b(T)!=="svelte-lb5c3c"&&(T.innerHTML=l),w=r(e),oe=p(e,"DIV",{style:!0,"data-svelte-h":!0}),b(oe)!=="svelte-wa5t4p"&&(oe.innerHTML=cn),qt=r(e),f(ye.$$.fragment,e),Wt=r(e),Me=p(e,"P",{"data-svelte-h":!0}),b(Me)!=="svelte-ytp4je"&&(Me.innerHTML=pn),Zt=r(e),be=p(e,"P",{"data-svelte-h":!0}),b(be)!=="svelte-kd0wv2"&&(be.innerHTML=un),Nt=r(e),f(ne.$$.fragment,e),Rt=r(e),Te=p(e,"P",{"data-svelte-h":!0}),b(Te)!=="svelte-u0wxna"&&(Te.innerHTML=mn),Bt=r(e),f(se.$$.fragment,e),Qt=r(e),f(ke.$$.fragment,e),Vt=r(e),we=p(e,"UL",{"data-svelte-h":!0}),b(we)!=="svelte-w2cayx"&&(we.innerHTML=hn),Et=r(e),f(ve.$$.fragment,e),Gt=r(e),Le=p(e,"UL",{"data-svelte-h":!0}),b(Le)!=="svelte-1he2yq9"&&(Le.innerHTML=fn),At=r(e),f(Ue.$$.fragment,e),Ht=r(e),f(je.$$.fragment,e),Xt=r(e),Je=p(e,"P",{"data-svelte-h":!0}),b(Je)!=="svelte-gbqnu4"&&(Je.textContent=gn),St=r(e),xe=p(e,"UL",{"data-svelte-h":!0}),b(xe)!=="svelte-hxs6su"&&(xe.innerHTML=_n),Dt=r(e),f(Ce.$$.fragment,e),Yt=r(e),I=p(e,"DIV",{class:!0});var H=v(I);f(ze.$$.fragment,H),fo=r(H),ot=p(H,"P",{"data-svelte-h":!0}),b(ot)!=="svelte-1wwzu4i"&&(ot.innerHTML=yn),go=r(H),nt=p(H,"P",{"data-svelte-h":!0}),b(nt)!=="svelte-xa1djz"&&(nt.innerHTML=Mn),_o=r(H),f(ae.$$.fragment,H),H.forEach(s),Pt=r(e),f($e.$$.fragment,e),Ot=r(e),F=p(e,"DIV",{class:!0});var X=v(F);f(Ie.$$.fragment,X),yo=r(X),st=p(X,"P",{"data-svelte-h":!0}),b(st)!=="svelte-10n506g"&&(st.textContent=bn),Mo=r(X),at=p(X,"P",{"data-svelte-h":!0}),b(at)!=="svelte-ntrhio"&&(at.innerHTML=Tn),bo=r(X),re=p(X,"DIV",{class:!0});var et=v(re);f(Fe.$$.fragment,et),To=r(et),rt=p(et,"P",{"data-svelte-h":!0}),b(rt)!=="svelte-kpxj0c"&&(rt.textContent=kn),et.forEach(s),X.forEach(s),Kt=r(e),f(qe.$$.fragment,e),eo=r(e),q=p(e,"DIV",{class:!0});var S=v(q);f(We.$$.fragment,S),ko=r(S),it=p(S,"P",{"data-svelte-h":!0}),b(it)!=="svelte-gn4u9l"&&(it.innerHTML=wn),wo=r(S),lt=p(S,"P",{"data-svelte-h":!0}),b(lt)!=="svelte-gxzj9w"&&(lt.innerHTML=vn),vo=r(S),ie=p(S,"DIV",{class:!0});var tt=v(ie);f(Ze.$$.fragment,tt),Lo=r(tt),dt=p(tt,"P",{"data-svelte-h":!0}),b(dt)!=="svelte-kpxj0c"&&(dt.textContent=Ln),tt.forEach(s),S.forEach(s),to=r(e),f(Ne.$$.fragment,e),oo=r(e),j=p(e,"DIV",{class:!0});var W=v(j);f(Re.$$.fragment,W),Uo=r(W),ct=p(W,"P",{"data-svelte-h":!0}),b(ct)!=="svelte-1ohdmax"&&(ct.textContent=Un),jo=r(W),pt=p(W,"P",{"data-svelte-h":!0}),b(pt)!=="svelte-q52n56"&&(pt.innerHTML=jn),Jo=r(W),ut=p(W,"P",{"data-svelte-h":!0}),b(ut)!=="svelte-hswkmf"&&(ut.innerHTML=Jn),xo=r(W),Q=p(W,"DIV",{class:!0});var D=v(Q);f(Be.$$.fragment,D),Co=r(D),mt=p(D,"P",{"data-svelte-h":!0}),b(mt)!=="svelte-2kxzja"&&(mt.innerHTML=xn),zo=r(D),f(le.$$.fragment,D),$o=r(D),f(de.$$.fragment,D),D.forEach(s),W.forEach(s),no=r(e),f(Qe.$$.fragment,e),so=r(e),J=p(e,"DIV",{class:!0});var Z=v(J);f(Ve.$$.fragment,Z),Io=r(Z),ht=p(Z,"P",{"data-svelte-h":!0}),b(ht)!=="svelte-eek1w7"&&(ht.innerHTML=Cn),Fo=r(Z),ft=p(Z,"P",{"data-svelte-h":!0}),b(ft)!=="svelte-q52n56"&&(ft.innerHTML=zn),qo=r(Z),gt=p(Z,"P",{"data-svelte-h":!0}),b(gt)!=="svelte-hswkmf"&&(gt.innerHTML=$n),Wo=r(Z),V=p(Z,"DIV",{class:!0});var Y=v(V);f(Ee.$$.fragment,Y),Zo=r(Y),_t=p(Y,"P",{"data-svelte-h":!0}),b(_t)!=="svelte-ch5xsm"&&(_t.innerHTML=In),No=r(Y),f(ce.$$.fragment,Y),Ro=r(Y),f(pe.$$.fragment,Y),Y.forEach(s),Z.forEach(s),ao=r(e),f(Ge.$$.fragment,e),ro=r(e),x=p(e,"DIV",{class:!0});var N=v(x);f(Ae.$$.fragment,N),Bo=r(N),yt=p(N,"P",{"data-svelte-h":!0}),b(yt)!=="svelte-36uopi"&&(yt.innerHTML=Fn),Qo=r(N),Mt=p(N,"P",{"data-svelte-h":!0}),b(Mt)!=="svelte-q52n56"&&(Mt.innerHTML=qn),Vo=r(N),bt=p(N,"P",{"data-svelte-h":!0}),b(bt)!=="svelte-hswkmf"&&(bt.innerHTML=Wn),Eo=r(N),E=p(N,"DIV",{class:!0});var P=v(E);f(He.$$.fragment,P),Go=r(P),Tt=p(P,"P",{"data-svelte-h":!0}),b(Tt)!=="svelte-1ygr17i"&&(Tt.innerHTML=Zn),Ao=r(P),f(ue.$$.fragment,P),Ho=r(P),f(me.$$.fragment,P),P.forEach(s),N.forEach(s),io=r(e),f(Xe.$$.fragment,e),lo=r(e),C=p(e,"DIV",{class:!0});var R=v(C);f(Se.$$.fragment,R),Xo=r(R),kt=p(R,"P",{"data-svelte-h":!0}),b(kt)!=="svelte-1yummv5"&&(kt.innerHTML=Nn),So=r(R),wt=p(R,"P",{"data-svelte-h":!0}),b(wt)!=="svelte-q52n56"&&(wt.innerHTML=Rn),Do=r(R),vt=p(R,"P",{"data-svelte-h":!0}),b(vt)!=="svelte-hswkmf"&&(vt.innerHTML=Bn),Yo=r(R),G=p(R,"DIV",{class:!0});var O=v(G);f(De.$$.fragment,O),Po=r(O),Lt=p(O,"P",{"data-svelte-h":!0}),b(Lt)!=="svelte-giyvxw"&&(Lt.innerHTML=Qn),Oo=r(O),f(he.$$.fragment,O),Ko=r(O),f(fe.$$.fragment,O),O.forEach(s),R.forEach(s),co=r(e),f(Ye.$$.fragment,e),po=r(e),z=p(e,"DIV",{class:!0});var ee=v(z);f(Pe.$$.fragment,ee),en=r(ee),Ut=p(ee,"P",{"data-svelte-h":!0}),b(Ut)!=="svelte-za2rme"&&(Ut.innerHTML=Vn),tn=r(ee),jt=p(ee,"P",{"data-svelte-h":!0}),b(jt)!=="svelte-q52n56"&&(jt.innerHTML=En),on=r(ee),Jt=p(ee,"P",{"data-svelte-h":!0}),b(Jt)!=="svelte-hswkmf"&&(Jt.innerHTML=Gn),nn=r(ee),U=p(ee,"DIV",{class:!0});var A=v(U);f(Oe.$$.fragment,A),sn=r(A),xt=p(A,"P",{"data-svelte-h":!0}),b(xt)!=="svelte-14ycbfi"&&(xt.innerHTML=An),an=r(A),f(ge.$$.fragment,A),rn=r(A),Ct=p(A,"P",{"data-svelte-h":!0}),b(Ct)!=="svelte-11lpom8"&&(Ct.textContent=Hn),ln=r(A),zt=p(A,"P",{"data-svelte-h":!0}),b(zt)!=="svelte-1jlo8qy"&&(zt.textContent=Xn),dn=r(A),f(_e.$$.fragment,A),A.forEach(s),ee.forEach(s),uo=r(e),f(Ke.$$.fragment,e),mo=r(e),Ft=p(e,"P",{}),v(Ft).forEach(s),this.h()},h(){L(t,"name","hf:doc:metadata"),L(t,"content",ys),Kn(oe,"float","right"),L(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){i(document.head,t),d(e,m,o),d(e,n,o),d(e,u,o),d(e,T,o),d(e,w,o),d(e,oe,o),d(e,qt,o),g(ye,e,o),d(e,Wt,o),d(e,Me,o),d(e,Zt,o),d(e,be,o),d(e,Nt,o),g(ne,e,o),d(e,Rt,o),d(e,Te,o),d(e,Bt,o),g(se,e,o),d(e,Qt,o),g(ke,e,o),d(e,Vt,o),d(e,we,o),d(e,Et,o),g(ve,e,o),d(e,Gt,o),d(e,Le,o),d(e,At,o),g(Ue,e,o),d(e,Ht,o),g(je,e,o),d(e,Xt,o),d(e,Je,o),d(e,St,o),d(e,xe,o),d(e,Dt,o),g(Ce,e,o),d(e,Yt,o),d(e,I,o),g(ze,I,null),i(I,fo),i(I,ot),i(I,go),i(I,nt),i(I,_o),g(ae,I,null),d(e,Pt,o),g($e,e,o),d(e,Ot,o),d(e,F,o),g(Ie,F,null),i(F,yo),i(F,st),i(F,Mo),i(F,at),i(F,bo),i(F,re),g(Fe,re,null),i(re,To),i(re,rt),d(e,Kt,o),g(qe,e,o),d(e,eo,o),d(e,q,o),g(We,q,null),i(q,ko),i(q,it),i(q,wo),i(q,lt),i(q,vo),i(q,ie),g(Ze,ie,null),i(ie,Lo),i(ie,dt),d(e,to,o),g(Ne,e,o),d(e,oo,o),d(e,j,o),g(Re,j,null),i(j,Uo),i(j,ct),i(j,jo),i(j,pt),i(j,Jo),i(j,ut),i(j,xo),i(j,Q),g(Be,Q,null),i(Q,Co),i(Q,mt),i(Q,zo),g(le,Q,null),i(Q,$o),g(de,Q,null),d(e,no,o),g(Qe,e,o),d(e,so,o),d(e,J,o),g(Ve,J,null),i(J,Io),i(J,ht),i(J,Fo),i(J,ft),i(J,qo),i(J,gt),i(J,Wo),i(J,V),g(Ee,V,null),i(V,Zo),i(V,_t),i(V,No),g(ce,V,null),i(V,Ro),g(pe,V,null),d(e,ao,o),g(Ge,e,o),d(e,ro,o),d(e,x,o),g(Ae,x,null),i(x,Bo),i(x,yt),i(x,Qo),i(x,Mt),i(x,Vo),i(x,bt),i(x,Eo),i(x,E),g(He,E,null),i(E,Go),i(E,Tt),i(E,Ao),g(ue,E,null),i(E,Ho),g(me,E,null),d(e,io,o),g(Xe,e,o),d(e,lo,o),d(e,C,o),g(Se,C,null),i(C,Xo),i(C,kt),i(C,So),i(C,wt),i(C,Do),i(C,vt),i(C,Yo),i(C,G),g(De,G,null),i(G,Po),i(G,Lt),i(G,Oo),g(he,G,null),i(G,Ko),g(fe,G,null),d(e,co,o),g(Ye,e,o),d(e,po,o),d(e,z,o),g(Pe,z,null),i(z,en),i(z,Ut),i(z,tn),i(z,jt),i(z,on),i(z,Jt),i(z,nn),i(z,U),g(Oe,U,null),i(U,sn),i(U,xt),i(U,an),g(ge,U,null),i(U,rn),i(U,Ct),i(U,ln),i(U,zt),i(U,dn),g(_e,U,null),d(e,uo,o),g(Ke,e,o),d(e,mo,o),d(e,Ft,o),ho=!0},p(e,[o]){const H={};o&2&&(H.$$scope={dirty:o,ctx:e}),ne.$set(H);const X={};o&2&&(X.$$scope={dirty:o,ctx:e}),se.$set(X);const et={};o&2&&(et.$$scope={dirty:o,ctx:e}),ae.$set(et);const S={};o&2&&(S.$$scope={dirty:o,ctx:e}),le.$set(S);const tt={};o&2&&(tt.$$scope={dirty:o,ctx:e}),de.$set(tt);const W={};o&2&&(W.$$scope={dirty:o,ctx:e}),ce.$set(W);const D={};o&2&&(D.$$scope={dirty:o,ctx:e}),pe.$set(D);const Z={};o&2&&(Z.$$scope={dirty:o,ctx:e}),ue.$set(Z);const Y={};o&2&&(Y.$$scope={dirty:o,ctx:e}),me.$set(Y);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),he.$set(N);const P={};o&2&&(P.$$scope={dirty:o,ctx:e}),fe.$set(P);const R={};o&2&&(R.$$scope={dirty:o,ctx:e}),ge.$set(R);const O={};o&2&&(O.$$scope={dirty:o,ctx:e}),_e.$set(O)},i(e){ho||(_(ye.$$.fragment,e),_(ne.$$.fragment,e),_(se.$$.fragment,e),_(ke.$$.fragment,e),_(ve.$$.fragment,e),_(Ue.$$.fragment,e),_(je.$$.fragment,e),_(Ce.$$.fragment,e),_(ze.$$.fragment,e),_(ae.$$.fragment,e),_($e.$$.fragment,e),_(Ie.$$.fragment,e),_(Fe.$$.fragment,e),_(qe.$$.fragment,e),_(We.$$.fragment,e),_(Ze.$$.fragment,e),_(Ne.$$.fragment,e),_(Re.$$.fragment,e),_(Be.$$.fragment,e),_(le.$$.fragment,e),_(de.$$.fragment,e),_(Qe.$$.fragment,e),_(Ve.$$.fragment,e),_(Ee.$$.fragment,e),_(ce.$$.fragment,e),_(pe.$$.fragment,e),_(Ge.$$.fragment,e),_(Ae.$$.fragment,e),_(He.$$.fragment,e),_(ue.$$.fragment,e),_(me.$$.fragment,e),_(Xe.$$.fragment,e),_(Se.$$.fragment,e),_(De.$$.fragment,e),_(he.$$.fragment,e),_(fe.$$.fragment,e),_(Ye.$$.fragment,e),_(Pe.$$.fragment,e),_(Oe.$$.fragment,e),_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(Ke.$$.fragment,e),ho=!0)},o(e){y(ye.$$.fragment,e),y(ne.$$.fragment,e),y(se.$$.fragment,e),y(ke.$$.fragment,e),y(ve.$$.fragment,e),y(Ue.$$.fragment,e),y(je.$$.fragment,e),y(Ce.$$.fragment,e),y(ze.$$.fragment,e),y(ae.$$.fragment,e),y($e.$$.fragment,e),y(Ie.$$.fragment,e),y(Fe.$$.fragment,e),y(qe.$$.fragment,e),y(We.$$.fragment,e),y(Ze.$$.fragment,e),y(Ne.$$.fragment,e),y(Re.$$.fragment,e),y(Be.$$.fragment,e),y(le.$$.fragment,e),y(de.$$.fragment,e),y(Qe.$$.fragment,e),y(Ve.$$.fragment,e),y(Ee.$$.fragment,e),y(ce.$$.fragment,e),y(pe.$$.fragment,e),y(Ge.$$.fragment,e),y(Ae.$$.fragment,e),y(He.$$.fragment,e),y(ue.$$.fragment,e),y(me.$$.fragment,e),y(Xe.$$.fragment,e),y(Se.$$.fragment,e),y(De.$$.fragment,e),y(he.$$.fragment,e),y(fe.$$.fragment,e),y(Ye.$$.fragment,e),y(Pe.$$.fragment,e),y(Oe.$$.fragment,e),y(ge.$$.fragment,e),y(_e.$$.fragment,e),y(Ke.$$.fragment,e),ho=!1},d(e){e&&(s(m),s(n),s(u),s(T),s(w),s(oe),s(qt),s(Wt),s(Me),s(Zt),s(be),s(Nt),s(Rt),s(Te),s(Bt),s(Qt),s(Vt),s(we),s(Et),s(Gt),s(Le),s(At),s(Ht),s(Xt),s(Je),s(St),s(xe),s(Dt),s(Yt),s(I),s(Pt),s(Ot),s(F),s(Kt),s(eo),s(q),s(to),s(oo),s(j),s(no),s(so),s(J),s(ao),s(ro),s(x),s(io),s(lo),s(C),s(co),s(po),s(z),s(uo),s(mo),s(Ft)),s(t),M(ye,e),M(ne,e),M(se,e),M(ke,e),M(ve,e),M(Ue,e),M(je,e),M(Ce,e),M(ze),M(ae),M($e,e),M(Ie),M(Fe),M(qe,e),M(We),M(Ze),M(Ne,e),M(Re),M(Be),M(le),M(de),M(Qe,e),M(Ve),M(Ee),M(ce),M(pe),M(Ge,e),M(Ae),M(He),M(ue),M(me),M(Xe,e),M(Se),M(De),M(he),M(fe),M(Ye,e),M(Pe),M(Oe),M(ge),M(_e),M(Ke,e)}}}const ys='{"title":"LayoutLM","local":"layoutlm","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"LayoutLMConfig","local":"transformers.LayoutLMConfig","sections":[],"depth":2},{"title":"LayoutLMTokenizer","local":"transformers.LayoutLMTokenizer","sections":[],"depth":2},{"title":"LayoutLMTokenizerFast","local":"transformers.LayoutLMTokenizerFast","sections":[],"depth":2},{"title":"LayoutLMModel","local":"transformers.LayoutLMModel","sections":[],"depth":2},{"title":"LayoutLMForMaskedLM","local":"transformers.LayoutLMForMaskedLM","sections":[],"depth":2},{"title":"LayoutLMForSequenceClassification","local":"transformers.LayoutLMForSequenceClassification","sections":[],"depth":2},{"title":"LayoutLMForTokenClassification","local":"transformers.LayoutLMForTokenClassification","sections":[],"depth":2},{"title":"LayoutLMForQuestionAnswering","local":"transformers.LayoutLMForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Ms(k){return Dn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Js extends Yn{constructor(t){super(),Pn(this,t,Ms,_s,Sn,{})}}export{Js as component};
