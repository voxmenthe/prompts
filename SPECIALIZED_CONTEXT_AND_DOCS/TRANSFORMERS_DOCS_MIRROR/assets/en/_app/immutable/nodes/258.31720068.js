import{s as Go,z as No,o as So,n as H}from"../chunks/scheduler.18a86fab.js";import{S as Eo,i as Ho,g as d,s as r,r as u,A as Qo,h as c,f as o,c as i,j as G,x as h,u as f,k as C,y as m,a,v as g,d as _,t as b,w as T}from"../chunks/index.98837b22.js";import{T as Ut}from"../chunks/Tip.77304350.js";import{D as S}from"../chunks/Docstring.a1ef7999.js";import{C as Ke}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as tt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as E,E as Po}from"../chunks/getInferenceSnippets.06c2775f.js";function Ao(M){let n,w="Examples:",l,p,y;return p=new Ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExpbHRDb25maWclMkMlMjBMaWx0TW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTGlMVCUyMFNDVVQtRExWQ0xhYiUyRmxpbHQtcm9iZXJ0YS1lbi1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMExpbHRDb25maWcoKSUwQSUyMyUyMFJhbmRvbWx5JTIwaW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMFNDVVQtRExWQ0xhYiUyRmxpbHQtcm9iZXJ0YS1lbi1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBMaWx0TW9kZWwoY29uZmlndXJhdGlvbiklMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LiltConfig, LiltModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a LiLT SCUT-DLVCLab/lilt-roberta-en-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = LiltConfig()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Randomly initializing a model from the SCUT-DLVCLab/lilt-roberta-en-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LiltModel(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){n=d("p"),n.textContent=w,l=r(),u(p.$$.fragment)},l(s){n=c(s,"P",{"data-svelte-h":!0}),h(n)!=="svelte-kvfsh7"&&(n.textContent=w),l=i(s),f(p.$$.fragment,s)},m(s,v){a(s,n,v),a(s,l,v),g(p,s,v),y=!0},p:H,i(s){y||(_(p.$$.fragment,s),y=!0)},o(s){b(p.$$.fragment,s),y=!1},d(s){s&&(o(n),o(l)),T(p,s)}}}function Yo(M){let n,w=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=w},l(l){n=c(l,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=w)},m(l,p){a(l,n,p)},p:H,d(l){l&&o(n)}}}function Oo(M){let n,w="Examples:",l,p,y;return p=new Ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBdXRvTW9kZWwlMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJTQ1VULURMVkNMYWIlMkZsaWx0LXJvYmVydGEtZW4tYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyU0NVVC1ETFZDTGFiJTJGbGlsdC1yb2JlcnRhLWVuLWJhc2UlMjIpJTBBJTBBZGF0YXNldCUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJuaWVsc3IlMkZmdW5zZC1sYXlvdXRsbXYzJTIyJTJDJTIwc3BsaXQlM0QlMjJ0cmFpbiUyMiklMEFleGFtcGxlJTIwJTNEJTIwZGF0YXNldCU1QjAlNUQlMEF3b3JkcyUyMCUzRCUyMGV4YW1wbGUlNUIlMjJ0b2tlbnMlMjIlNUQlMEFib3hlcyUyMCUzRCUyMGV4YW1wbGUlNUIlMjJiYm94ZXMlMjIlNUQlMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplcih3b3JkcyUyQyUyMGJveGVzJTNEYm94ZXMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmVuY29kaW5nKSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;SCUT-DLVCLab/lilt-roberta-en-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModel.from_pretrained(<span class="hljs-string">&quot;SCUT-DLVCLab/lilt-roberta-en-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd-layoutlmv3&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;tokens&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(words, boxes=boxes, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){n=d("p"),n.textContent=w,l=r(),u(p.$$.fragment)},l(s){n=c(s,"P",{"data-svelte-h":!0}),h(n)!=="svelte-kvfsh7"&&(n.textContent=w),l=i(s),f(p.$$.fragment,s)},m(s,v){a(s,n,v),a(s,l,v),g(p,s,v),y=!0},p:H,i(s){y||(_(p.$$.fragment,s),y=!0)},o(s){b(p.$$.fragment,s),y=!1},d(s){s&&(o(n),o(l)),T(p,s)}}}function Do(M){let n,w=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=w},l(l){n=c(l,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=w)},m(l,p){a(l,n,p)},p:H,d(l){l&&o(n)}}}function Ko(M){let n,w="Examples:",l,p,y;return p=new Ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBdXRvTW9kZWxGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyU0NVVC1ETFZDTGFiJTJGbGlsdC1yb2JlcnRhLWVuLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJTQ1VULURMVkNMYWIlMkZsaWx0LXJvYmVydGEtZW4tYmFzZSUyMiklMEElMEFkYXRhc2V0JTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMm5pZWxzciUyRmZ1bnNkLWxheW91dGxtdjMlMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQWV4YW1wbGUlMjAlM0QlMjBkYXRhc2V0JTVCMCU1RCUwQXdvcmRzJTIwJTNEJTIwZXhhbXBsZSU1QiUyMnRva2VucyUyMiU1RCUwQWJveGVzJTIwJTNEJTIwZXhhbXBsZSU1QiUyMmJib3hlcyUyMiU1RCUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKHdvcmRzJTJDJTIwYm94ZXMlM0Rib3hlcyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmcpJTBBcHJlZGljdGVkX2NsYXNzX2lkeCUyMCUzRCUyMG91dHB1dHMubG9naXRzLmFyZ21heCgtMSkuaXRlbSgpJTBBcHJlZGljdGVkX2NsYXNzJTIwJTNEJTIwbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkeCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;SCUT-DLVCLab/lilt-roberta-en-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;SCUT-DLVCLab/lilt-roberta-en-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd-layoutlmv3&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;tokens&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(words, boxes=boxes, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_idx = outputs.logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class = model.config.id2label[predicted_class_idx]`,wrap:!1}}),{c(){n=d("p"),n.textContent=w,l=r(),u(p.$$.fragment)},l(s){n=c(s,"P",{"data-svelte-h":!0}),h(n)!=="svelte-kvfsh7"&&(n.textContent=w),l=i(s),f(p.$$.fragment,s)},m(s,v){a(s,n,v),a(s,l,v),g(p,s,v),y=!0},p:H,i(s){y||(_(p.$$.fragment,s),y=!0)},o(s){b(p.$$.fragment,s),y=!1},d(s){s&&(o(n),o(l)),T(p,s)}}}function en(M){let n,w=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=w},l(l){n=c(l,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=w)},m(l,p){a(l,n,p)},p:H,d(l){l&&o(n)}}}function tn(M){let n,w="Examples:",l,p,y;return p=new Ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBdXRvTW9kZWxGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyU0NVVC1ETFZDTGFiJTJGbGlsdC1yb2JlcnRhLWVuLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJTQ1VULURMVkNMYWIlMkZsaWx0LXJvYmVydGEtZW4tYmFzZSUyMiklMEElMEFkYXRhc2V0JTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMm5pZWxzciUyRmZ1bnNkLWxheW91dGxtdjMlMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQWV4YW1wbGUlMjAlM0QlMjBkYXRhc2V0JTVCMCU1RCUwQXdvcmRzJTIwJTNEJTIwZXhhbXBsZSU1QiUyMnRva2VucyUyMiU1RCUwQWJveGVzJTIwJTNEJTIwZXhhbXBsZSU1QiUyMmJib3hlcyUyMiU1RCUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKHdvcmRzJTJDJTIwYm94ZXMlM0Rib3hlcyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmcpJTBBcHJlZGljdGVkX2NsYXNzX2luZGljZXMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cy5hcmdtYXgoLTEp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;SCUT-DLVCLab/lilt-roberta-en-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForTokenClassification.from_pretrained(<span class="hljs-string">&quot;SCUT-DLVCLab/lilt-roberta-en-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd-layoutlmv3&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;tokens&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(words, boxes=boxes, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_indices = outputs.logits.argmax(-<span class="hljs-number">1</span>)`,wrap:!1}}),{c(){n=d("p"),n.textContent=w,l=r(),u(p.$$.fragment)},l(s){n=c(s,"P",{"data-svelte-h":!0}),h(n)!=="svelte-kvfsh7"&&(n.textContent=w),l=i(s),f(p.$$.fragment,s)},m(s,v){a(s,n,v),a(s,l,v),g(p,s,v),y=!0},p:H,i(s){y||(_(p.$$.fragment,s),y=!0)},o(s){b(p.$$.fragment,s),y=!1},d(s){s&&(o(n),o(l)),T(p,s)}}}function on(M){let n,w=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=w},l(l){n=c(l,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=w)},m(l,p){a(l,n,p)},p:H,d(l){l&&o(n)}}}function nn(M){let n,w="Examples:",l,p,y;return p=new Ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBdXRvTW9kZWxGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlNDVVQtRExWQ0xhYiUyRmxpbHQtcm9iZXJ0YS1lbi1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMlNDVVQtRExWQ0xhYiUyRmxpbHQtcm9iZXJ0YS1lbi1iYXNlJTIyKSUwQSUwQWRhdGFzZXQlMjAlM0QlMjBsb2FkX2RhdGFzZXQoJTIybmllbHNyJTJGZnVuc2QtbGF5b3V0bG12MyUyMiUyQyUyMHNwbGl0JTNEJTIydHJhaW4lMjIpJTBBZXhhbXBsZSUyMCUzRCUyMGRhdGFzZXQlNUIwJTVEJTBBd29yZHMlMjAlM0QlMjBleGFtcGxlJTVCJTIydG9rZW5zJTIyJTVEJTBBYm94ZXMlMjAlM0QlMjBleGFtcGxlJTVCJTIyYmJveGVzJTIyJTVEJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIod29yZHMlMkMlMjBib3hlcyUzRGJveGVzJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiplbmNvZGluZyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBlbmNvZGluZy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEFwcmVkaWN0ZWRfYW5zd2VyJTIwJTNEJTIwdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;SCUT-DLVCLab/lilt-roberta-en-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;SCUT-DLVCLab/lilt-roberta-en-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;nielsr/funsd-layoutlmv3&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example = dataset[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>words = example[<span class="hljs-string">&quot;tokens&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = example[<span class="hljs-string">&quot;bboxes&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(words, boxes=boxes, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = encoding.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_answer = tokenizer.decode(predict_answer_tokens)`,wrap:!1}}),{c(){n=d("p"),n.textContent=w,l=r(),u(p.$$.fragment)},l(s){n=c(s,"P",{"data-svelte-h":!0}),h(n)!=="svelte-kvfsh7"&&(n.textContent=w),l=i(s),f(p.$$.fragment,s)},m(s,v){a(s,n,v),a(s,l,v),g(p,s,v),y=!0},p:H,i(s){y||(_(p.$$.fragment,s),y=!0)},o(s){b(p.$$.fragment,s),y=!1},d(s){s&&(o(n),o(l)),T(p,s)}}}function sn(M){let n,w,l,p,y,s="<em>This model was released on 2022-02-28 and added to Hugging Face Transformers on 2022-10-12.</em>",v,se,ot,Q,po='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',nt,ae,st,re,mo=`The LiLT model was proposed in <a href="https://huggingface.co/papers/2202.13669" rel="nofollow">LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding</a> by Jiapeng Wang, Lianwen Jin, Kai Ding.
LiLT allows to combine any pre-trained RoBERTa text encoder with a lightweight Layout Transformer, to enable <a href="layoutlm">LayoutLM</a>-like document understanding for many
languages.`,at,ie,ho="The abstract from the paper is the following:",rt,le,uo="<em>Structured document understanding has attracted considerable attention and made significant progress recently, owing to its crucial role in intelligent document processing. However, most existing related models can only deal with the document data of specific language(s) (typically English) included in the pre-training collection, which is extremely limited. To address this issue, we propose a simple yet effective Language-independent Layout Transformer (LiLT) for structured document understanding. LiLT can be pre-trained on the structured documents of a single language and then directly fine-tuned on other languages with the corresponding off-the-shelf monolingual/multilingual pre-trained textual models. Experimental results on eight languages have shown that LiLT can achieve competitive or even superior performance on diverse widely-used downstream benchmarks, which enables language-independent benefit from the pre-training of document layout structure.</em>",it,P,fo,lt,de,go='LiLT architecture. Taken from the <a href="https://huggingface.co/papers/2202.13669">original paper</a>.',dt,ce,_o=`This model was contributed by <a href="https://huggingface.co/nielsr" rel="nofollow">nielsr</a>.
The original code can be found <a href="https://github.com/jpwang/lilt" rel="nofollow">here</a>.`,ct,pe,pt,me,bo=`<li>To combine the Language-Independent Layout Transformer with a new RoBERTa checkpoint from the <a href="https://huggingface.co/models?search=roberta" rel="nofollow">hub</a>, refer to <a href="https://github.com/jpWang/LiLT#or-generate-your-own-checkpoint-optional" rel="nofollow">this guide</a>.
The script will result in <code>config.json</code> and <code>pytorch_model.bin</code> files being stored locally. After doing this, one can do the following (assuming you’re logged in with your HuggingFace account):</li>`,mt,he,ht,ue,To=`<li>When preparing data for the model, make sure to use the token vocabulary that corresponds to the RoBERTa checkpoint you combined with the Layout Transformer.</li> <li>As <a href="https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base" rel="nofollow">lilt-roberta-en-base</a> uses the same vocabulary as <a href="layoutlmv3">LayoutLMv3</a>, one can use <a href="/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast">LayoutLMv3TokenizerFast</a> to prepare data for the model.
The same is true for <a href="https://huggingface.co/SCUT-DLVCLab/lilt-infoxlm-base" rel="nofollow">lilt-roberta-en-base</a>: one can use <a href="/docs/transformers/v4.56.2/en/model_doc/layoutxlm#transformers.LayoutXLMTokenizerFast">LayoutXLMTokenizerFast</a> for that model.</li>`,ut,fe,ft,ge,yo="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LiLT.",gt,_e,wo='<li>Demo notebooks for LiLT can be found <a href="https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LiLT" rel="nofollow">here</a>.</li>',_t,be,vo="<strong>Documentation resources</strong>",bt,Te,Mo='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li>',Tt,ye,ko="If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",yt,we,wt,I,ve,qt,Ze,Lo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltModel">LiltModel</a>. It is used to instantiate a LiLT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the LiLT
<a href="https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base" rel="nofollow">SCUT-DLVCLab/lilt-roberta-en-base</a> architecture.
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Wt,A,vt,Me,Mt,k,ke,Zt,Ie,xo="The bare Lilt Model outputting raw hidden-states without any specific head on top.",It,Re,$o=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rt,Ve,Co=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Vt,U,Le,Xt,Xe,Fo='The <a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltModel">LiltModel</a> forward method, overrides the <code>__call__</code> special method.',Bt,Y,Gt,O,kt,xe,Lt,L,$e,Nt,Be,zo=`LiLT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,St,Ge,Jo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Et,Ne,jo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ht,q,Ce,Qt,Se,Uo='The <a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForSequenceClassification">LiltForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Pt,D,At,K,xt,Fe,$t,x,ze,Yt,Ee,qo=`The Lilt transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Ot,He,Wo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Dt,Qe,Zo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Kt,W,Je,eo,Pe,Io='The <a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForTokenClassification">LiltForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',to,ee,oo,te,Ct,je,Ft,$,Ue,no,Ae,Ro=`The Lilt transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,so,Ye,Vo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ao,Oe,Xo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ro,Z,qe,io,De,Bo='The <a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForQuestionAnswering">LiltForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',lo,oe,co,ne,zt,We,Jt,et,jt;return se=new E({props:{title:"LiLT",local:"lilt",headingTag:"h1"}}),ae=new E({props:{title:"Overview",local:"overview",headingTag:"h2"}}),pe=new E({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),he=new Ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExpbHRNb2RlbCUwQSUwQW1vZGVsJTIwJTNEJTIwTGlsdE1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJwYXRoX3RvX3lvdXJfZmlsZXMlMjIpJTBBbW9kZWwucHVzaF90b19odWIoJTIybmFtZV9vZl9yZXBvX29uX3RoZV9odWIlMjIp",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LiltModel

model = LiltModel.from_pretrained(<span class="hljs-string">&quot;path_to_your_files&quot;</span>)
model.push_to_hub(<span class="hljs-string">&quot;name_of_repo_on_the_hub&quot;</span>)`,wrap:!1}}),fe=new E({props:{title:"Resources",local:"resources",headingTag:"h2"}}),we=new E({props:{title:"LiltConfig",local:"transformers.LiltConfig",headingTag:"h2"}}),ve=new S({props:{name:"class transformers.LiltConfig",anchor:"transformers.LiltConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"classifier_dropout",val:" = None"},{name:"channel_shrink_ratio",val:" = 4"},{name:"max_2d_position_embeddings",val:" = 1024"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LiltConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the LiLT model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltModel">LiltModel</a>.`,name:"vocab_size"},{anchor:"transformers.LiltConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer. Should be a multiple of 24.`,name:"hidden_size"},{anchor:"transformers.LiltConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.LiltConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.LiltConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.LiltConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.LiltConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.LiltConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.LiltConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.LiltConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltModel">LiltModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.LiltConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.LiltConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.LiltConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.LiltConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"},{anchor:"transformers.LiltConfig.channel_shrink_ratio",description:`<strong>channel_shrink_ratio</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The shrink ratio compared to the <code>hidden_size</code> for the channel dimension of the layout embeddings.`,name:"channel_shrink_ratio"},{anchor:"transformers.LiltConfig.max_2d_position_embeddings",description:`<strong>max_2d_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum value that the 2D position embedding might ever be used with. Typically set this to something
large just in case (e.g., 1024).`,name:"max_2d_position_embeddings"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/configuration_lilt.py#L24"}}),A=new tt({props:{anchor:"transformers.LiltConfig.example",$$slots:{default:[Ao]},$$scope:{ctx:M}}}),Me=new E({props:{title:"LiltModel",local:"transformers.LiltModel",headingTag:"h2"}}),ke=new S({props:{name:"class transformers.LiltModel",anchor:"transformers.LiltModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.LiltModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltModel">LiltModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.LiltModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/modeling_lilt.py#L586"}}),Le=new S({props:{name:"forward",anchor:"transformers.LiltModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"bbox",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LiltModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LiltModel.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LiltModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LiltModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LiltModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LiltModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LiltModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LiltModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LiltModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LiltModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/modeling_lilt.py#L618",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig"
>LiltConfig</a>) and inputs.</p>
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
`}}),Y=new Ut({props:{$$slots:{default:[Yo]},$$scope:{ctx:M}}}),O=new tt({props:{anchor:"transformers.LiltModel.forward.example",$$slots:{default:[Oo]},$$scope:{ctx:M}}}),xe=new E({props:{title:"LiltForSequenceClassification",local:"transformers.LiltForSequenceClassification",headingTag:"h2"}}),$e=new S({props:{name:"class transformers.LiltForSequenceClassification",anchor:"transformers.LiltForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LiltForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForSequenceClassification">LiltForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/modeling_lilt.py#L740"}}),Ce=new S({props:{name:"forward",anchor:"transformers.LiltForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LiltForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LiltForSequenceClassification.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LiltForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LiltForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LiltForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LiltForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LiltForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LiltForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.LiltForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LiltForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LiltForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/modeling_lilt.py#L753",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig"
>LiltConfig</a>) and inputs.</p>
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
`}}),D=new Ut({props:{$$slots:{default:[Do]},$$scope:{ctx:M}}}),K=new tt({props:{anchor:"transformers.LiltForSequenceClassification.forward.example",$$slots:{default:[Ko]},$$scope:{ctx:M}}}),Fe=new E({props:{title:"LiltForTokenClassification",local:"transformers.LiltForTokenClassification",headingTag:"h2"}}),ze=new S({props:{name:"class transformers.LiltForTokenClassification",anchor:"transformers.LiltForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LiltForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForTokenClassification">LiltForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/modeling_lilt.py#L854"}}),Je=new S({props:{name:"forward",anchor:"transformers.LiltForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LiltForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LiltForTokenClassification.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LiltForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LiltForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LiltForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LiltForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LiltForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LiltForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.LiltForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LiltForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LiltForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/modeling_lilt.py#L870",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig"
>LiltConfig</a>) and inputs.</p>
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
`}}),ee=new Ut({props:{$$slots:{default:[en]},$$scope:{ctx:M}}}),te=new tt({props:{anchor:"transformers.LiltForTokenClassification.forward.example",$$slots:{default:[tn]},$$scope:{ctx:M}}}),je=new E({props:{title:"LiltForQuestionAnswering",local:"transformers.LiltForQuestionAnswering",headingTag:"h2"}}),Ue=new S({props:{name:"class transformers.LiltForQuestionAnswering",anchor:"transformers.LiltForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LiltForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltForQuestionAnswering">LiltForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/modeling_lilt.py#L976"}}),qe=new S({props:{name:"forward",anchor:"transformers.LiltForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"bbox",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LiltForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LiltForQuestionAnswering.forward.bbox",description:`<strong>bbox</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>) &#x2014;
Bounding boxes of each input sequence tokens. Selected in the range <code>[0, config.max_2d_position_embeddings-1]</code>. Each bounding box should be a normalized version in (x0, y0, x1, y1)
format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
y1) represents the position of the lower right corner. See <a href="#Overview">Overview</a> for normalization.`,name:"bbox"},{anchor:"transformers.LiltForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LiltForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LiltForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LiltForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LiltForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LiltForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.LiltForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.LiltForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LiltForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LiltForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lilt/modeling_lilt.py#L988",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/lilt#transformers.LiltConfig"
>LiltConfig</a>) and inputs.</p>
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
`}}),oe=new Ut({props:{$$slots:{default:[on]},$$scope:{ctx:M}}}),ne=new tt({props:{anchor:"transformers.LiltForQuestionAnswering.forward.example",$$slots:{default:[nn]},$$scope:{ctx:M}}}),We=new Po({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/lilt.md"}}),{c(){n=d("meta"),w=r(),l=d("p"),p=r(),y=d("p"),y.innerHTML=s,v=r(),u(se.$$.fragment),ot=r(),Q=d("div"),Q.innerHTML=po,nt=r(),u(ae.$$.fragment),st=r(),re=d("p"),re.innerHTML=mo,at=r(),ie=d("p"),ie.textContent=ho,rt=r(),le=d("p"),le.innerHTML=uo,it=r(),P=d("img"),lt=r(),de=d("small"),de.innerHTML=go,dt=r(),ce=d("p"),ce.innerHTML=_o,ct=r(),u(pe.$$.fragment),pt=r(),me=d("ul"),me.innerHTML=bo,mt=r(),u(he.$$.fragment),ht=r(),ue=d("ul"),ue.innerHTML=To,ut=r(),u(fe.$$.fragment),ft=r(),ge=d("p"),ge.textContent=yo,gt=r(),_e=d("ul"),_e.innerHTML=wo,_t=r(),be=d("p"),be.innerHTML=vo,bt=r(),Te=d("ul"),Te.innerHTML=Mo,Tt=r(),ye=d("p"),ye.textContent=ko,yt=r(),u(we.$$.fragment),wt=r(),I=d("div"),u(ve.$$.fragment),qt=r(),Ze=d("p"),Ze.innerHTML=Lo,Wt=r(),u(A.$$.fragment),vt=r(),u(Me.$$.fragment),Mt=r(),k=d("div"),u(ke.$$.fragment),Zt=r(),Ie=d("p"),Ie.textContent=xo,It=r(),Re=d("p"),Re.innerHTML=$o,Rt=r(),Ve=d("p"),Ve.innerHTML=Co,Vt=r(),U=d("div"),u(Le.$$.fragment),Xt=r(),Xe=d("p"),Xe.innerHTML=Fo,Bt=r(),u(Y.$$.fragment),Gt=r(),u(O.$$.fragment),kt=r(),u(xe.$$.fragment),Lt=r(),L=d("div"),u($e.$$.fragment),Nt=r(),Be=d("p"),Be.textContent=zo,St=r(),Ge=d("p"),Ge.innerHTML=Jo,Et=r(),Ne=d("p"),Ne.innerHTML=jo,Ht=r(),q=d("div"),u(Ce.$$.fragment),Qt=r(),Se=d("p"),Se.innerHTML=Uo,Pt=r(),u(D.$$.fragment),At=r(),u(K.$$.fragment),xt=r(),u(Fe.$$.fragment),$t=r(),x=d("div"),u(ze.$$.fragment),Yt=r(),Ee=d("p"),Ee.textContent=qo,Ot=r(),He=d("p"),He.innerHTML=Wo,Dt=r(),Qe=d("p"),Qe.innerHTML=Zo,Kt=r(),W=d("div"),u(Je.$$.fragment),eo=r(),Pe=d("p"),Pe.innerHTML=Io,to=r(),u(ee.$$.fragment),oo=r(),u(te.$$.fragment),Ct=r(),u(je.$$.fragment),Ft=r(),$=d("div"),u(Ue.$$.fragment),no=r(),Ae=d("p"),Ae.innerHTML=Ro,so=r(),Ye=d("p"),Ye.innerHTML=Vo,ao=r(),Oe=d("p"),Oe.innerHTML=Xo,ro=r(),Z=d("div"),u(qe.$$.fragment),io=r(),De=d("p"),De.innerHTML=Bo,lo=r(),u(oe.$$.fragment),co=r(),u(ne.$$.fragment),zt=r(),u(We.$$.fragment),Jt=r(),et=d("p"),this.h()},l(e){const t=Qo("svelte-u9bgzb",document.head);n=c(t,"META",{name:!0,content:!0}),t.forEach(o),w=i(e),l=c(e,"P",{}),G(l).forEach(o),p=i(e),y=c(e,"P",{"data-svelte-h":!0}),h(y)!=="svelte-anmr76"&&(y.innerHTML=s),v=i(e),f(se.$$.fragment,e),ot=i(e),Q=c(e,"DIV",{class:!0,"data-svelte-h":!0}),h(Q)!=="svelte-13t8s2t"&&(Q.innerHTML=po),nt=i(e),f(ae.$$.fragment,e),st=i(e),re=c(e,"P",{"data-svelte-h":!0}),h(re)!=="svelte-zpetvy"&&(re.innerHTML=mo),at=i(e),ie=c(e,"P",{"data-svelte-h":!0}),h(ie)!=="svelte-vfdo9a"&&(ie.textContent=ho),rt=i(e),le=c(e,"P",{"data-svelte-h":!0}),h(le)!=="svelte-1949dob"&&(le.innerHTML=uo),it=i(e),P=c(e,"IMG",{src:!0,alt:!0,width:!0}),lt=i(e),de=c(e,"SMALL",{"data-svelte-h":!0}),h(de)!=="svelte-tga31f"&&(de.innerHTML=go),dt=i(e),ce=c(e,"P",{"data-svelte-h":!0}),h(ce)!=="svelte-42ikqz"&&(ce.innerHTML=_o),ct=i(e),f(pe.$$.fragment,e),pt=i(e),me=c(e,"UL",{"data-svelte-h":!0}),h(me)!=="svelte-1yaggt4"&&(me.innerHTML=bo),mt=i(e),f(he.$$.fragment,e),ht=i(e),ue=c(e,"UL",{"data-svelte-h":!0}),h(ue)!=="svelte-l8dppp"&&(ue.innerHTML=To),ut=i(e),f(fe.$$.fragment,e),ft=i(e),ge=c(e,"P",{"data-svelte-h":!0}),h(ge)!=="svelte-tf03pc"&&(ge.textContent=yo),gt=i(e),_e=c(e,"UL",{"data-svelte-h":!0}),h(_e)!=="svelte-11y7yle"&&(_e.innerHTML=wo),_t=i(e),be=c(e,"P",{"data-svelte-h":!0}),h(be)!=="svelte-27ts0a"&&(be.innerHTML=vo),bt=i(e),Te=c(e,"UL",{"data-svelte-h":!0}),h(Te)!=="svelte-fiyac8"&&(Te.innerHTML=Mo),Tt=i(e),ye=c(e,"P",{"data-svelte-h":!0}),h(ye)!=="svelte-1xesile"&&(ye.textContent=ko),yt=i(e),f(we.$$.fragment,e),wt=i(e),I=c(e,"DIV",{class:!0});var N=G(I);f(ve.$$.fragment,N),qt=i(N),Ze=c(N,"P",{"data-svelte-h":!0}),h(Ze)!=="svelte-18xk0nr"&&(Ze.innerHTML=Lo),Wt=i(N),f(A.$$.fragment,N),N.forEach(o),vt=i(e),f(Me.$$.fragment,e),Mt=i(e),k=c(e,"DIV",{class:!0});var F=G(k);f(ke.$$.fragment,F),Zt=i(F),Ie=c(F,"P",{"data-svelte-h":!0}),h(Ie)!=="svelte-1kvbq9b"&&(Ie.textContent=xo),It=i(F),Re=c(F,"P",{"data-svelte-h":!0}),h(Re)!=="svelte-q52n56"&&(Re.innerHTML=$o),Rt=i(F),Ve=c(F,"P",{"data-svelte-h":!0}),h(Ve)!=="svelte-hswkmf"&&(Ve.innerHTML=Co),Vt=i(F),U=c(F,"DIV",{class:!0});var R=G(U);f(Le.$$.fragment,R),Xt=i(R),Xe=c(R,"P",{"data-svelte-h":!0}),h(Xe)!=="svelte-98s6u4"&&(Xe.innerHTML=Fo),Bt=i(R),f(Y.$$.fragment,R),Gt=i(R),f(O.$$.fragment,R),R.forEach(o),F.forEach(o),kt=i(e),f(xe.$$.fragment,e),Lt=i(e),L=c(e,"DIV",{class:!0});var z=G(L);f($e.$$.fragment,z),Nt=i(z),Be=c(z,"P",{"data-svelte-h":!0}),h(Be)!=="svelte-m04gc7"&&(Be.textContent=zo),St=i(z),Ge=c(z,"P",{"data-svelte-h":!0}),h(Ge)!=="svelte-q52n56"&&(Ge.innerHTML=Jo),Et=i(z),Ne=c(z,"P",{"data-svelte-h":!0}),h(Ne)!=="svelte-hswkmf"&&(Ne.innerHTML=jo),Ht=i(z),q=c(z,"DIV",{class:!0});var V=G(q);f(Ce.$$.fragment,V),Qt=i(V),Se=c(V,"P",{"data-svelte-h":!0}),h(Se)!=="svelte-1nbmua8"&&(Se.innerHTML=Uo),Pt=i(V),f(D.$$.fragment,V),At=i(V),f(K.$$.fragment,V),V.forEach(o),z.forEach(o),xt=i(e),f(Fe.$$.fragment,e),$t=i(e),x=c(e,"DIV",{class:!0});var J=G(x);f(ze.$$.fragment,J),Yt=i(J),Ee=c(J,"P",{"data-svelte-h":!0}),h(Ee)!=="svelte-tjyllt"&&(Ee.textContent=qo),Ot=i(J),He=c(J,"P",{"data-svelte-h":!0}),h(He)!=="svelte-q52n56"&&(He.innerHTML=Wo),Dt=i(J),Qe=c(J,"P",{"data-svelte-h":!0}),h(Qe)!=="svelte-hswkmf"&&(Qe.innerHTML=Zo),Kt=i(J),W=c(J,"DIV",{class:!0});var X=G(W);f(Je.$$.fragment,X),eo=i(X),Pe=c(X,"P",{"data-svelte-h":!0}),h(Pe)!=="svelte-1zc5oi"&&(Pe.innerHTML=Io),to=i(X),f(ee.$$.fragment,X),oo=i(X),f(te.$$.fragment,X),X.forEach(o),J.forEach(o),Ct=i(e),f(je.$$.fragment,e),Ft=i(e),$=c(e,"DIV",{class:!0});var j=G($);f(Ue.$$.fragment,j),no=i(j),Ae=c(j,"P",{"data-svelte-h":!0}),h(Ae)!=="svelte-tv4ayk"&&(Ae.innerHTML=Ro),so=i(j),Ye=c(j,"P",{"data-svelte-h":!0}),h(Ye)!=="svelte-q52n56"&&(Ye.innerHTML=Vo),ao=i(j),Oe=c(j,"P",{"data-svelte-h":!0}),h(Oe)!=="svelte-hswkmf"&&(Oe.innerHTML=Xo),ro=i(j),Z=c(j,"DIV",{class:!0});var B=G(Z);f(qe.$$.fragment,B),io=i(B),De=c(B,"P",{"data-svelte-h":!0}),h(De)!=="svelte-15pcgjc"&&(De.innerHTML=Bo),lo=i(B),f(oe.$$.fragment,B),co=i(B),f(ne.$$.fragment,B),B.forEach(o),j.forEach(o),zt=i(e),f(We.$$.fragment,e),Jt=i(e),et=c(e,"P",{}),G(et).forEach(o),this.h()},h(){C(n,"name","hf:doc:metadata"),C(n,"content",an),C(Q,"class","flex flex-wrap space-x-1"),No(P.src,fo="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/lilt_architecture.jpg")||C(P,"src",fo),C(P,"alt","drawing"),C(P,"width","600"),C(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){m(document.head,n),a(e,w,t),a(e,l,t),a(e,p,t),a(e,y,t),a(e,v,t),g(se,e,t),a(e,ot,t),a(e,Q,t),a(e,nt,t),g(ae,e,t),a(e,st,t),a(e,re,t),a(e,at,t),a(e,ie,t),a(e,rt,t),a(e,le,t),a(e,it,t),a(e,P,t),a(e,lt,t),a(e,de,t),a(e,dt,t),a(e,ce,t),a(e,ct,t),g(pe,e,t),a(e,pt,t),a(e,me,t),a(e,mt,t),g(he,e,t),a(e,ht,t),a(e,ue,t),a(e,ut,t),g(fe,e,t),a(e,ft,t),a(e,ge,t),a(e,gt,t),a(e,_e,t),a(e,_t,t),a(e,be,t),a(e,bt,t),a(e,Te,t),a(e,Tt,t),a(e,ye,t),a(e,yt,t),g(we,e,t),a(e,wt,t),a(e,I,t),g(ve,I,null),m(I,qt),m(I,Ze),m(I,Wt),g(A,I,null),a(e,vt,t),g(Me,e,t),a(e,Mt,t),a(e,k,t),g(ke,k,null),m(k,Zt),m(k,Ie),m(k,It),m(k,Re),m(k,Rt),m(k,Ve),m(k,Vt),m(k,U),g(Le,U,null),m(U,Xt),m(U,Xe),m(U,Bt),g(Y,U,null),m(U,Gt),g(O,U,null),a(e,kt,t),g(xe,e,t),a(e,Lt,t),a(e,L,t),g($e,L,null),m(L,Nt),m(L,Be),m(L,St),m(L,Ge),m(L,Et),m(L,Ne),m(L,Ht),m(L,q),g(Ce,q,null),m(q,Qt),m(q,Se),m(q,Pt),g(D,q,null),m(q,At),g(K,q,null),a(e,xt,t),g(Fe,e,t),a(e,$t,t),a(e,x,t),g(ze,x,null),m(x,Yt),m(x,Ee),m(x,Ot),m(x,He),m(x,Dt),m(x,Qe),m(x,Kt),m(x,W),g(Je,W,null),m(W,eo),m(W,Pe),m(W,to),g(ee,W,null),m(W,oo),g(te,W,null),a(e,Ct,t),g(je,e,t),a(e,Ft,t),a(e,$,t),g(Ue,$,null),m($,no),m($,Ae),m($,so),m($,Ye),m($,ao),m($,Oe),m($,ro),m($,Z),g(qe,Z,null),m(Z,io),m(Z,De),m(Z,lo),g(oe,Z,null),m(Z,co),g(ne,Z,null),a(e,zt,t),g(We,e,t),a(e,Jt,t),a(e,et,t),jt=!0},p(e,[t]){const N={};t&2&&(N.$$scope={dirty:t,ctx:e}),A.$set(N);const F={};t&2&&(F.$$scope={dirty:t,ctx:e}),Y.$set(F);const R={};t&2&&(R.$$scope={dirty:t,ctx:e}),O.$set(R);const z={};t&2&&(z.$$scope={dirty:t,ctx:e}),D.$set(z);const V={};t&2&&(V.$$scope={dirty:t,ctx:e}),K.$set(V);const J={};t&2&&(J.$$scope={dirty:t,ctx:e}),ee.$set(J);const X={};t&2&&(X.$$scope={dirty:t,ctx:e}),te.$set(X);const j={};t&2&&(j.$$scope={dirty:t,ctx:e}),oe.$set(j);const B={};t&2&&(B.$$scope={dirty:t,ctx:e}),ne.$set(B)},i(e){jt||(_(se.$$.fragment,e),_(ae.$$.fragment,e),_(pe.$$.fragment,e),_(he.$$.fragment,e),_(fe.$$.fragment,e),_(we.$$.fragment,e),_(ve.$$.fragment,e),_(A.$$.fragment,e),_(Me.$$.fragment,e),_(ke.$$.fragment,e),_(Le.$$.fragment,e),_(Y.$$.fragment,e),_(O.$$.fragment,e),_(xe.$$.fragment,e),_($e.$$.fragment,e),_(Ce.$$.fragment,e),_(D.$$.fragment,e),_(K.$$.fragment,e),_(Fe.$$.fragment,e),_(ze.$$.fragment,e),_(Je.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(je.$$.fragment,e),_(Ue.$$.fragment,e),_(qe.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(We.$$.fragment,e),jt=!0)},o(e){b(se.$$.fragment,e),b(ae.$$.fragment,e),b(pe.$$.fragment,e),b(he.$$.fragment,e),b(fe.$$.fragment,e),b(we.$$.fragment,e),b(ve.$$.fragment,e),b(A.$$.fragment,e),b(Me.$$.fragment,e),b(ke.$$.fragment,e),b(Le.$$.fragment,e),b(Y.$$.fragment,e),b(O.$$.fragment,e),b(xe.$$.fragment,e),b($e.$$.fragment,e),b(Ce.$$.fragment,e),b(D.$$.fragment,e),b(K.$$.fragment,e),b(Fe.$$.fragment,e),b(ze.$$.fragment,e),b(Je.$$.fragment,e),b(ee.$$.fragment,e),b(te.$$.fragment,e),b(je.$$.fragment,e),b(Ue.$$.fragment,e),b(qe.$$.fragment,e),b(oe.$$.fragment,e),b(ne.$$.fragment,e),b(We.$$.fragment,e),jt=!1},d(e){e&&(o(w),o(l),o(p),o(y),o(v),o(ot),o(Q),o(nt),o(st),o(re),o(at),o(ie),o(rt),o(le),o(it),o(P),o(lt),o(de),o(dt),o(ce),o(ct),o(pt),o(me),o(mt),o(ht),o(ue),o(ut),o(ft),o(ge),o(gt),o(_e),o(_t),o(be),o(bt),o(Te),o(Tt),o(ye),o(yt),o(wt),o(I),o(vt),o(Mt),o(k),o(kt),o(Lt),o(L),o(xt),o($t),o(x),o(Ct),o(Ft),o($),o(zt),o(Jt),o(et)),o(n),T(se,e),T(ae,e),T(pe,e),T(he,e),T(fe,e),T(we,e),T(ve),T(A),T(Me,e),T(ke),T(Le),T(Y),T(O),T(xe,e),T($e),T(Ce),T(D),T(K),T(Fe,e),T(ze),T(Je),T(ee),T(te),T(je,e),T(Ue),T(qe),T(oe),T(ne),T(We,e)}}}const an='{"title":"LiLT","local":"lilt","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"LiltConfig","local":"transformers.LiltConfig","sections":[],"depth":2},{"title":"LiltModel","local":"transformers.LiltModel","sections":[],"depth":2},{"title":"LiltForSequenceClassification","local":"transformers.LiltForSequenceClassification","sections":[],"depth":2},{"title":"LiltForTokenClassification","local":"transformers.LiltForTokenClassification","sections":[],"depth":2},{"title":"LiltForQuestionAnswering","local":"transformers.LiltForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function rn(M){return So(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class fn extends Eo{constructor(n){super(),Ho(this,n,rn,sn,Go,{})}}export{fn as component};
