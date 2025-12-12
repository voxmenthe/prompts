import{s as da,o as la,n as G}from"../chunks/scheduler.18a86fab.js";import{S as ca,i as pa,g as d,s as a,r as f,A as ma,h as l,f as i,c as r,j,x as m,u as g,k as x,y as n,a as c,v as _,d as v,t as b,w as k}from"../chunks/index.98837b22.js";import{T as yt}from"../chunks/Tip.77304350.js";import{D as Z}from"../chunks/Docstring.a1ef7999.js";import{C as E}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as K}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as P,E as ha}from"../chunks/getInferenceSnippets.06c2775f.js";function ua(y){let t,p="Example:",o,h,M;return h=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME12cENvbmZpZyUyQyUyME12cE1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyME1WUCUyMFJVQ0FJQm94JTJGbXZwJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME12cENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBSVUNBSUJveCUyRm12cCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwTXZwTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MvpConfig, MvpModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a MVP RUCAIBox/mvp style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MvpConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the RUCAIBox/mvp style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=d("p"),t.textContent=p,o=a(),f(h.$$.fragment)},l(u){t=l(u,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=p),o=r(u),g(h.$$.fragment,u)},m(u,C){c(u,t,C),c(u,o,C),_(h,u,C),M=!0},p:G,i(u){M||(v(h.$$.fragment,u),M=!0)},o(u){b(h.$$.fragment,u),M=!1},d(u){u&&(i(t),i(o)),k(h,u)}}}function fa(y){let t,p="be encoded differently whether it is at the beginning of the sentence (without space) or not:",o,h,M;return h=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME12cFRva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyME12cFRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyUlVDQUlCb3glMkZtdnAlMjIpJTBBdG9rZW5pemVyKCUyMkhlbGxvJTIwd29ybGQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTBBdG9rZW5pemVyKCUyMiUyMEhlbGxvJTIwd29ybGQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MvpTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MvpTokenizer.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">31414</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">20920</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]`,wrap:!1}}),{c(){t=d("p"),t.textContent=p,o=a(),f(h.$$.fragment)},l(u){t=l(u,"P",{"data-svelte-h":!0}),m(t)!=="svelte-12atnao"&&(t.textContent=p),o=r(u),g(h.$$.fragment,u)},m(u,C){c(u,t,C),c(u,o,C),_(h,u,C),M=!0},p:G,i(u){M||(v(h.$$.fragment,u),M=!0)},o(u){b(h.$$.fragment,u),M=!1},d(u){u&&(i(t),i(o)),k(h,u)}}}function ga(y){let t,p="When used with <code>is_split_into_words=True</code>, this tokenizer will add a space before each word (even the first one).";return{c(){t=d("p"),t.innerHTML=p},l(o){t=l(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-jhmxzm"&&(t.innerHTML=p)},m(o,h){c(o,t,h)},p:G,d(o){o&&i(t)}}}function _a(y){let t,p="be encoded differently whether it is at the beginning of the sentence (without space) or not:",o,h,M;return h=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME12cFRva2VuaXplckZhc3QlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBNdnBUb2tlbml6ZXJGYXN0LmZyb21fcHJldHJhaW5lZCglMjJSVUNBSUJveCUyRm12cCUyMiklMEF0b2tlbml6ZXIoJTIySGVsbG8lMjB3b3JsZCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMEF0b2tlbml6ZXIoJTIyJTIwSGVsbG8lMjB3b3JsZCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MvpTokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MvpTokenizerFast.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">31414</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">20920</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]`,wrap:!1}}),{c(){t=d("p"),t.textContent=p,o=a(),f(h.$$.fragment)},l(u){t=l(u,"P",{"data-svelte-h":!0}),m(t)!=="svelte-12atnao"&&(t.textContent=p),o=r(u),g(h.$$.fragment,u)},m(u,C){c(u,t,C),c(u,o,C),_(h,u,C),M=!0},p:G,i(u){M||(v(h.$$.fragment,u),M=!0)},o(u){b(h.$$.fragment,u),M=!1},d(u){u&&(i(t),i(o)),k(h,u)}}}function va(y){let t,p="When used with <code>is_split_into_words=True</code>, this tokenizer needs to be instantiated with <code>add_prefix_space=True</code>.";return{c(){t=d("p"),t.innerHTML=p},l(o){t=l(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-9gg91e"&&(t.innerHTML=p)},m(o,h){c(o,t,h)},p:G,d(o){o&&i(t)}}}function ba(y){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=p},l(o){t=l(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,h){c(o,t,h)},p:G,d(o){o&&i(t)}}}function ka(y){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=p},l(o){t=l(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,h){c(o,t,h)},p:G,d(o){o&&i(t)}}}function ya(y){let t,p;return t=new E({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME12cEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlJVQ0FJQm94JTJGbXZwJTIyKSUwQW1vZGVsJTIwJTNEJTIwTXZwRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJSVUNBSUJveCUyRm12cCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIyU3VtbWFyaXplJTNBJTIwWW91JTIwbWF5JTIwd2FudCUyMHRvJTIwc3RpY2slMjBpdCUyMHRvJTIweW91ciUyMGJvc3MlMjBhbmQlMjBsZWF2ZSUyMHlvdXIlMjBqb2IlMkMlMjBidXQlMjBkb24ndCUyMGRvJTIwaXQlMjBpZiUyMHRoZXNlJTIwYXJlJTIweW91ciUyMHJlYXNvbnMuJTIyJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUwQSklMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyQmFkJTIwUmVhc29ucyUyMFRvJTIwUXVpdCUyMFlvdXIlMjBKb2IlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQWxvc3MuYmFja3dhcmQoKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MvpForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Summarize: You may want to stick it to your boss and leave your job, but don&#x27;t do it if these are your reasons.&quot;</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;Bad Reasons To Quit Your Job&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,h){_(t,o,h),p=!0},p:G,i(o){p||(v(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){k(t,o)}}}function Ma(y){let t,p;return t=new E({props:{code:"d2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyklMEElMEFnZW5lcmF0ZWRfdGV4dCUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    generated_ids = model.generate(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,h){_(t,o,h),p=!0},p:G,i(o){p||(v(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){k(t,o)}}}function Ta(y){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=p},l(o){t=l(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,h){c(o,t,h)},p:G,d(o){o&&i(t)}}}function wa(y){let t,p;return t=new E({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME12cEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEFudW1fbGFiZWxzJTIwJTNEJTIwMiUyMCUyMCUyMyUyMGZvciUyMGV4YW1wbGUlMkMlMjB0aGlzJTIwaXMlMjBhJTIwYmluYXJ5JTIwY2xhc3NpZmljYXRpb24lMjB0YXNrJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyUlVDQUlCb3glMkZtdnAlMjIpJTBBbW9kZWwlMjAlM0QlMjBNdnBGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJSVUNBSUJveCUyRm12cCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJDbGFzc2lmeSUzQSUyMEhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigxKSUyMCUyMCUyMyUyMHRoZSUyMHJlYWwlMjBsYWJlbCUyMGZvciUyMGlucHV0cyUwQSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQWxvc3MuYmFja3dhcmQoKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MvpForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-number">2</span>  <span class="hljs-comment"># for example, this is a binary classification task</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Classify: Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">1</span>)  <span class="hljs-comment"># the real label for inputs</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,h){_(t,o,h),p=!0},p:G,i(o){p||(v(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){k(t,o)}}}function $a(y){let t,p;return t=new E({props:{code:"d2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax()`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,h){_(t,o,h),p=!0},p:G,i(o){p||(v(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){k(t,o)}}}function xa(y){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=p},l(o){t=l(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,h){c(o,t,h)},p:G,d(o){o&&i(t)}}}function Ca(y){let t,p;return t=new E({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME12cEZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyUlVDQUlCb3glMkZtdnAlMjIpJTBBbW9kZWwlMjAlM0QlMjBNdnBGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyUlVDQUlCb3glMkZtdnAlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkFuc3dlciUyMHRoZSUyMGZvbGxvd2luZyUyMHF1ZXN0aW9uJTNBJTIwV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIwJTVCU0VQJTVEJTIwSmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBKSUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxOCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE5JTVEKSUwQSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KS5sb3NzJTBBbG9zcy5iYWNrd2FyZCgp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MvpForQuestionAnswering

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Answer the following question: Who was Jim Henson? [SEP] Jim Henson was a nice puppet&quot;</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index = torch.tensor([<span class="hljs-number">18</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>target_end_index = torch.tensor([<span class="hljs-number">19</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, start_positions=target_start_index, end_positions=target_end_index).loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,h){_(t,o,h),p=!0},p:G,i(o){p||(v(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){k(t,o)}}}function Fa(y){let t,p;return t=new E({props:{code:"d2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBcHJlZGljdF9hbnN3ZXIlMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2Vucyk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer = tokenizer.decode(predict_answer_tokens)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,h){_(t,o,h),p=!0},p:G,i(o){p||(v(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){k(t,o)}}}function za(y){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=p},l(o){t=l(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,h){c(o,t,h)},p:G,d(o){o&&i(t)}}}function qa(y){let t,p="Example:",o,h,M;return h=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNdnBGb3JDYXVzYWxMTSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlJVQ0FJQm94JTJGbXZwJTIyKSUwQW1vZGVsJTIwJTNEJTIwTXZwRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMlJVQ0FJQm94JTJGbXZwJTIyJTJDJTIwYWRkX2Nyb3NzX2F0dGVudGlvbiUzREZhbHNlKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEFsaXN0KGxvZ2l0cy5zaGFwZSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MvpForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForCausalLM.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">8</span>, <span class="hljs-number">50267</span>]`,wrap:!1}}),{c(){t=d("p"),t.textContent=p,o=a(),f(h.$$.fragment)},l(u){t=l(u,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=p),o=r(u),g(h.$$.fragment,u)},m(u,C){c(u,t,C),c(u,o,C),_(h,u,C),M=!0},p:G,i(u){M||(v(h.$$.fragment,u),M=!0)},o(u){b(h.$$.fragment,u),M=!1},d(u){u&&(i(t),i(o)),k(h,u)}}}function ja(y){let t,p,o,h,M,u="<em>This model was released on 2022-06-24 and added to Hugging Face Transformers on 2022-06-29.</em>",C,xe,mo,ne,cs='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',ho,Ce,uo,Fe,ps='The MVP model was proposed in <a href="https://huggingface.co/papers/2206.12131" rel="nofollow">MVP: Multi-task Supervised Pre-training for Natural Language Generation</a> by Tianyi Tang, Junyi Li, Wayne Xin Zhao and Ji-Rong Wen.',fo,ze,ms="According to the abstract,",go,qe,hs="<li>MVP follows a standard Transformer encoder-decoder architecture.</li> <li>MVP is supervised pre-trained using labeled datasets.</li> <li>MVP also has task-specific soft prompts to stimulate the model’s capacity in performing a certain task.</li> <li>MVP is specially designed for natural language generation and can be adapted to a wide range of generation tasks, including but not limited to summarization, data-to-text generation, open-ended dialogue system, story generation, question answering, question generation, task-oriented dialogue system, commonsense generation, paraphrase generation, text style transfer, and text simplification. Our model can also be adapted to natural language understanding tasks such as sequence classification and (extractive) question answering.</li>",_o,je,us='This model was contributed by <a href="https://huggingface.co/StevenTang" rel="nofollow">Tianyi Tang</a>. The detailed information and instructions can be found <a href="https://github.com/RUCAIBox/MVP" rel="nofollow">here</a>.',vo,Ue,bo,Je,fs='<li>We have released a series of models <a href="https://huggingface.co/models?filter=mvp" rel="nofollow">here</a>, including MVP, MVP with task-specific prompts, and multi-task pre-trained variants.</li> <li>If you want to use a model without prompts (standard Transformer), you can load it through <code>MvpForConditionalGeneration.from_pretrained(&#39;RUCAIBox/mvp&#39;)</code>.</li> <li>If you want to use a model with task-specific prompts, such as summarization, you can load it through <code>MvpForConditionalGeneration.from_pretrained(&#39;RUCAIBox/mvp-summarization&#39;)</code>.</li> <li>Our model supports lightweight prompt tuning following <a href="https://huggingface.co/papers/2101.00190" rel="nofollow">Prefix-tuning</a> with method <code>set_lightweight_tuning()</code>.</li>',ko,Ie,yo,We,gs="For summarization, it is an example to use MVP and MVP with summarization-specific prompts.",Mo,Ze,To,Ge,_s="For data-to-text generation, it is an example to use MVP and multi-task pre-trained variants.",wo,Ve,$o,Be,vs='For lightweight tuning, <em>i.e.</em>, fixing the model and only tuning prompts, you can load MVP with randomly initialized prompts or with task-specific prompts. Our code also supports Prefix-tuning with BART following the <a href="https://huggingface.co/papers/2101.00190" rel="nofollow">original paper</a>.',xo,Re,Co,Se,Fo,He,bs='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/translation">Translation task guide</a></li> <li><a href="../tasks/summarization">Summarization task guide</a></li>',zo,Le,qo,H,Xe,Eo,Mt,ks=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpModel">MvpModel</a>. It is used to instantiate a MVP model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the MVP <a href="https://huggingface.co/RUCAIBox/mvp" rel="nofollow">RUCAIBox/mvp</a>
architecture.`,Ao,Tt,ys=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Oo,se,jo,Qe,Uo,T,Ne,Yo,wt,Ms="Constructs a MVP tokenizer, which is smilar to the RoBERTa tokenizer, using byte-level Byte-Pair-Encoding.",Do,$t,Ts="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",Ko,ae,en,xt,ws=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.`,tn,re,on,Ct,$s=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,nn,Y,Pe,sn,Ft,xs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A MVP sequence has the following format:`,an,zt,Cs="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",rn,ie,Ee,dn,qt,Fs="Converts a sequence of tokens (string) in a single string.",ln,de,Ae,cn,jt,zs=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. MVP does not
make use of token type ids, therefore a list of zeros is returned.`,pn,le,Oe,mn,Ut,qs=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Jo,Ye,Io,$,De,hn,Jt,js=`Construct a “fast” MVP tokenizer (backed by HuggingFace’s <em>tokenizers</em> library), derived from the GPT-2 tokenizer,
using byte-level Byte-Pair-Encoding.`,un,It,Us="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",fn,ce,gn,Wt,Js=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.`,_n,pe,vn,Zt,Is=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,bn,me,Ke,kn,Gt,Ws=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. MVP does not
make use of token type ids, therefore a list of zeros is returned.`,Wo,et,Zo,V,tt,yn,Vt,Zs="The bare Mvp Model outputting raw hidden-states without any specific head on top.",Mn,Bt,Gs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Tn,Rt,Vs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wn,D,ot,$n,St,Bs='The <a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpModel">MvpModel</a> forward method, overrides the <code>__call__</code> special method.',xn,he,Go,nt,Vo,B,st,Cn,Ht,Rs="The MVP Model with a language modeling head. Can be used for various text generation tasks.",Fn,Lt,Ss=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zn,Xt,Hs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,qn,F,at,jn,Qt,Ls='The <a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForConditionalGeneration">MvpForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Un,ue,Jn,Nt,Xs="Example of summarization:",In,Pt,Qs="Fine-tuning a model",Wn,fe,Zn,Et,Ns="Inference after the model fine-tuned",Gn,ge,Bo,rt,Ro,R,it,Vn,At,Ps=`Mvp model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.`,Bn,Ot,Es=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rn,Yt,As=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Sn,z,dt,Hn,Dt,Os='The <a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForSequenceClassification">MvpForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Ln,_e,Xn,Kt,Ys="Example of single-label classification:",Qn,eo,Ds="Fine-tuning a model on <code>num_labels</code> classes",Nn,ve,Pn,to,Ks="Inference after the model fine-tuned",En,be,So,lt,Ho,S,ct,An,oo,ea=`The Mvp transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,On,no,ta=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Yn,so,oa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Dn,q,pt,Kn,ao,na='The <a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForQuestionAnswering">MvpForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',es,ke,ts,ro,sa="Example:",os,io,aa=`Fine-tuning a model for extrative question answering, and our model also supports generative question answering
using <code>BartForConditionalGeneration</code>`,ns,ye,ss,lo,ra="Inference after the model fine-tuned",as,Me,Lo,mt,Xo,ee,ht,rs,A,ut,is,co,ia='The <a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForCausalLM">MvpForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',ds,Te,ls,we,Qo,ft,No,po,Po;return xe=new P({props:{title:"MVP",local:"mvp",headingTag:"h1"}}),Ce=new P({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ue=new P({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Ie=new P({props:{title:"Usage examples",local:"usage-examples",headingTag:"h2"}}),Ze=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME12cFRva2VuaXplciUyQyUyME12cEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyME12cFRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyUlVDQUlCb3glMkZtdnAlMjIpJTBBbW9kZWwlMjAlM0QlMjBNdnBGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMlJVQ0FJQm94JTJGbXZwJTIyKSUwQW1vZGVsX3dpdGhfcHJvbXB0JTIwJTNEJTIwTXZwRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJSVUNBSUJveCUyRm12cC1zdW1tYXJpemF0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJTdW1tYXJpemUlM0ElMjBZb3UlMjBtYXklMjB3YW50JTIwdG8lMjBzdGljayUyMGl0JTIwdG8lMjB5b3VyJTIwYm9zcyUyMGFuZCUyMGxlYXZlJTIweW91ciUyMGpvYiUyQyUyMGJ1dCUyMGRvbid0JTIwZG8lMjBpdCUyMGlmJTIwdGhlc2UlMjBhcmUlMjB5b3VyJTIwcmVhc29ucy4lMjIlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBKSUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlZF9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEFnZW5lcmF0ZWRfaWRzJTIwJTNEJTIwbW9kZWxfd2l0aF9wcm9tcHQuZ2VuZXJhdGUoKippbnB1dHMpJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZWRfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MvpTokenizer, MvpForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MvpTokenizer.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model_with_prompt = MvpForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp-summarization&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Summarize: You may want to stick it to your boss and leave your job, but don&#x27;t do it if these are your reasons.&quot;</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
[<span class="hljs-string">&quot;Why You Shouldn&#x27;t Quit Your Job&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model_with_prompt.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
[<span class="hljs-string">&quot;Don&#x27;t do it if these are your reasons&quot;</span>]`,wrap:!1}}),Ve=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME12cFRva2VuaXplckZhc3QlMkMlMjBNdnBGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBNdnBUb2tlbml6ZXJGYXN0LmZyb21fcHJldHJhaW5lZCglMjJSVUNBSUJveCUyRm12cCUyMiklMEFtb2RlbCUyMCUzRCUyME12cEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyUlVDQUlCb3glMkZtdnAlMjIpJTBBbW9kZWxfd2l0aF9tdGwlMjAlM0QlMjBNdnBGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMlJVQ0FJQm94JTJGbXRsLWRhdGEtdG8tdGV4dCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIyRGVzY3JpYmUlMjB0aGUlMjBmb2xsb3dpbmclMjBkYXRhJTNBJTIwSXJvbiUyME1hbiUyMCU3QyUyMGluc3RhbmNlJTIwb2YlMjAlN0MlMjBTdXBlcmhlcm8lMjAlNUJTRVAlNUQlMjBTdGFuJTIwTGVlJTIwJTdDJTIwY3JlYXRvciUyMCU3QyUyMElyb24lMjBNYW4lMjIlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBKSUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlZF9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEFnZW5lcmF0ZWRfaWRzJTIwJTNEJTIwbW9kZWxfd2l0aF9tdGwuZ2VuZXJhdGUoKippbnB1dHMpJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZWRfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MvpTokenizerFast, MvpForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MvpTokenizerFast.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model_with_mtl = MvpForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mtl-data-to-text&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Describe the following data: Iron Man | instance of | Superhero [SEP] Stan Lee | creator | Iron Man&quot;</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
[<span class="hljs-string">&#x27;Stan Lee created the character of Iron Man, a fictional superhero appearing in American comic&#x27;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model_with_mtl.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
[<span class="hljs-string">&#x27;Iron Man is a fictional superhero appearing in American comic books published by Marvel Comics.&#x27;</span>]`,wrap:!1}}),Re=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME12cEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQW1vZGVsJTIwJTNEJTIwTXZwRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJSVUNBSUJveCUyRm12cCUyMiUyQyUyMHVzZV9wcm9tcHQlM0RUcnVlKSUwQSUyMyUyMHRoZSUyMG51bWJlciUyMG9mJTIwdHJhaW5hYmxlJTIwcGFyYW1ldGVycyUyMChmdWxsJTIwdHVuaW5nKSUwQXN1bShwLm51bWVsKCklMjBmb3IlMjBwJTIwaW4lMjBtb2RlbC5wYXJhbWV0ZXJzKCklMjBpZiUyMHAucmVxdWlyZXNfZ3JhZCklMEElMEElMjMlMjBsaWdodHdlaWdodCUyMHR1bmluZyUyMHdpdGglMjByYW5kb21seSUyMGluaXRpYWxpemVkJTIwcHJvbXB0cyUwQW1vZGVsLnNldF9saWdodHdlaWdodF90dW5pbmcoKSUwQSUyMyUyMHRoZSUyMG51bWJlciUyMG9mJTIwdHJhaW5hYmxlJTIwcGFyYW1ldGVycyUyMChsaWdodHdlaWdodCUyMHR1bmluZyklMEFzdW0ocC5udW1lbCgpJTIwZm9yJTIwcCUyMGluJTIwbW9kZWwucGFyYW1ldGVycygpJTIwaWYlMjBwLnJlcXVpcmVzX2dyYWQpJTBBJTBBJTIzJTIwbGlnaHR3ZWlnaHQlMjB0dW5pbmclMjB3aXRoJTIwdGFzay1zcGVjaWZpYyUyMHByb21wdHMlMEFtb2RlbCUyMCUzRCUyME12cEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyUlVDQUlCb3glMkZtdGwtZGF0YS10by10ZXh0JTIyKSUwQW1vZGVsLnNldF9saWdodHdlaWdodF90dW5pbmcoKSUwQSUyMyUyMG9yaWdpbmFsJTIwbGlnaHR3ZWlnaHQlMjBQcmVmaXgtdHVuaW5nJTBBbW9kZWwlMjAlM0QlMjBNdnBGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGYmFydC1sYXJnZSUyMiUyQyUyMHVzZV9wcm9tcHQlM0RUcnVlKSUwQW1vZGVsLnNldF9saWdodHdlaWdodF90dW5pbmcoKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MvpForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mvp&quot;</span>, use_prompt=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the number of trainable parameters (full tuning)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">sum</span>(p.numel() <span class="hljs-keyword">for</span> p <span class="hljs-keyword">in</span> model.parameters() <span class="hljs-keyword">if</span> p.requires_grad)
<span class="hljs-number">468116832</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># lightweight tuning with randomly initialized prompts</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model.set_lightweight_tuning()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the number of trainable parameters (lightweight tuning)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">sum</span>(p.numel() <span class="hljs-keyword">for</span> p <span class="hljs-keyword">in</span> model.parameters() <span class="hljs-keyword">if</span> p.requires_grad)
<span class="hljs-number">61823328</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># lightweight tuning with task-specific prompts</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;RUCAIBox/mtl-data-to-text&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model.set_lightweight_tuning()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># original lightweight Prefix-tuning</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MvpForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/bart-large&quot;</span>, use_prompt=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model.set_lightweight_tuning()`,wrap:!1}}),Se=new P({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Le=new P({props:{title:"MvpConfig",local:"transformers.MvpConfig",headingTag:"h2"}}),Xe=new Z({props:{name:"class transformers.MvpConfig",anchor:"transformers.MvpConfig",parameters:[{name:"vocab_size",val:" = 50267"},{name:"max_position_embeddings",val:" = 1024"},{name:"encoder_layers",val:" = 12"},{name:"encoder_ffn_dim",val:" = 4096"},{name:"encoder_attention_heads",val:" = 16"},{name:"decoder_layers",val:" = 12"},{name:"decoder_ffn_dim",val:" = 4096"},{name:"decoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 1024"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"classifier_dropout",val:" = 0.0"},{name:"scale_embedding",val:" = False"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"is_encoder_decoder",val:" = True"},{name:"decoder_start_token_id",val:" = 2"},{name:"forced_eos_token_id",val:" = 2"},{name:"use_prompt",val:" = False"},{name:"prompt_length",val:" = 100"},{name:"prompt_mid_dim",val:" = 800"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MvpConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50267) &#x2014;
Vocabulary size of the MVP model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpModel">MvpModel</a>.`,name:"vocab_size"},{anchor:"transformers.MvpConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.MvpConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.MvpConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.MvpConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.MvpConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.MvpConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.MvpConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.MvpConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.MvpConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.MvpConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.MvpConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.MvpConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.MvpConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.MvpConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.MvpConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.MvpConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.MvpConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.MvpConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.MvpConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"},{anchor:"transformers.MvpConfig.use_prompt",description:`<strong>use_prompt</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use prompt.`,name:"use_prompt"},{anchor:"transformers.MvpConfig.prompt_length",description:`<strong>prompt_length</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
The length of prompt.`,name:"prompt_length"},{anchor:"transformers.MvpConfig.prompt_mid_dim",description:`<strong>prompt_mid_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 800) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; layer in prompt.`,name:"prompt_mid_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/configuration_mvp.py#L26"}}),se=new K({props:{anchor:"transformers.MvpConfig.example",$$slots:{default:[ua]},$$scope:{ctx:y}}}),Qe=new P({props:{title:"MvpTokenizer",local:"transformers.MvpTokenizer",headingTag:"h2"}}),Ne=new Z({props:{name:"class transformers.MvpTokenizer",anchor:"transformers.MvpTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MvpTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.MvpTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.MvpTokenizer.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.MvpTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.MvpTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.MvpTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MvpTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MvpTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MvpTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MvpTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MvpTokenizer.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (MVP tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/tokenization_mvp.py#L74"}}),ae=new K({props:{anchor:"transformers.MvpTokenizer.example",$$slots:{default:[fa]},$$scope:{ctx:y}}}),re=new yt({props:{$$slots:{default:[ga]},$$scope:{ctx:y}}}),Pe=new Z({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MvpTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MvpTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MvpTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/tokenization_mvp.py#L312",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ee=new Z({props:{name:"convert_tokens_to_string",anchor:"transformers.MvpTokenizer.convert_tokens_to_string",parameters:[{name:"tokens",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/tokenization_mvp.py#L277"}}),Ae=new Z({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.MvpTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MvpTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.MvpTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/tokenization_mvp.py#L364",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Oe=new Z({props:{name:"get_special_tokens_mask",anchor:"transformers.MvpTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MvpTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.MvpTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.MvpTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/tokenization_mvp.py#L337",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ye=new P({props:{title:"MvpTokenizerFast",local:"transformers.MvpTokenizerFast",headingTag:"h2"}}),De=new Z({props:{name:"class transformers.MvpTokenizerFast",anchor:"transformers.MvpTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"trim_offsets",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MvpTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.MvpTokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.MvpTokenizerFast.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.MvpTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.MvpTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.MvpTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MvpTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MvpTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MvpTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MvpTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MvpTokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (MVP tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"},{anchor:"transformers.MvpTokenizerFast.trim_offsets",description:`<strong>trim_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the post processing step should trim offsets to avoid including whitespaces.`,name:"trim_offsets"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/tokenization_mvp_fast.py#L35"}}),ce=new K({props:{anchor:"transformers.MvpTokenizerFast.example",$$slots:{default:[_a]},$$scope:{ctx:y}}}),pe=new yt({props:{$$slots:{default:[va]},$$scope:{ctx:y}}}),Ke=new Z({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.MvpTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MvpTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.MvpTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/tokenization_mvp_fast.py#L250",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),et=new P({props:{title:"MvpModel",local:"transformers.MvpModel",headingTag:"h2"}}),tt=new Z({props:{name:"class transformers.MvpModel",anchor:"transformers.MvpModel",parameters:[{name:"config",val:": MvpConfig"}],parametersDescription:[{anchor:"transformers.MvpModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig">MvpConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L962"}}),ot=new Z({props:{name:"forward",anchor:"transformers.MvpModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MvpModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MvpModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MvpModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Mvp uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.MvpModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read <code>modeling_mvp._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.MvpModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MvpModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MvpModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MvpModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MvpModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MvpModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MvpModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MvpModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MvpModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MvpModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MvpModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MvpModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L998",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig"
>MvpConfig</a>) and inputs.</p>
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
`}}),he=new yt({props:{$$slots:{default:[ba]},$$scope:{ctx:y}}}),nt=new P({props:{title:"MvpForConditionalGeneration",local:"transformers.MvpForConditionalGeneration",headingTag:"h2"}}),st=new Z({props:{name:"class transformers.MvpForConditionalGeneration",anchor:"transformers.MvpForConditionalGeneration",parameters:[{name:"config",val:": MvpConfig"}],parametersDescription:[{anchor:"transformers.MvpForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig">MvpConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L1123"}}),at=new Z({props:{name:"forward",anchor:"transformers.MvpForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MvpForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MvpForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MvpForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Mvp uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.MvpForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read <code>modeling_mvp._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.MvpForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MvpForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MvpForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MvpForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MvpForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MvpForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MvpForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MvpForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MvpForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MvpForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MvpForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MvpForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MvpForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L1161",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig"
>MvpConfig</a>) and inputs.</p>
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
`}}),ue=new yt({props:{$$slots:{default:[ka]},$$scope:{ctx:y}}}),fe=new K({props:{anchor:"transformers.MvpForConditionalGeneration.forward.example",$$slots:{default:[ya]},$$scope:{ctx:y}}}),ge=new K({props:{anchor:"transformers.MvpForConditionalGeneration.forward.example-2",$$slots:{default:[Ma]},$$scope:{ctx:y}}}),rt=new P({props:{title:"MvpForSequenceClassification",local:"transformers.MvpForSequenceClassification",headingTag:"h2"}}),it=new Z({props:{name:"class transformers.MvpForSequenceClassification",anchor:"transformers.MvpForSequenceClassification",parameters:[{name:"config",val:": MvpConfig"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MvpForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig">MvpConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L1305"}}),dt=new Z({props:{name:"forward",anchor:"transformers.MvpForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MvpForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MvpForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MvpForSequenceClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Mvp uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.MvpForSequenceClassification.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read <code>modeling_mvp._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.MvpForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MvpForSequenceClassification.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MvpForSequenceClassification.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MvpForSequenceClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MvpForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MvpForSequenceClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MvpForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.MvpForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MvpForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MvpForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MvpForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L1325",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig"
>MvpConfig</a>) and inputs.</p>
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
`}}),_e=new yt({props:{$$slots:{default:[Ta]},$$scope:{ctx:y}}}),ve=new K({props:{anchor:"transformers.MvpForSequenceClassification.forward.example",$$slots:{default:[wa]},$$scope:{ctx:y}}}),be=new K({props:{anchor:"transformers.MvpForSequenceClassification.forward.example-2",$$slots:{default:[$a]},$$scope:{ctx:y}}}),lt=new P({props:{title:"MvpForQuestionAnswering",local:"transformers.MvpForQuestionAnswering",headingTag:"h2"}}),ct=new Z({props:{name:"class transformers.MvpForQuestionAnswering",anchor:"transformers.MvpForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MvpForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpForQuestionAnswering">MvpForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L1478"}}),pt=new Z({props:{name:"forward",anchor:"transformers.MvpForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MvpForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MvpForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MvpForQuestionAnswering.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Mvp uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.MvpForQuestionAnswering.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read <code>modeling_mvp._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.MvpForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MvpForQuestionAnswering.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MvpForQuestionAnswering.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MvpForQuestionAnswering.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MvpForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.MvpForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.MvpForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MvpForQuestionAnswering.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MvpForQuestionAnswering.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MvpForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MvpForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MvpForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L1497",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput"
>transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig"
>MvpConfig</a>) and inputs.</p>
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
`}}),ke=new yt({props:{$$slots:{default:[xa]},$$scope:{ctx:y}}}),ye=new K({props:{anchor:"transformers.MvpForQuestionAnswering.forward.example",$$slots:{default:[Ca]},$$scope:{ctx:y}}}),Me=new K({props:{anchor:"transformers.MvpForQuestionAnswering.forward.example-2",$$slots:{default:[Fa]},$$scope:{ctx:y}}}),mt=new P({props:{title:"MvpForCausalLM",local:"transformers.MvpForCausalLM",headingTag:"h2"}}),ht=new Z({props:{name:"class transformers.MvpForCausalLM",anchor:"transformers.MvpForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L1661"}}),ut=new Z({props:{name:"forward",anchor:"transformers.MvpForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MvpForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MvpForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MvpForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MvpForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.MvpForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MvpForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MvpForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MvpForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MvpForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MvpForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MvpForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MvpForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MvpForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MvpForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mvp/modeling_mvp.py#L1691",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mvp#transformers.MvpConfig"
>MvpConfig</a>) and inputs.</p>
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
`}}),Te=new yt({props:{$$slots:{default:[za]},$$scope:{ctx:y}}}),we=new K({props:{anchor:"transformers.MvpForCausalLM.forward.example",$$slots:{default:[qa]},$$scope:{ctx:y}}}),ft=new ha({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mvp.md"}}),{c(){t=d("meta"),p=a(),o=d("p"),h=a(),M=d("p"),M.innerHTML=u,C=a(),f(xe.$$.fragment),mo=a(),ne=d("div"),ne.innerHTML=cs,ho=a(),f(Ce.$$.fragment),uo=a(),Fe=d("p"),Fe.innerHTML=ps,fo=a(),ze=d("p"),ze.textContent=ms,go=a(),qe=d("ul"),qe.innerHTML=hs,_o=a(),je=d("p"),je.innerHTML=us,vo=a(),f(Ue.$$.fragment),bo=a(),Je=d("ul"),Je.innerHTML=fs,ko=a(),f(Ie.$$.fragment),yo=a(),We=d("p"),We.textContent=gs,Mo=a(),f(Ze.$$.fragment),To=a(),Ge=d("p"),Ge.textContent=_s,wo=a(),f(Ve.$$.fragment),$o=a(),Be=d("p"),Be.innerHTML=vs,xo=a(),f(Re.$$.fragment),Co=a(),f(Se.$$.fragment),Fo=a(),He=d("ul"),He.innerHTML=bs,zo=a(),f(Le.$$.fragment),qo=a(),H=d("div"),f(Xe.$$.fragment),Eo=a(),Mt=d("p"),Mt.innerHTML=ks,Ao=a(),Tt=d("p"),Tt.innerHTML=ys,Oo=a(),f(se.$$.fragment),jo=a(),f(Qe.$$.fragment),Uo=a(),T=d("div"),f(Ne.$$.fragment),Yo=a(),wt=d("p"),wt.textContent=Ms,Do=a(),$t=d("p"),$t.textContent=Ts,Ko=a(),f(ae.$$.fragment),en=a(),xt=d("p"),xt.innerHTML=ws,tn=a(),f(re.$$.fragment),on=a(),Ct=d("p"),Ct.innerHTML=$s,nn=a(),Y=d("div"),f(Pe.$$.fragment),sn=a(),Ft=d("p"),Ft.textContent=xs,an=a(),zt=d("ul"),zt.innerHTML=Cs,rn=a(),ie=d("div"),f(Ee.$$.fragment),dn=a(),qt=d("p"),qt.textContent=Fs,ln=a(),de=d("div"),f(Ae.$$.fragment),cn=a(),jt=d("p"),jt.textContent=zs,pn=a(),le=d("div"),f(Oe.$$.fragment),mn=a(),Ut=d("p"),Ut.innerHTML=qs,Jo=a(),f(Ye.$$.fragment),Io=a(),$=d("div"),f(De.$$.fragment),hn=a(),Jt=d("p"),Jt.innerHTML=js,un=a(),It=d("p"),It.textContent=Us,fn=a(),f(ce.$$.fragment),gn=a(),Wt=d("p"),Wt.innerHTML=Js,_n=a(),f(pe.$$.fragment),vn=a(),Zt=d("p"),Zt.innerHTML=Is,bn=a(),me=d("div"),f(Ke.$$.fragment),kn=a(),Gt=d("p"),Gt.textContent=Ws,Wo=a(),f(et.$$.fragment),Zo=a(),V=d("div"),f(tt.$$.fragment),yn=a(),Vt=d("p"),Vt.textContent=Zs,Mn=a(),Bt=d("p"),Bt.innerHTML=Gs,Tn=a(),Rt=d("p"),Rt.innerHTML=Vs,wn=a(),D=d("div"),f(ot.$$.fragment),$n=a(),St=d("p"),St.innerHTML=Bs,xn=a(),f(he.$$.fragment),Go=a(),f(nt.$$.fragment),Vo=a(),B=d("div"),f(st.$$.fragment),Cn=a(),Ht=d("p"),Ht.textContent=Rs,Fn=a(),Lt=d("p"),Lt.innerHTML=Ss,zn=a(),Xt=d("p"),Xt.innerHTML=Hs,qn=a(),F=d("div"),f(at.$$.fragment),jn=a(),Qt=d("p"),Qt.innerHTML=Ls,Un=a(),f(ue.$$.fragment),Jn=a(),Nt=d("p"),Nt.textContent=Xs,In=a(),Pt=d("p"),Pt.textContent=Qs,Wn=a(),f(fe.$$.fragment),Zn=a(),Et=d("p"),Et.textContent=Ns,Gn=a(),f(ge.$$.fragment),Bo=a(),f(rt.$$.fragment),Ro=a(),R=d("div"),f(it.$$.fragment),Vn=a(),At=d("p"),At.textContent=Ps,Bn=a(),Ot=d("p"),Ot.innerHTML=Es,Rn=a(),Yt=d("p"),Yt.innerHTML=As,Sn=a(),z=d("div"),f(dt.$$.fragment),Hn=a(),Dt=d("p"),Dt.innerHTML=Os,Ln=a(),f(_e.$$.fragment),Xn=a(),Kt=d("p"),Kt.textContent=Ys,Qn=a(),eo=d("p"),eo.innerHTML=Ds,Nn=a(),f(ve.$$.fragment),Pn=a(),to=d("p"),to.textContent=Ks,En=a(),f(be.$$.fragment),So=a(),f(lt.$$.fragment),Ho=a(),S=d("div"),f(ct.$$.fragment),An=a(),oo=d("p"),oo.innerHTML=ea,On=a(),no=d("p"),no.innerHTML=ta,Yn=a(),so=d("p"),so.innerHTML=oa,Dn=a(),q=d("div"),f(pt.$$.fragment),Kn=a(),ao=d("p"),ao.innerHTML=na,es=a(),f(ke.$$.fragment),ts=a(),ro=d("p"),ro.textContent=sa,os=a(),io=d("p"),io.innerHTML=aa,ns=a(),f(ye.$$.fragment),ss=a(),lo=d("p"),lo.textContent=ra,as=a(),f(Me.$$.fragment),Lo=a(),f(mt.$$.fragment),Xo=a(),ee=d("div"),f(ht.$$.fragment),rs=a(),A=d("div"),f(ut.$$.fragment),is=a(),co=d("p"),co.innerHTML=ia,ds=a(),f(Te.$$.fragment),ls=a(),f(we.$$.fragment),Qo=a(),f(ft.$$.fragment),No=a(),po=d("p"),this.h()},l(e){const s=ma("svelte-u9bgzb",document.head);t=l(s,"META",{name:!0,content:!0}),s.forEach(i),p=r(e),o=l(e,"P",{}),j(o).forEach(i),h=r(e),M=l(e,"P",{"data-svelte-h":!0}),m(M)!=="svelte-1sl83gj"&&(M.innerHTML=u),C=r(e),g(xe.$$.fragment,e),mo=r(e),ne=l(e,"DIV",{class:!0,"data-svelte-h":!0}),m(ne)!=="svelte-13t8s2t"&&(ne.innerHTML=cs),ho=r(e),g(Ce.$$.fragment,e),uo=r(e),Fe=l(e,"P",{"data-svelte-h":!0}),m(Fe)!=="svelte-ch4bsh"&&(Fe.innerHTML=ps),fo=r(e),ze=l(e,"P",{"data-svelte-h":!0}),m(ze)!=="svelte-1j5100k"&&(ze.textContent=ms),go=r(e),qe=l(e,"UL",{"data-svelte-h":!0}),m(qe)!=="svelte-wsulm6"&&(qe.innerHTML=hs),_o=r(e),je=l(e,"P",{"data-svelte-h":!0}),m(je)!=="svelte-z7b5m6"&&(je.innerHTML=us),vo=r(e),g(Ue.$$.fragment,e),bo=r(e),Je=l(e,"UL",{"data-svelte-h":!0}),m(Je)!=="svelte-b98k9s"&&(Je.innerHTML=fs),ko=r(e),g(Ie.$$.fragment,e),yo=r(e),We=l(e,"P",{"data-svelte-h":!0}),m(We)!=="svelte-6z8y0c"&&(We.textContent=gs),Mo=r(e),g(Ze.$$.fragment,e),To=r(e),Ge=l(e,"P",{"data-svelte-h":!0}),m(Ge)!=="svelte-15tmjxf"&&(Ge.textContent=_s),wo=r(e),g(Ve.$$.fragment,e),$o=r(e),Be=l(e,"P",{"data-svelte-h":!0}),m(Be)!=="svelte-15ckr5j"&&(Be.innerHTML=vs),xo=r(e),g(Re.$$.fragment,e),Co=r(e),g(Se.$$.fragment,e),Fo=r(e),He=l(e,"UL",{"data-svelte-h":!0}),m(He)!=="svelte-1mjt1kg"&&(He.innerHTML=bs),zo=r(e),g(Le.$$.fragment,e),qo=r(e),H=l(e,"DIV",{class:!0});var O=j(H);g(Xe.$$.fragment,O),Eo=r(O),Mt=l(O,"P",{"data-svelte-h":!0}),m(Mt)!=="svelte-vnrx8c"&&(Mt.innerHTML=ks),Ao=r(O),Tt=l(O,"P",{"data-svelte-h":!0}),m(Tt)!=="svelte-1ek1ss9"&&(Tt.innerHTML=ys),Oo=r(O),g(se.$$.fragment,O),O.forEach(i),jo=r(e),g(Qe.$$.fragment,e),Uo=r(e),T=l(e,"DIV",{class:!0});var w=j(T);g(Ne.$$.fragment,w),Yo=r(w),wt=l(w,"P",{"data-svelte-h":!0}),m(wt)!=="svelte-34vcv8"&&(wt.textContent=Ms),Do=r(w),$t=l(w,"P",{"data-svelte-h":!0}),m($t)!=="svelte-1s077p3"&&($t.textContent=Ts),Ko=r(w),g(ae.$$.fragment,w),en=r(w),xt=l(w,"P",{"data-svelte-h":!0}),m(xt)!=="svelte-1jfcabo"&&(xt.innerHTML=ws),tn=r(w),g(re.$$.fragment,w),on=r(w),Ct=l(w,"P",{"data-svelte-h":!0}),m(Ct)!=="svelte-ntrhio"&&(Ct.innerHTML=$s),nn=r(w),Y=l(w,"DIV",{class:!0});var te=j(Y);g(Pe.$$.fragment,te),sn=r(te),Ft=l(te,"P",{"data-svelte-h":!0}),m(Ft)!=="svelte-1w2ttey"&&(Ft.textContent=xs),an=r(te),zt=l(te,"UL",{"data-svelte-h":!0}),m(zt)!=="svelte-rq8uot"&&(zt.innerHTML=Cs),te.forEach(i),rn=r(w),ie=l(w,"DIV",{class:!0});var gt=j(ie);g(Ee.$$.fragment,gt),dn=r(gt),qt=l(gt,"P",{"data-svelte-h":!0}),m(qt)!=="svelte-b3k2yi"&&(qt.textContent=Fs),gt.forEach(i),ln=r(w),de=l(w,"DIV",{class:!0});var _t=j(de);g(Ae.$$.fragment,_t),cn=r(_t),jt=l(_t,"P",{"data-svelte-h":!0}),m(jt)!=="svelte-ycp5iu"&&(jt.textContent=zs),_t.forEach(i),pn=r(w),le=l(w,"DIV",{class:!0});var vt=j(le);g(Oe.$$.fragment,vt),mn=r(vt),Ut=l(vt,"P",{"data-svelte-h":!0}),m(Ut)!=="svelte-1f4f5kp"&&(Ut.innerHTML=qs),vt.forEach(i),w.forEach(i),Jo=r(e),g(Ye.$$.fragment,e),Io=r(e),$=l(e,"DIV",{class:!0});var U=j($);g(De.$$.fragment,U),hn=r(U),Jt=l(U,"P",{"data-svelte-h":!0}),m(Jt)!=="svelte-9entgk"&&(Jt.innerHTML=js),un=r(U),It=l(U,"P",{"data-svelte-h":!0}),m(It)!=="svelte-1s077p3"&&(It.textContent=Us),fn=r(U),g(ce.$$.fragment,U),gn=r(U),Wt=l(U,"P",{"data-svelte-h":!0}),m(Wt)!=="svelte-1jfcabo"&&(Wt.innerHTML=Js),_n=r(U),g(pe.$$.fragment,U),vn=r(U),Zt=l(U,"P",{"data-svelte-h":!0}),m(Zt)!=="svelte-gxzj9w"&&(Zt.innerHTML=Is),bn=r(U),me=l(U,"DIV",{class:!0});var bt=j(me);g(Ke.$$.fragment,bt),kn=r(bt),Gt=l(bt,"P",{"data-svelte-h":!0}),m(Gt)!=="svelte-ycp5iu"&&(Gt.textContent=Ws),bt.forEach(i),U.forEach(i),Wo=r(e),g(et.$$.fragment,e),Zo=r(e),V=l(e,"DIV",{class:!0});var L=j(V);g(tt.$$.fragment,L),yn=r(L),Vt=l(L,"P",{"data-svelte-h":!0}),m(Vt)!=="svelte-gdlyqx"&&(Vt.textContent=Zs),Mn=r(L),Bt=l(L,"P",{"data-svelte-h":!0}),m(Bt)!=="svelte-q52n56"&&(Bt.innerHTML=Gs),Tn=r(L),Rt=l(L,"P",{"data-svelte-h":!0}),m(Rt)!=="svelte-hswkmf"&&(Rt.innerHTML=Vs),wn=r(L),D=l(L,"DIV",{class:!0});var oe=j(D);g(ot.$$.fragment,oe),$n=r(oe),St=l(oe,"P",{"data-svelte-h":!0}),m(St)!=="svelte-rq42hk"&&(St.innerHTML=Bs),xn=r(oe),g(he.$$.fragment,oe),oe.forEach(i),L.forEach(i),Go=r(e),g(nt.$$.fragment,e),Vo=r(e),B=l(e,"DIV",{class:!0});var X=j(B);g(st.$$.fragment,X),Cn=r(X),Ht=l(X,"P",{"data-svelte-h":!0}),m(Ht)!=="svelte-1677kv6"&&(Ht.textContent=Rs),Fn=r(X),Lt=l(X,"P",{"data-svelte-h":!0}),m(Lt)!=="svelte-q52n56"&&(Lt.innerHTML=Ss),zn=r(X),Xt=l(X,"P",{"data-svelte-h":!0}),m(Xt)!=="svelte-hswkmf"&&(Xt.innerHTML=Hs),qn=r(X),F=l(X,"DIV",{class:!0});var J=j(F);g(at.$$.fragment,J),jn=r(J),Qt=l(J,"P",{"data-svelte-h":!0}),m(Qt)!=="svelte-werdrg"&&(Qt.innerHTML=Ls),Un=r(J),g(ue.$$.fragment,J),Jn=r(J),Nt=l(J,"P",{"data-svelte-h":!0}),m(Nt)!=="svelte-a8wege"&&(Nt.textContent=Xs),In=r(J),Pt=l(J,"P",{"data-svelte-h":!0}),m(Pt)!=="svelte-qrgbe4"&&(Pt.textContent=Qs),Wn=r(J),g(fe.$$.fragment,J),Zn=r(J),Et=l(J,"P",{"data-svelte-h":!0}),m(Et)!=="svelte-1t77lwe"&&(Et.textContent=Ns),Gn=r(J),g(ge.$$.fragment,J),J.forEach(i),X.forEach(i),Bo=r(e),g(rt.$$.fragment,e),Ro=r(e),R=l(e,"DIV",{class:!0});var Q=j(R);g(it.$$.fragment,Q),Vn=r(Q),At=l(Q,"P",{"data-svelte-h":!0}),m(At)!=="svelte-cuuv55"&&(At.textContent=Ps),Bn=r(Q),Ot=l(Q,"P",{"data-svelte-h":!0}),m(Ot)!=="svelte-q52n56"&&(Ot.innerHTML=Es),Rn=r(Q),Yt=l(Q,"P",{"data-svelte-h":!0}),m(Yt)!=="svelte-hswkmf"&&(Yt.innerHTML=As),Sn=r(Q),z=l(Q,"DIV",{class:!0});var I=j(z);g(dt.$$.fragment,I),Hn=r(I),Dt=l(I,"P",{"data-svelte-h":!0}),m(Dt)!=="svelte-omq4au"&&(Dt.innerHTML=Os),Ln=r(I),g(_e.$$.fragment,I),Xn=r(I),Kt=l(I,"P",{"data-svelte-h":!0}),m(Kt)!=="svelte-ykxpe4"&&(Kt.textContent=Ys),Qn=r(I),eo=l(I,"P",{"data-svelte-h":!0}),m(eo)!=="svelte-eulckq"&&(eo.innerHTML=Ds),Nn=r(I),g(ve.$$.fragment,I),Pn=r(I),to=l(I,"P",{"data-svelte-h":!0}),m(to)!=="svelte-1t77lwe"&&(to.textContent=Ks),En=r(I),g(be.$$.fragment,I),I.forEach(i),Q.forEach(i),So=r(e),g(lt.$$.fragment,e),Ho=r(e),S=l(e,"DIV",{class:!0});var N=j(S);g(ct.$$.fragment,N),An=r(N),oo=l(N,"P",{"data-svelte-h":!0}),m(oo)!=="svelte-wx4bbk"&&(oo.innerHTML=ea),On=r(N),no=l(N,"P",{"data-svelte-h":!0}),m(no)!=="svelte-q52n56"&&(no.innerHTML=ta),Yn=r(N),so=l(N,"P",{"data-svelte-h":!0}),m(so)!=="svelte-hswkmf"&&(so.innerHTML=oa),Dn=r(N),q=l(N,"DIV",{class:!0});var W=j(q);g(pt.$$.fragment,W),Kn=r(W),ao=l(W,"P",{"data-svelte-h":!0}),m(ao)!=="svelte-ov8958"&&(ao.innerHTML=na),es=r(W),g(ke.$$.fragment,W),ts=r(W),ro=l(W,"P",{"data-svelte-h":!0}),m(ro)!=="svelte-11lpom8"&&(ro.textContent=sa),os=r(W),io=l(W,"P",{"data-svelte-h":!0}),m(io)!=="svelte-heukip"&&(io.innerHTML=aa),ns=r(W),g(ye.$$.fragment,W),ss=r(W),lo=l(W,"P",{"data-svelte-h":!0}),m(lo)!=="svelte-1t77lwe"&&(lo.textContent=ra),as=r(W),g(Me.$$.fragment,W),W.forEach(i),N.forEach(i),Lo=r(e),g(mt.$$.fragment,e),Xo=r(e),ee=l(e,"DIV",{class:!0});var kt=j(ee);g(ht.$$.fragment,kt),rs=r(kt),A=l(kt,"DIV",{class:!0});var $e=j(A);g(ut.$$.fragment,$e),is=r($e),co=l($e,"P",{"data-svelte-h":!0}),m(co)!=="svelte-qbbo68"&&(co.innerHTML=ia),ds=r($e),g(Te.$$.fragment,$e),ls=r($e),g(we.$$.fragment,$e),$e.forEach(i),kt.forEach(i),Qo=r(e),g(ft.$$.fragment,e),No=r(e),po=l(e,"P",{}),j(po).forEach(i),this.h()},h(){x(t,"name","hf:doc:metadata"),x(t,"content",Ua),x(ne,"class","flex flex-wrap space-x-1"),x(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){n(document.head,t),c(e,p,s),c(e,o,s),c(e,h,s),c(e,M,s),c(e,C,s),_(xe,e,s),c(e,mo,s),c(e,ne,s),c(e,ho,s),_(Ce,e,s),c(e,uo,s),c(e,Fe,s),c(e,fo,s),c(e,ze,s),c(e,go,s),c(e,qe,s),c(e,_o,s),c(e,je,s),c(e,vo,s),_(Ue,e,s),c(e,bo,s),c(e,Je,s),c(e,ko,s),_(Ie,e,s),c(e,yo,s),c(e,We,s),c(e,Mo,s),_(Ze,e,s),c(e,To,s),c(e,Ge,s),c(e,wo,s),_(Ve,e,s),c(e,$o,s),c(e,Be,s),c(e,xo,s),_(Re,e,s),c(e,Co,s),_(Se,e,s),c(e,Fo,s),c(e,He,s),c(e,zo,s),_(Le,e,s),c(e,qo,s),c(e,H,s),_(Xe,H,null),n(H,Eo),n(H,Mt),n(H,Ao),n(H,Tt),n(H,Oo),_(se,H,null),c(e,jo,s),_(Qe,e,s),c(e,Uo,s),c(e,T,s),_(Ne,T,null),n(T,Yo),n(T,wt),n(T,Do),n(T,$t),n(T,Ko),_(ae,T,null),n(T,en),n(T,xt),n(T,tn),_(re,T,null),n(T,on),n(T,Ct),n(T,nn),n(T,Y),_(Pe,Y,null),n(Y,sn),n(Y,Ft),n(Y,an),n(Y,zt),n(T,rn),n(T,ie),_(Ee,ie,null),n(ie,dn),n(ie,qt),n(T,ln),n(T,de),_(Ae,de,null),n(de,cn),n(de,jt),n(T,pn),n(T,le),_(Oe,le,null),n(le,mn),n(le,Ut),c(e,Jo,s),_(Ye,e,s),c(e,Io,s),c(e,$,s),_(De,$,null),n($,hn),n($,Jt),n($,un),n($,It),n($,fn),_(ce,$,null),n($,gn),n($,Wt),n($,_n),_(pe,$,null),n($,vn),n($,Zt),n($,bn),n($,me),_(Ke,me,null),n(me,kn),n(me,Gt),c(e,Wo,s),_(et,e,s),c(e,Zo,s),c(e,V,s),_(tt,V,null),n(V,yn),n(V,Vt),n(V,Mn),n(V,Bt),n(V,Tn),n(V,Rt),n(V,wn),n(V,D),_(ot,D,null),n(D,$n),n(D,St),n(D,xn),_(he,D,null),c(e,Go,s),_(nt,e,s),c(e,Vo,s),c(e,B,s),_(st,B,null),n(B,Cn),n(B,Ht),n(B,Fn),n(B,Lt),n(B,zn),n(B,Xt),n(B,qn),n(B,F),_(at,F,null),n(F,jn),n(F,Qt),n(F,Un),_(ue,F,null),n(F,Jn),n(F,Nt),n(F,In),n(F,Pt),n(F,Wn),_(fe,F,null),n(F,Zn),n(F,Et),n(F,Gn),_(ge,F,null),c(e,Bo,s),_(rt,e,s),c(e,Ro,s),c(e,R,s),_(it,R,null),n(R,Vn),n(R,At),n(R,Bn),n(R,Ot),n(R,Rn),n(R,Yt),n(R,Sn),n(R,z),_(dt,z,null),n(z,Hn),n(z,Dt),n(z,Ln),_(_e,z,null),n(z,Xn),n(z,Kt),n(z,Qn),n(z,eo),n(z,Nn),_(ve,z,null),n(z,Pn),n(z,to),n(z,En),_(be,z,null),c(e,So,s),_(lt,e,s),c(e,Ho,s),c(e,S,s),_(ct,S,null),n(S,An),n(S,oo),n(S,On),n(S,no),n(S,Yn),n(S,so),n(S,Dn),n(S,q),_(pt,q,null),n(q,Kn),n(q,ao),n(q,es),_(ke,q,null),n(q,ts),n(q,ro),n(q,os),n(q,io),n(q,ns),_(ye,q,null),n(q,ss),n(q,lo),n(q,as),_(Me,q,null),c(e,Lo,s),_(mt,e,s),c(e,Xo,s),c(e,ee,s),_(ht,ee,null),n(ee,rs),n(ee,A),_(ut,A,null),n(A,is),n(A,co),n(A,ds),_(Te,A,null),n(A,ls),_(we,A,null),c(e,Qo,s),_(ft,e,s),c(e,No,s),c(e,po,s),Po=!0},p(e,[s]){const O={};s&2&&(O.$$scope={dirty:s,ctx:e}),se.$set(O);const w={};s&2&&(w.$$scope={dirty:s,ctx:e}),ae.$set(w);const te={};s&2&&(te.$$scope={dirty:s,ctx:e}),re.$set(te);const gt={};s&2&&(gt.$$scope={dirty:s,ctx:e}),ce.$set(gt);const _t={};s&2&&(_t.$$scope={dirty:s,ctx:e}),pe.$set(_t);const vt={};s&2&&(vt.$$scope={dirty:s,ctx:e}),he.$set(vt);const U={};s&2&&(U.$$scope={dirty:s,ctx:e}),ue.$set(U);const bt={};s&2&&(bt.$$scope={dirty:s,ctx:e}),fe.$set(bt);const L={};s&2&&(L.$$scope={dirty:s,ctx:e}),ge.$set(L);const oe={};s&2&&(oe.$$scope={dirty:s,ctx:e}),_e.$set(oe);const X={};s&2&&(X.$$scope={dirty:s,ctx:e}),ve.$set(X);const J={};s&2&&(J.$$scope={dirty:s,ctx:e}),be.$set(J);const Q={};s&2&&(Q.$$scope={dirty:s,ctx:e}),ke.$set(Q);const I={};s&2&&(I.$$scope={dirty:s,ctx:e}),ye.$set(I);const N={};s&2&&(N.$$scope={dirty:s,ctx:e}),Me.$set(N);const W={};s&2&&(W.$$scope={dirty:s,ctx:e}),Te.$set(W);const kt={};s&2&&(kt.$$scope={dirty:s,ctx:e}),we.$set(kt)},i(e){Po||(v(xe.$$.fragment,e),v(Ce.$$.fragment,e),v(Ue.$$.fragment,e),v(Ie.$$.fragment,e),v(Ze.$$.fragment,e),v(Ve.$$.fragment,e),v(Re.$$.fragment,e),v(Se.$$.fragment,e),v(Le.$$.fragment,e),v(Xe.$$.fragment,e),v(se.$$.fragment,e),v(Qe.$$.fragment,e),v(Ne.$$.fragment,e),v(ae.$$.fragment,e),v(re.$$.fragment,e),v(Pe.$$.fragment,e),v(Ee.$$.fragment,e),v(Ae.$$.fragment,e),v(Oe.$$.fragment,e),v(Ye.$$.fragment,e),v(De.$$.fragment,e),v(ce.$$.fragment,e),v(pe.$$.fragment,e),v(Ke.$$.fragment,e),v(et.$$.fragment,e),v(tt.$$.fragment,e),v(ot.$$.fragment,e),v(he.$$.fragment,e),v(nt.$$.fragment,e),v(st.$$.fragment,e),v(at.$$.fragment,e),v(ue.$$.fragment,e),v(fe.$$.fragment,e),v(ge.$$.fragment,e),v(rt.$$.fragment,e),v(it.$$.fragment,e),v(dt.$$.fragment,e),v(_e.$$.fragment,e),v(ve.$$.fragment,e),v(be.$$.fragment,e),v(lt.$$.fragment,e),v(ct.$$.fragment,e),v(pt.$$.fragment,e),v(ke.$$.fragment,e),v(ye.$$.fragment,e),v(Me.$$.fragment,e),v(mt.$$.fragment,e),v(ht.$$.fragment,e),v(ut.$$.fragment,e),v(Te.$$.fragment,e),v(we.$$.fragment,e),v(ft.$$.fragment,e),Po=!0)},o(e){b(xe.$$.fragment,e),b(Ce.$$.fragment,e),b(Ue.$$.fragment,e),b(Ie.$$.fragment,e),b(Ze.$$.fragment,e),b(Ve.$$.fragment,e),b(Re.$$.fragment,e),b(Se.$$.fragment,e),b(Le.$$.fragment,e),b(Xe.$$.fragment,e),b(se.$$.fragment,e),b(Qe.$$.fragment,e),b(Ne.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(Pe.$$.fragment,e),b(Ee.$$.fragment,e),b(Ae.$$.fragment,e),b(Oe.$$.fragment,e),b(Ye.$$.fragment,e),b(De.$$.fragment,e),b(ce.$$.fragment,e),b(pe.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(ot.$$.fragment,e),b(he.$$.fragment,e),b(nt.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b(ue.$$.fragment,e),b(fe.$$.fragment,e),b(ge.$$.fragment,e),b(rt.$$.fragment,e),b(it.$$.fragment,e),b(dt.$$.fragment,e),b(_e.$$.fragment,e),b(ve.$$.fragment,e),b(be.$$.fragment,e),b(lt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(ke.$$.fragment,e),b(ye.$$.fragment,e),b(Me.$$.fragment,e),b(mt.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(Te.$$.fragment,e),b(we.$$.fragment,e),b(ft.$$.fragment,e),Po=!1},d(e){e&&(i(p),i(o),i(h),i(M),i(C),i(mo),i(ne),i(ho),i(uo),i(Fe),i(fo),i(ze),i(go),i(qe),i(_o),i(je),i(vo),i(bo),i(Je),i(ko),i(yo),i(We),i(Mo),i(To),i(Ge),i(wo),i($o),i(Be),i(xo),i(Co),i(Fo),i(He),i(zo),i(qo),i(H),i(jo),i(Uo),i(T),i(Jo),i(Io),i($),i(Wo),i(Zo),i(V),i(Go),i(Vo),i(B),i(Bo),i(Ro),i(R),i(So),i(Ho),i(S),i(Lo),i(Xo),i(ee),i(Qo),i(No),i(po)),i(t),k(xe,e),k(Ce,e),k(Ue,e),k(Ie,e),k(Ze,e),k(Ve,e),k(Re,e),k(Se,e),k(Le,e),k(Xe),k(se),k(Qe,e),k(Ne),k(ae),k(re),k(Pe),k(Ee),k(Ae),k(Oe),k(Ye,e),k(De),k(ce),k(pe),k(Ke),k(et,e),k(tt),k(ot),k(he),k(nt,e),k(st),k(at),k(ue),k(fe),k(ge),k(rt,e),k(it),k(dt),k(_e),k(ve),k(be),k(lt,e),k(ct),k(pt),k(ke),k(ye),k(Me),k(mt,e),k(ht),k(ut),k(Te),k(we),k(ft,e)}}}const Ua='{"title":"MVP","local":"mvp","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Usage examples","local":"usage-examples","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"MvpConfig","local":"transformers.MvpConfig","sections":[],"depth":2},{"title":"MvpTokenizer","local":"transformers.MvpTokenizer","sections":[],"depth":2},{"title":"MvpTokenizerFast","local":"transformers.MvpTokenizerFast","sections":[],"depth":2},{"title":"MvpModel","local":"transformers.MvpModel","sections":[],"depth":2},{"title":"MvpForConditionalGeneration","local":"transformers.MvpForConditionalGeneration","sections":[],"depth":2},{"title":"MvpForSequenceClassification","local":"transformers.MvpForSequenceClassification","sections":[],"depth":2},{"title":"MvpForQuestionAnswering","local":"transformers.MvpForQuestionAnswering","sections":[],"depth":2},{"title":"MvpForCausalLM","local":"transformers.MvpForCausalLM","sections":[],"depth":2}],"depth":1}';function Ja(y){return la(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Sa extends ca{constructor(t){super(),pa(this,t,Ja,ja,da,{})}}export{Sa as component};
