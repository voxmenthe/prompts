import{s as or,o as sr,n as C}from"../chunks/scheduler.18a86fab.js";import{S as ar,i as rr,g as c,s as a,r as f,A as ir,h as d,f as l,c as r,j as v,x as u,u as g,k as $,l as lr,y as n,a as m,v as _,d as b,t as T,w as y}from"../chunks/index.98837b22.js";import{T as ye}from"../chunks/Tip.77304350.js";import{D as E}from"../chunks/Docstring.a1ef7999.js";import{C as Y}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Me}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as L,E as cr}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as dr,a as Ks}from"../chunks/HfOption.6641485e.js";function pr(w){let t,h="Click on the right sidebar for more examples of how to use ELECTRA for different language tasks like sequence classification, token classification, and question answering.";return{c(){t=c("p"),t.textContent=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-1xbd85q"&&(t.textContent=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function mr(w){let t,h;return t=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFjbGFzc2lmaWVyJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMnRleHQtY2xhc3NpZmljYXRpb24lMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmJoYWRyZXNoLXNhdmFuaSUyRmVsZWN0cmEtYmFzZS1lbW90aW9uJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUwQSklMEFjbGFzc2lmaWVyKCUyMlRoaXMlMjByZXN0YXVyYW50JTIwaGFzJTIwYW1hemluZyUyMGZvb2QhJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

classifier = pipeline(
    task=<span class="hljs-string">&quot;text-classification&quot;</span>,
    model=<span class="hljs-string">&quot;bhadresh-savani/electra-base-emotion&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
classifier(<span class="hljs-string">&quot;This restaurant has amazing food!&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,p){_(t,o,p),h=!0},p:C,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){T(t.$$.fragment,o),h=!1},d(o){y(t,o)}}}function hr(w){let t,h;return t=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJiaGFkcmVzaC1zYXZhbmklMkZlbGVjdHJhLWJhc2UtZW1vdGlvbiUyMiUyQyUwQSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmJoYWRyZXNoLXNhdmFuaSUyRmVsZWN0cmEtYmFzZS1lbW90aW9uJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTBBKSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJFTEVDVFJBJTIwaXMlMjBtb3JlJTIwZWZmaWNpZW50JTIwdGhhbiUyMEJFUlQlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEElMjAlMjAlMjAlMjBwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KGRpbSUzRC0xKS5pdGVtKCklMEElMjAlMjAlMjAlMjBwcmVkaWN0ZWRfbGFiZWwlMjAlM0QlMjBtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEFwcmludChmJTIyUHJlZGljdGVkJTIwbGFiZWwlM0ElMjAlN0JwcmVkaWN0ZWRfbGFiZWwlN0QlMjIp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;bhadresh-savani/electra-base-emotion&quot;</span>,
)
model = AutoModelForSequenceClassification.from_pretrained(
    <span class="hljs-string">&quot;bhadresh-savani/electra-base-emotion&quot;</span>,
    dtype=torch.float16
)
inputs = tokenizer(<span class="hljs-string">&quot;ELECTRA is more efficient than BERT&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=-<span class="hljs-number">1</span>).item()
    predicted_label = model.config.id2label[predicted_class_id]
<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Predicted label: <span class="hljs-subst">{predicted_label}</span>&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,p){_(t,o,p),h=!0},p:C,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){T(t.$$.fragment,o),h=!1},d(o){y(t,o)}}}function ur(w){let t,h;return t=new Y({props:{code:"ZWNobyUyMC1lJTIwJTIyVGhpcyUyMHJlc3RhdXJhbnQlMjBoYXMlMjBhbWF6aW5nJTIwZm9vZC4lMjIlMjAlN0MlMjB0cmFuc2Zvcm1lcnMlMjBydW4lMjAtLXRhc2slMjB0ZXh0LWNsYXNzaWZpY2F0aW9uJTIwLS1tb2RlbCUyMGJoYWRyZXNoLXNhdmFuaSUyRmVsZWN0cmEtYmFzZS1lbW90aW9uJTIwLS1kZXZpY2UlMjAw",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;This restaurant has amazing food.&quot;</span> | transformers run --task text-classification --model bhadresh-savani/electra-base-emotion --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,p){_(t,o,p),h=!0},p:C,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){T(t.$$.fragment,o),h=!1},d(o){y(t,o)}}}function fr(w){let t,h,o,p,k,s;return t=new Ks({props:{id:"usage",option:"Pipeline",$$slots:{default:[mr]},$$scope:{ctx:w}}}),o=new Ks({props:{id:"usage",option:"AutoModel",$$slots:{default:[hr]},$$scope:{ctx:w}}}),k=new Ks({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ur]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),h=a(),f(o.$$.fragment),p=a(),f(k.$$.fragment)},l(M){g(t.$$.fragment,M),h=r(M),g(o.$$.fragment,M),p=r(M),g(k.$$.fragment,M)},m(M,j){_(t,M,j),m(M,h,j),_(o,M,j),m(M,p,j),_(k,M,j),s=!0},p(M,j){const Gn={};j&2&&(Gn.$$scope={dirty:j,ctx:M}),t.$set(Gn);const He={};j&2&&(He.$$scope={dirty:j,ctx:M}),o.$set(He);const ce={};j&2&&(ce.$$scope={dirty:j,ctx:M}),k.$set(ce)},i(M){s||(b(t.$$.fragment,M),b(o.$$.fragment,M),b(k.$$.fragment,M),s=!0)},o(M){T(t.$$.fragment,M),T(o.$$.fragment,M),T(k.$$.fragment,M),s=!1},d(M){M&&(l(h),l(p)),y(t,M),y(o,M),y(k,M)}}}function gr(w){let t,h="Examples:",o,p,k;return p=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEVsZWN0cmFDb25maWclMkMlMjBFbGVjdHJhTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwRUxFQ1RSQSUyMGVsZWN0cmEtYmFzZS11bmNhc2VkJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEVsZWN0cmFDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwZWxlY3RyYS1iYXNlLXVuY2FzZWQlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEVsZWN0cmFNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ElectraConfig, ElectraModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a ELECTRA electra-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = ElectraConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the electra-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function _r(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function br(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function Tr(w){let t,h="Examples:",o,p,k;return p=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEVsZWN0cmFGb3JQcmVUcmFpbmluZyUyQyUyMEF1dG9Ub2tlbml6ZXIlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQWRpc2NyaW1pbmF0b3IlMjAlM0QlMjBFbGVjdHJhRm9yUHJlVHJhaW5pbmcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtYmFzZS1kaXNjcmltaW5hdG9yJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtYmFzZS1kaXNjcmltaW5hdG9yJTIyKSUwQSUwQXNlbnRlbmNlJTIwJTNEJTIwJTIyVGhlJTIwcXVpY2slMjBicm93biUyMGZveCUyMGp1bXBzJTIwb3ZlciUyMHRoZSUyMGxhenklMjBkb2clMjIlMEFmYWtlX3NlbnRlbmNlJTIwJTNEJTIwJTIyVGhlJTIwcXVpY2slMjBicm93biUyMGZveCUyMGZha2UlMjBvdmVyJTIwdGhlJTIwbGF6eSUyMGRvZyUyMiUwQSUwQWZha2VfdG9rZW5zJTIwJTNEJTIwdG9rZW5pemVyLnRva2VuaXplKGZha2Vfc2VudGVuY2UlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQWZha2VfaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyLmVuY29kZShmYWtlX3NlbnRlbmNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFkaXNjcmltaW5hdG9yX291dHB1dHMlMjAlM0QlMjBkaXNjcmltaW5hdG9yKGZha2VfaW5wdXRzKSUwQXByZWRpY3Rpb25zJTIwJTNEJTIwdG9yY2gucm91bmQoKHRvcmNoLnNpZ24oZGlzY3JpbWluYXRvcl9vdXRwdXRzJTVCMCU1RCklMjAlMkIlMjAxKSUyMCUyRiUyMDIpJTBBJTBBZmFrZV90b2tlbnMlMEElMEFwcmVkaWN0aW9ucy5zcXVlZXplKCkudG9saXN0KCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ElectraForPreTraining, AutoTokenizer
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>discriminator = ElectraForPreTraining.from_pretrained(<span class="hljs-string">&quot;google/electra-base-discriminator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/electra-base-discriminator&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>sentence = <span class="hljs-string">&quot;The quick brown fox jumps over the lazy dog&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>fake_sentence = <span class="hljs-string">&quot;The quick brown fox fake over the lazy dog&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>fake_tokens = tokenizer.tokenize(fake_sentence, add_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>fake_inputs = tokenizer.encode(fake_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>discriminator_outputs = discriminator(fake_inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>predictions = torch.<span class="hljs-built_in">round</span>((torch.sign(discriminator_outputs[<span class="hljs-number">0</span>]) + <span class="hljs-number">1</span>) / <span class="hljs-number">2</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>fake_tokens
[<span class="hljs-string">&#x27;[CLS]&#x27;</span>, <span class="hljs-string">&#x27;the&#x27;</span>, <span class="hljs-string">&#x27;quick&#x27;</span>, <span class="hljs-string">&#x27;brown&#x27;</span>, <span class="hljs-string">&#x27;fox&#x27;</span>, <span class="hljs-string">&#x27;fake&#x27;</span>, <span class="hljs-string">&#x27;over&#x27;</span>, <span class="hljs-string">&#x27;the&#x27;</span>, <span class="hljs-string">&#x27;lazy&#x27;</span>, <span class="hljs-string">&#x27;dog&#x27;</span>, <span class="hljs-string">&#x27;[SEP]&#x27;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predictions.squeeze().tolist()
[<span class="hljs-number">0.0</span>, <span class="hljs-number">0.0</span>, <span class="hljs-number">0.0</span>, <span class="hljs-number">0.0</span>, <span class="hljs-number">0.0</span>, <span class="hljs-number">1.0</span>, <span class="hljs-number">0.0</span>, <span class="hljs-number">0.0</span>, <span class="hljs-number">0.0</span>, <span class="hljs-number">0.0</span>, <span class="hljs-number">0.0</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function yr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function Mr(w){let t,h="Example:",o,p,k;return p=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFbGVjdHJhRm9yQ2F1c2FsTE0lMkMlMjBFbGVjdHJhQ29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZlbGVjdHJhLWJhc2UtZ2VuZXJhdG9yJTIyKSUwQWNvbmZpZyUyMCUzRCUyMEVsZWN0cmFDb25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtYmFzZS1nZW5lcmF0b3IlMjIpJTBBY29uZmlnLmlzX2RlY29kZXIlMjAlM0QlMjBUcnVlJTBBbW9kZWwlMjAlM0QlMjBFbGVjdHJhRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtYmFzZS1nZW5lcmF0b3IlMjIlMkMlMjBjb25maWclM0Rjb25maWcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ElectraForCausalLM, ElectraConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/electra-base-generator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = ElectraConfig.from_pretrained(<span class="hljs-string">&quot;google/electra-base-generator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForCausalLM.from_pretrained(<span class="hljs-string">&quot;google/electra-base-generator&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function kr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function wr(w){let t,h="Example:",o,p,k;return p=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFbGVjdHJhRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtc21hbGwtZGlzY3JpbWluYXRvciUyMiklMEFtb2RlbCUyMCUzRCUyMEVsZWN0cmFGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZWxlY3RyYS1zbWFsbC1kaXNjcmltaW5hdG9yJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCUzQ21hc2slM0UuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCUzQ21hc2slM0UlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTIzJTIwbWFzayUyMGxhYmVscyUyMG9mJTIwbm9uLSUzQ21hc2slM0UlMjB0b2tlbnMlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQlMkMlMjBsYWJlbHMlMkMlMjAtMTAwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFyb3VuZChvdXRwdXRzLmxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ElectraForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForMaskedLM.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of &lt;mask&gt;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
...

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-&lt;mask&gt; tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(outputs.loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function vr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function $r(w){let t,h="Example of single-label classification:",o,p,k;return p=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEVsZWN0cmFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZWxlY3RyYS1zbWFsbC1kaXNjcmltaW5hdG9yJTIyKSUwQW1vZGVsJTIwJTNEJTIwRWxlY3RyYUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtc21hbGwtZGlzY3JpbWluYXRvciUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwRWxlY3RyYUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtc21hbGwtZGlzY3JpbWluYXRvciUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ElectraForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-ykxpe4"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function Er(w){let t,h="Example of multi-label classification:",o,p,k;return p=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEVsZWN0cmFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZWxlY3RyYS1zbWFsbC1kaXNjcmltaW5hdG9yJTIyKSUwQW1vZGVsJTIwJTNEJTIwRWxlY3RyYUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtc21hbGwtZGlzY3JpbWluYXRvciUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwRWxlY3RyYUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRmVsZWN0cmEtc21hbGwtZGlzY3JpbWluYXRvciUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ElectraForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-1l8e32d"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function jr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function Cr(w){let t,h="Example:",o,p,k;return p=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFbGVjdHJhRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtc21hbGwtZGlzY3JpbWluYXRvciUyMiklMEFtb2RlbCUyMCUzRCUyMEVsZWN0cmFGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZWxlY3RyYS1zbWFsbC1kaXNjcmltaW5hdG9yJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFjaG9pY2UwJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2l0aCUyMGElMjBmb3JrJTIwYW5kJTIwYSUyMGtuaWZlLiUyMiUwQWNob2ljZTElMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aGlsZSUyMGhlbGQlMjBpbiUyMHRoZSUyMGhhbmQuJTIyJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDApLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMGNob2ljZTAlMjBpcyUyMGNvcnJlY3QlMjAoYWNjb3JkaW5nJTIwdG8lMjBXaWtpcGVkaWElMjAlM0IpKSUyQyUyMGJhdGNoJTIwc2l6ZSUyMDElMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlMkMlMjBwcm9tcHQlNUQlMkMlMjAlNUJjaG9pY2UwJTJDJTIwY2hvaWNlMSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiolN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUyQyUyMGxhYmVscyUzRGxhYmVscyklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBpcyUyMDElMEElMEElMjMlMjB0aGUlMjBsaW5lYXIlMjBjbGFzc2lmaWVyJTIwc3RpbGwlMjBuZWVkcyUyMHRvJTIwYmUlMjB0cmFpbmVkJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ElectraForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function Jr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function xr(w){let t,h="Example:",o,p,k;return p=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFbGVjdHJhRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZWxlY3RyYS1zbWFsbC1kaXNjcmltaW5hdG9yJTIyKSUwQW1vZGVsJTIwJTNEJTIwRWxlY3RyYUZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtc21hbGwtZGlzY3JpbWluYXRvciUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ElectraForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function zr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=d(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,p){m(o,t,p)},p:C,d(o){o&&l(t)}}}function Ur(w){let t,h="Example:",o,p,k;return p=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFbGVjdHJhRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmVsZWN0cmEtc21hbGwtZGlzY3JpbWluYXRvciUyMiklMEFtb2RlbCUyMCUzRCUyMEVsZWN0cmFGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZWxlY3RyYS1zbWFsbC1kaXNjcmltaW5hdG9yJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ElectraForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ElectraForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google/electra-small-discriminator&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(p.$$.fragment)},l(s){t=d(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(s),g(p.$$.fragment,s)},m(s,M){m(s,t,M),m(s,o,M),_(p,s,M),k=!0},p:C,i(s){k||(b(p.$$.fragment,s),k=!0)},o(s){T(p.$$.fragment,s),k=!1},d(s){s&&(l(t),l(o)),y(p,s)}}}function Fr(w){let t,h,o,p,k,s="<em>This model was released on 2020-03-23 and added to Hugging Face Transformers on 2020-11-16.</em>",M,j,Gn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',He,ce,Ln,Pe,ea='<a href="https://huggingface.co/papers/2003.10555" rel="nofollow">ELECTRA</a> modifies the pretraining objective of traditional masked language models like BERT. Instead of just masking tokens and asking the model to predict them, ELECTRA trains two models, a generator and a discriminator. The generator replaces some tokens with plausible alternatives and the discriminator (the model you’ll actually use) learns to detect which tokens are original and which were replaced. This training approach is very efficient and scales to larger models while using considerably less compute.',Xn,Qe,ta="This approach is super efficient because ELECTRA learns from every single token in the input, not just the masked ones. That’s why even the small ELECTRA models can match or outperform much larger models while using way less computing resources.",Sn,Ae,na='You can find all the original ELECTRA checkpoints under the <a href="https://huggingface.co/collections/google/electra-release-64ff6e8b18830fabea30a1ab" rel="nofollow">ELECTRA</a> release.',Hn,ke,Pn,Ye,oa='The example below demonstrates how to classify text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',Qn,we,An,Oe,Yn,G,Gt,sa="<p>ELECTRA consists of two transformer models, a generator (G) and a discriminator (D). For most downstream tasks, use the discriminator model (as indicated by <code>*-discriminator</code> in the name) rather than the generator.</p>",$o,Bt,aa="<p>ELECTRA comes in three sizes: small (14M parameters), base (110M parameters), and large (335M parameters).</p>",Eo,Lt,ra="<p>ELECTRA can use a smaller embedding size than the hidden size for efficiency. When <code>embedding_size</code> is smaller than <code>hidden_size</code> in the configuration, a projection layer connects them.</p>",jo,De,Xt,ia="When using batched inputs with padding, make sure to use attention masks to prevent the model from attending to padding tokens.",Co,Ke,Jo,St,la='<p>When using the discriminator for a downstream task, you can load it into any of the ELECTRA model classes (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForSequenceClassification">ElectraForSequenceClassification</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForTokenClassification">ElectraForTokenClassification</a>, etc.).</p>',On,et,Dn,X,tt,xo,Ht,ca=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraModel">ElectraModel</a> or a <code>TFElectraModel</code>. It is
used to instantiate a ELECTRA model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the ELECTRA
<a href="https://huggingface.co/google/electra-small-discriminator" rel="nofollow">google/electra-small-discriminator</a> architecture.`,zo,Pt,da=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Uo,ve,Kn,nt,eo,J,ot,Fo,Qt,pa="Construct a Electra tokenizer. Based on WordPiece.",Io,At,ma=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,Wo,de,st,Zo,Yt,ha=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A Electra sequence has the following format:`,qo,Ot,ua="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Ro,$e,at,No,Dt,fa="Converts a sequence of tokens (string) in a single string.",Vo,Ee,rt,Go,Kt,ga=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,to,it,no,S,lt,Bo,en,_a="Construct a “fast” ELECTRA tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on WordPiece.",Lo,tn,ba=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Xo,pe,ct,So,nn,Ta=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A ELECTRA sequence has the following format:`,Ho,on,ya="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",oo,dt,so,ge,pt,Po,sn,Ma='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForPreTraining">ElectraForPreTraining</a>.',ao,mt,ro,F,ht,Qo,an,ka="The bare Electra Model outputting raw hidden-states without any specific head on top.",Ao,rn,wa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Yo,ln,va=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Oo,me,ut,Do,cn,$a='The <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraModel">ElectraModel</a> forward method, overrides the <code>__call__</code> special method.',Ko,je,io,ft,lo,x,gt,es,dn,Ea="Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.",ts,pn,ja="It is recommended to load the discriminator checkpoint into that model.",ns,mn,Ca=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,os,hn,Ja=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ss,O,_t,as,un,xa='The <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForPreTraining">ElectraForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',rs,Ce,is,Je,co,bt,po,I,Tt,ls,fn,za="ELECTRA Model with a <code>language modeling</code> head on top for CLM fine-tuning.",cs,gn,Ua=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ds,_n,Fa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ps,D,yt,ms,bn,Ia='The <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForCausalLM">ElectraForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',hs,xe,us,ze,mo,Mt,ho,z,kt,fs,Tn,Wa="Electra model with a language modeling head on top.",gs,yn,Za=`Even though both the discriminator and generator may be loaded into this model, the generator is the only model of
the two to have been trained for the masked language modeling task.`,_s,Mn,qa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,bs,kn,Ra=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ts,K,wt,ys,wn,Na='The <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForMaskedLM">ElectraForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Ms,Ue,ks,Fe,uo,vt,fo,W,$t,ws,vn,Va=`ELECTRA Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,vs,$n,Ga=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$s,En,Ba=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Es,B,Et,js,jn,La='The <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForSequenceClassification">ElectraForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Cs,Ie,Js,We,xs,Ze,go,jt,_o,Z,Ct,zs,Cn,Xa=`The Electra Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Us,Jn,Sa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Fs,xn,Ha=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Is,ee,Jt,Ws,zn,Pa='The <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForMultipleChoice">ElectraForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Zs,qe,qs,Re,bo,xt,To,U,zt,Rs,Un,Qa="Electra model with a token classification head on top.",Ns,Fn,Aa="Both the discriminator and generator may be loaded into this model.",Vs,In,Ya=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Gs,Wn,Oa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bs,te,Ut,Ls,Zn,Da='The <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForTokenClassification">ElectraForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Xs,Ne,Ss,Ve,yo,Ft,Mo,q,It,Hs,qn,Ka=`The Electra transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Ps,Rn,er=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qs,Nn,tr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,As,ne,Wt,Ys,Vn,nr='The <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForQuestionAnswering">ElectraForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Os,Ge,Ds,Be,ko,Zt,wo,Bn,vo;return ce=new L({props:{title:"ELECTRA",local:"electra",headingTag:"h1"}}),ke=new ye({props:{warning:!1,$$slots:{default:[pr]},$$scope:{ctx:w}}}),we=new dr({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[fr]},$$scope:{ctx:w}}}),Oe=new L({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Ke=new Y({props:{code:"JTIzJTIwRXhhbXBsZSUyMG9mJTIwcHJvcGVybHklMjBoYW5kbGluZyUyMHBhZGRpbmclMjB3aXRoJTIwYXR0ZW50aW9uJTIwbWFza3MlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTVCJTIyU2hvcnQlMjB0ZXh0JTIyJTJDJTIwJTIyVGhpcyUyMGlzJTIwYSUyMG11Y2glMjBsb25nZXIlMjB0ZXh0JTIwdGhhdCUyMG5lZWRzJTIwcGFkZGluZyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHBhZGRpbmclM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTIwJTIwJTIzJTIwYXV0b21hdGljYWxseSUyMHVzZXMlMjB0aGUlMjBhdHRlbnRpb25fbWFzaw==",highlighted:`<span class="hljs-comment"># Example of properly handling padding with attention masks</span>
inputs = tokenizer([<span class="hljs-string">&quot;Short text&quot;</span>, <span class="hljs-string">&quot;This is a much longer text that needs padding&quot;</span>],
                padding=<span class="hljs-literal">True</span>,
                return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
outputs = model(**inputs)  <span class="hljs-comment"># automatically uses the attention_mask</span>`,wrap:!1}}),et=new L({props:{title:"ElectraConfig",local:"transformers.ElectraConfig",headingTag:"h2"}}),tt=new E({props:{name:"class transformers.ElectraConfig",anchor:"transformers.ElectraConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"embedding_size",val:" = 128"},{name:"hidden_size",val:" = 256"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 4"},{name:"intermediate_size",val:" = 1024"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"summary_type",val:" = 'first'"},{name:"summary_use_proj",val:" = True"},{name:"summary_activation",val:" = 'gelu'"},{name:"summary_last_dropout",val:" = 0.1"},{name:"pad_token_id",val:" = 0"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ElectraConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraModel">ElectraModel</a> or <code>TFElectraModel</code>.`,name:"vocab_size"},{anchor:"transformers.ElectraConfig.embedding_size",description:`<strong>embedding_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"embedding_size"},{anchor:"transformers.ElectraConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.ElectraConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.ElectraConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.ElectraConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.ElectraConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.ElectraConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.ElectraConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.ElectraConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.ElectraConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraModel">ElectraModel</a> or <code>TFElectraModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.ElectraConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.ElectraConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.ElectraConfig.summary_type",description:`<strong>summary_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;first&quot;</code>) &#x2014;
Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.</p>
<p>Has to be one of the following options:</p>
<ul>
<li><code>&quot;last&quot;</code>: Take the last token hidden state (like XLNet).</li>
<li><code>&quot;first&quot;</code>: Take the first token hidden state (like BERT).</li>
<li><code>&quot;mean&quot;</code>: Take the mean of all tokens hidden states.</li>
<li><code>&quot;cls_index&quot;</code>: Supply a Tensor of classification token position (like GPT/GPT-2).</li>
<li><code>&quot;attn&quot;</code>: Not implemented now, use multi-head attention.</li>
</ul>`,name:"summary_type"},{anchor:"transformers.ElectraConfig.summary_use_proj",description:`<strong>summary_use_proj</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.</p>
<p>Whether or not to add a projection after the vector extraction.`,name:"summary_use_proj"},{anchor:"transformers.ElectraConfig.summary_activation",description:`<strong>summary_activation</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.</p>
<p>Pass <code>&quot;gelu&quot;</code> for a gelu activation to the output, any other value will result in no activation.`,name:"summary_activation"},{anchor:"transformers.ElectraConfig.summary_last_dropout",description:`<strong>summary_last_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.</p>
<p>The dropout ratio to be used after the projection and activation.`,name:"summary_last_dropout"},{anchor:"transformers.ElectraConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.ElectraConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.ElectraConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/configuration_electra.py#L29"}}),ve=new Me({props:{anchor:"transformers.ElectraConfig.example",$$slots:{default:[gr]},$$scope:{ctx:w}}}),nt=new L({props:{title:"ElectraTokenizer",local:"transformers.ElectraTokenizer",headingTag:"h2"}}),ot=new E({props:{name:"class transformers.ElectraTokenizer",anchor:"transformers.ElectraTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ElectraTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.ElectraTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.ElectraTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.ElectraTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.ElectraTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.ElectraTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.ElectraTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.ElectraTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.ElectraTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.ElectraTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.ElectraTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original Electra).`,name:"strip_accents"},{anchor:"transformers.ElectraTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/tokenization_electra.py#L53"}}),st=new E({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.ElectraTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.ElectraTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.ElectraTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/tokenization_electra.py#L188",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),at=new E({props:{name:"convert_tokens_to_string",anchor:"transformers.ElectraTokenizer.convert_tokens_to_string",parameters:[{name:"tokens",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/tokenization_electra.py#L183"}}),rt=new E({props:{name:"get_special_tokens_mask",anchor:"transformers.ElectraTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.ElectraTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.ElectraTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.ElectraTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/tokenization_electra.py#L213",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),it=new L({props:{title:"ElectraTokenizerFast",local:"transformers.ElectraTokenizerFast",headingTag:"h2"}}),lt=new E({props:{name:"class transformers.ElectraTokenizerFast",anchor:"transformers.ElectraTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ElectraTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.ElectraTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.ElectraTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.ElectraTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.ElectraTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.ElectraTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.ElectraTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.ElectraTokenizerFast.clean_text",description:`<strong>clean_text</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to clean the text before tokenization by removing any control characters and replacing all
whitespaces by the classic one.`,name:"clean_text"},{anchor:"transformers.ElectraTokenizerFast.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see <a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">this
issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.ElectraTokenizerFast.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original ELECTRA).`,name:"strip_accents"},{anchor:"transformers.ElectraTokenizerFast.wordpieces_prefix",description:`<strong>wordpieces_prefix</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;##&quot;</code>) &#x2014;
The prefix for subwords.`,name:"wordpieces_prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/tokenization_electra_fast.py#L29"}}),ct=new E({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.ElectraTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.ElectraTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.ElectraTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/tokenization_electra_fast.py#L114",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),dt=new L({props:{title:"Electra specific outputs",local:"transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput",headingTag:"h2"}}),pt=new E({props:{name:"class transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput",anchor:"transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput.loss",description:`<strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
Total loss of the ELECTRA objective.`,name:"loss"},{anchor:"transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Prediction scores of the head (scores for each token before SoftMax).`,name:"logits"},{anchor:"transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L667"}}),mt=new L({props:{title:"ElectraModel",local:"transformers.ElectraModel",headingTag:"h2"}}),ht=new E({props:{name:"class transformers.ElectraModel",anchor:"transformers.ElectraModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ElectraModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraModel">ElectraModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L682"}}),ut=new E({props:{name:"forward",anchor:"transformers.ElectraModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ElectraModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ElectraModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ElectraModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ElectraModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ElectraModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ElectraModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ElectraModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ElectraModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ElectraModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ElectraModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ElectraModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ElectraModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ElectraModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L709",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig"
>ElectraConfig</a>) and inputs.</p>
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
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> and <code>config.add_cross_attention=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),je=new ye({props:{$$slots:{default:[_r]},$$scope:{ctx:w}}}),ft=new L({props:{title:"ElectraForPreTraining",local:"transformers.ElectraForPreTraining",headingTag:"h2"}}),gt=new E({props:{name:"class transformers.ElectraForPreTraining",anchor:"transformers.ElectraForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ElectraForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForPreTraining">ElectraForPreTraining</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1024"}}),_t=new E({props:{name:"forward",anchor:"transformers.ElectraForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ElectraForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ElectraForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ElectraForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ElectraForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ElectraForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ElectraForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ElectraForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the ELECTRA loss. Input should be a sequence of tokens (see <code>input_ids</code> docstring)
Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates the token is an original token,</li>
<li>1 indicates the token was replaced.</li>
</ul>`,name:"labels"},{anchor:"transformers.ElectraForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ElectraForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ElectraForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1033",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput"
>transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig"
>ElectraConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) — Total loss of the ELECTRA objective.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) — Prediction scores of the head (scores for each token before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput"
>transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ce=new ye({props:{$$slots:{default:[br]},$$scope:{ctx:w}}}),Je=new Me({props:{anchor:"transformers.ElectraForPreTraining.forward.example",$$slots:{default:[Tr]},$$scope:{ctx:w}}}),bt=new L({props:{title:"ElectraForCausalLM",local:"transformers.ElectraForCausalLM",headingTag:"h2"}}),Tt=new E({props:{name:"class transformers.ElectraForCausalLM",anchor:"transformers.ElectraForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ElectraForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForCausalLM">ElectraForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1467"}}),yt=new E({props:{name:"forward",anchor:"transformers.ElectraForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.Tensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ElectraForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ElectraForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ElectraForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ElectraForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ElectraForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ElectraForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ElectraForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ElectraForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ElectraForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.ElectraForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.Tensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ElectraForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ElectraForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ElectraForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ElectraForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1488",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig"
>ElectraConfig</a>) and inputs.</p>
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
`}}),xe=new ye({props:{$$slots:{default:[yr]},$$scope:{ctx:w}}}),ze=new Me({props:{anchor:"transformers.ElectraForCausalLM.forward.example",$$slots:{default:[Mr]},$$scope:{ctx:w}}}),Mt=new L({props:{title:"ElectraForMaskedLM",local:"transformers.ElectraForMaskedLM",headingTag:"h2"}}),kt=new E({props:{name:"class transformers.ElectraForMaskedLM",anchor:"transformers.ElectraForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ElectraForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForMaskedLM">ElectraForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1126"}}),wt=new E({props:{name:"forward",anchor:"transformers.ElectraForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ElectraForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ElectraForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ElectraForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ElectraForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ElectraForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ElectraForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ElectraForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.ElectraForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ElectraForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ElectraForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1145",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig"
>ElectraConfig</a>) and inputs.</p>
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
`}}),Ue=new ye({props:{$$slots:{default:[kr]},$$scope:{ctx:w}}}),Fe=new Me({props:{anchor:"transformers.ElectraForMaskedLM.forward.example",$$slots:{default:[wr]},$$scope:{ctx:w}}}),vt=new L({props:{title:"ElectraForSequenceClassification",local:"transformers.ElectraForSequenceClassification",headingTag:"h2"}}),$t=new E({props:{name:"class transformers.ElectraForSequenceClassification",anchor:"transformers.ElectraForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ElectraForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForSequenceClassification">ElectraForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L934"}}),Et=new E({props:{name:"forward",anchor:"transformers.ElectraForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ElectraForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ElectraForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ElectraForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ElectraForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ElectraForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ElectraForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ElectraForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.ElectraForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ElectraForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ElectraForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L945",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig"
>ElectraConfig</a>) and inputs.</p>
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
`}}),Ie=new ye({props:{$$slots:{default:[vr]},$$scope:{ctx:w}}}),We=new Me({props:{anchor:"transformers.ElectraForSequenceClassification.forward.example",$$slots:{default:[$r]},$$scope:{ctx:w}}}),Ze=new Me({props:{anchor:"transformers.ElectraForSequenceClassification.forward.example-2",$$slots:{default:[Er]},$$scope:{ctx:w}}}),jt=new L({props:{title:"ElectraForMultipleChoice",local:"transformers.ElectraForMultipleChoice",headingTag:"h2"}}),Ct=new E({props:{name:"class transformers.ElectraForMultipleChoice",anchor:"transformers.ElectraForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ElectraForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForMultipleChoice">ElectraForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1359"}}),Jt=new E({props:{name:"forward",anchor:"transformers.ElectraForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ElectraForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ElectraForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ElectraForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ElectraForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ElectraForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ElectraForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ElectraForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.ElectraForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ElectraForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ElectraForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1370",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig"
>ElectraConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) — <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),qe=new ye({props:{$$slots:{default:[jr]},$$scope:{ctx:w}}}),Re=new Me({props:{anchor:"transformers.ElectraForMultipleChoice.forward.example",$$slots:{default:[Cr]},$$scope:{ctx:w}}}),xt=new L({props:{title:"ElectraForTokenClassification",local:"transformers.ElectraForTokenClassification",headingTag:"h2"}}),zt=new E({props:{name:"class transformers.ElectraForTokenClassification",anchor:"transformers.ElectraForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ElectraForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForTokenClassification">ElectraForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1208"}}),Ut=new E({props:{name:"forward",anchor:"transformers.ElectraForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ElectraForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ElectraForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ElectraForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ElectraForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ElectraForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ElectraForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ElectraForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.ElectraForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ElectraForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ElectraForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1222",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig"
>ElectraConfig</a>) and inputs.</p>
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
`}}),Ne=new ye({props:{$$slots:{default:[Jr]},$$scope:{ctx:w}}}),Ve=new Me({props:{anchor:"transformers.ElectraForTokenClassification.forward.example",$$slots:{default:[xr]},$$scope:{ctx:w}}}),Ft=new L({props:{title:"ElectraForQuestionAnswering",local:"transformers.ElectraForQuestionAnswering",headingTag:"h2"}}),It=new E({props:{name:"class transformers.ElectraForQuestionAnswering",anchor:"transformers.ElectraForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ElectraForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraForQuestionAnswering">ElectraForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1276"}}),Wt=new E({props:{name:"forward",anchor:"transformers.ElectraForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ElectraForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ElectraForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ElectraForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ElectraForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ElectraForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ElectraForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ElectraForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.ElectraForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.ElectraForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ElectraForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ElectraForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/electra/modeling_electra.py#L1290",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/electra#transformers.ElectraConfig"
>ElectraConfig</a>) and inputs.</p>
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
`}}),Ge=new ye({props:{$$slots:{default:[zr]},$$scope:{ctx:w}}}),Be=new Me({props:{anchor:"transformers.ElectraForQuestionAnswering.forward.example",$$slots:{default:[Ur]},$$scope:{ctx:w}}}),Zt=new cr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/electra.md"}}),{c(){t=c("meta"),h=a(),o=c("p"),p=a(),k=c("p"),k.innerHTML=s,M=a(),j=c("div"),j.innerHTML=Gn,He=a(),f(ce.$$.fragment),Ln=a(),Pe=c("p"),Pe.innerHTML=ea,Xn=a(),Qe=c("p"),Qe.textContent=ta,Sn=a(),Ae=c("p"),Ae.innerHTML=na,Hn=a(),f(ke.$$.fragment),Pn=a(),Ye=c("p"),Ye.innerHTML=oa,Qn=a(),f(we.$$.fragment),An=a(),f(Oe.$$.fragment),Yn=a(),G=c("ul"),Gt=c("li"),Gt.innerHTML=sa,$o=a(),Bt=c("li"),Bt.innerHTML=aa,Eo=a(),Lt=c("li"),Lt.innerHTML=ra,jo=a(),De=c("li"),Xt=c("p"),Xt.textContent=ia,Co=a(),f(Ke.$$.fragment),Jo=a(),St=c("li"),St.innerHTML=la,On=a(),f(et.$$.fragment),Dn=a(),X=c("div"),f(tt.$$.fragment),xo=a(),Ht=c("p"),Ht.innerHTML=ca,zo=a(),Pt=c("p"),Pt.innerHTML=da,Uo=a(),f(ve.$$.fragment),Kn=a(),f(nt.$$.fragment),eo=a(),J=c("div"),f(ot.$$.fragment),Fo=a(),Qt=c("p"),Qt.textContent=pa,Io=a(),At=c("p"),At.innerHTML=ma,Wo=a(),de=c("div"),f(st.$$.fragment),Zo=a(),Yt=c("p"),Yt.textContent=ha,qo=a(),Ot=c("ul"),Ot.innerHTML=ua,Ro=a(),$e=c("div"),f(at.$$.fragment),No=a(),Dt=c("p"),Dt.textContent=fa,Vo=a(),Ee=c("div"),f(rt.$$.fragment),Go=a(),Kt=c("p"),Kt.innerHTML=ga,to=a(),f(it.$$.fragment),no=a(),S=c("div"),f(lt.$$.fragment),Bo=a(),en=c("p"),en.innerHTML=_a,Lo=a(),tn=c("p"),tn.innerHTML=ba,Xo=a(),pe=c("div"),f(ct.$$.fragment),So=a(),nn=c("p"),nn.textContent=Ta,Ho=a(),on=c("ul"),on.innerHTML=ya,oo=a(),f(dt.$$.fragment),so=a(),ge=c("div"),f(pt.$$.fragment),Po=a(),sn=c("p"),sn.innerHTML=Ma,ao=a(),f(mt.$$.fragment),ro=a(),F=c("div"),f(ht.$$.fragment),Qo=a(),an=c("p"),an.textContent=ka,Ao=a(),rn=c("p"),rn.innerHTML=wa,Yo=a(),ln=c("p"),ln.innerHTML=va,Oo=a(),me=c("div"),f(ut.$$.fragment),Do=a(),cn=c("p"),cn.innerHTML=$a,Ko=a(),f(je.$$.fragment),io=a(),f(ft.$$.fragment),lo=a(),x=c("div"),f(gt.$$.fragment),es=a(),dn=c("p"),dn.textContent=Ea,ts=a(),pn=c("p"),pn.textContent=ja,ns=a(),mn=c("p"),mn.innerHTML=Ca,os=a(),hn=c("p"),hn.innerHTML=Ja,ss=a(),O=c("div"),f(_t.$$.fragment),as=a(),un=c("p"),un.innerHTML=xa,rs=a(),f(Ce.$$.fragment),is=a(),f(Je.$$.fragment),co=a(),f(bt.$$.fragment),po=a(),I=c("div"),f(Tt.$$.fragment),ls=a(),fn=c("p"),fn.innerHTML=za,cs=a(),gn=c("p"),gn.innerHTML=Ua,ds=a(),_n=c("p"),_n.innerHTML=Fa,ps=a(),D=c("div"),f(yt.$$.fragment),ms=a(),bn=c("p"),bn.innerHTML=Ia,hs=a(),f(xe.$$.fragment),us=a(),f(ze.$$.fragment),mo=a(),f(Mt.$$.fragment),ho=a(),z=c("div"),f(kt.$$.fragment),fs=a(),Tn=c("p"),Tn.textContent=Wa,gs=a(),yn=c("p"),yn.textContent=Za,_s=a(),Mn=c("p"),Mn.innerHTML=qa,bs=a(),kn=c("p"),kn.innerHTML=Ra,Ts=a(),K=c("div"),f(wt.$$.fragment),ys=a(),wn=c("p"),wn.innerHTML=Na,Ms=a(),f(Ue.$$.fragment),ks=a(),f(Fe.$$.fragment),uo=a(),f(vt.$$.fragment),fo=a(),W=c("div"),f($t.$$.fragment),ws=a(),vn=c("p"),vn.textContent=Va,vs=a(),$n=c("p"),$n.innerHTML=Ga,$s=a(),En=c("p"),En.innerHTML=Ba,Es=a(),B=c("div"),f(Et.$$.fragment),js=a(),jn=c("p"),jn.innerHTML=La,Cs=a(),f(Ie.$$.fragment),Js=a(),f(We.$$.fragment),xs=a(),f(Ze.$$.fragment),go=a(),f(jt.$$.fragment),_o=a(),Z=c("div"),f(Ct.$$.fragment),zs=a(),Cn=c("p"),Cn.textContent=Xa,Us=a(),Jn=c("p"),Jn.innerHTML=Sa,Fs=a(),xn=c("p"),xn.innerHTML=Ha,Is=a(),ee=c("div"),f(Jt.$$.fragment),Ws=a(),zn=c("p"),zn.innerHTML=Pa,Zs=a(),f(qe.$$.fragment),qs=a(),f(Re.$$.fragment),bo=a(),f(xt.$$.fragment),To=a(),U=c("div"),f(zt.$$.fragment),Rs=a(),Un=c("p"),Un.textContent=Qa,Ns=a(),Fn=c("p"),Fn.textContent=Aa,Vs=a(),In=c("p"),In.innerHTML=Ya,Gs=a(),Wn=c("p"),Wn.innerHTML=Oa,Bs=a(),te=c("div"),f(Ut.$$.fragment),Ls=a(),Zn=c("p"),Zn.innerHTML=Da,Xs=a(),f(Ne.$$.fragment),Ss=a(),f(Ve.$$.fragment),yo=a(),f(Ft.$$.fragment),Mo=a(),q=c("div"),f(It.$$.fragment),Hs=a(),qn=c("p"),qn.innerHTML=Ka,Ps=a(),Rn=c("p"),Rn.innerHTML=er,Qs=a(),Nn=c("p"),Nn.innerHTML=tr,As=a(),ne=c("div"),f(Wt.$$.fragment),Ys=a(),Vn=c("p"),Vn.innerHTML=nr,Os=a(),f(Ge.$$.fragment),Ds=a(),f(Be.$$.fragment),ko=a(),f(Zt.$$.fragment),wo=a(),Bn=c("p"),this.h()},l(e){const i=ir("svelte-u9bgzb",document.head);t=d(i,"META",{name:!0,content:!0}),i.forEach(l),h=r(e),o=d(e,"P",{}),v(o).forEach(l),p=r(e),k=d(e,"P",{"data-svelte-h":!0}),u(k)!=="svelte-fw7bkh"&&(k.innerHTML=s),M=r(e),j=d(e,"DIV",{style:!0,"data-svelte-h":!0}),u(j)!=="svelte-ithiq1"&&(j.innerHTML=Gn),He=r(e),g(ce.$$.fragment,e),Ln=r(e),Pe=d(e,"P",{"data-svelte-h":!0}),u(Pe)!=="svelte-1viltjo"&&(Pe.innerHTML=ea),Xn=r(e),Qe=d(e,"P",{"data-svelte-h":!0}),u(Qe)!=="svelte-1yeingl"&&(Qe.textContent=ta),Sn=r(e),Ae=d(e,"P",{"data-svelte-h":!0}),u(Ae)!=="svelte-1arg5lu"&&(Ae.innerHTML=na),Hn=r(e),g(ke.$$.fragment,e),Pn=r(e),Ye=d(e,"P",{"data-svelte-h":!0}),u(Ye)!=="svelte-fba0sb"&&(Ye.innerHTML=oa),Qn=r(e),g(we.$$.fragment,e),An=r(e),g(Oe.$$.fragment,e),Yn=r(e),G=d(e,"UL",{});var H=v(G);Gt=d(H,"LI",{"data-svelte-h":!0}),u(Gt)!=="svelte-47eafw"&&(Gt.innerHTML=sa),$o=r(H),Bt=d(H,"LI",{"data-svelte-h":!0}),u(Bt)!=="svelte-4wjht9"&&(Bt.innerHTML=aa),Eo=r(H),Lt=d(H,"LI",{"data-svelte-h":!0}),u(Lt)!=="svelte-i5emw0"&&(Lt.innerHTML=ra),jo=r(H),De=d(H,"LI",{});var qt=v(De);Xt=d(qt,"P",{"data-svelte-h":!0}),u(Xt)!=="svelte-c4nqvw"&&(Xt.textContent=ia),Co=r(qt),g(Ke.$$.fragment,qt),qt.forEach(l),Jo=r(H),St=d(H,"LI",{"data-svelte-h":!0}),u(St)!=="svelte-h19fb1"&&(St.innerHTML=la),H.forEach(l),On=r(e),g(et.$$.fragment,e),Dn=r(e),X=d(e,"DIV",{class:!0});var se=v(X);g(tt.$$.fragment,se),xo=r(se),Ht=d(se,"P",{"data-svelte-h":!0}),u(Ht)!=="svelte-sw6fp5"&&(Ht.innerHTML=ca),zo=r(se),Pt=d(se,"P",{"data-svelte-h":!0}),u(Pt)!=="svelte-1ek1ss9"&&(Pt.innerHTML=da),Uo=r(se),g(ve.$$.fragment,se),se.forEach(l),Kn=r(e),g(nt.$$.fragment,e),eo=r(e),J=d(e,"DIV",{class:!0});var R=v(J);g(ot.$$.fragment,R),Fo=r(R),Qt=d(R,"P",{"data-svelte-h":!0}),u(Qt)!=="svelte-182t1uv"&&(Qt.textContent=pa),Io=r(R),At=d(R,"P",{"data-svelte-h":!0}),u(At)!=="svelte-ntrhio"&&(At.innerHTML=ma),Wo=r(R),de=d(R,"DIV",{class:!0});var _e=v(de);g(st.$$.fragment,_e),Zo=r(_e),Yt=d(_e,"P",{"data-svelte-h":!0}),u(Yt)!=="svelte-8x1xqz"&&(Yt.textContent=ha),qo=r(_e),Ot=d(_e,"UL",{"data-svelte-h":!0}),u(Ot)!=="svelte-xi6653"&&(Ot.innerHTML=ua),_e.forEach(l),Ro=r(R),$e=d(R,"DIV",{class:!0});var Rt=v($e);g(at.$$.fragment,Rt),No=r(Rt),Dt=d(Rt,"P",{"data-svelte-h":!0}),u(Dt)!=="svelte-b3k2yi"&&(Dt.textContent=fa),Rt.forEach(l),Vo=r(R),Ee=d(R,"DIV",{class:!0});var Nt=v(Ee);g(rt.$$.fragment,Nt),Go=r(Nt),Kt=d(Nt,"P",{"data-svelte-h":!0}),u(Kt)!=="svelte-1f4f5kp"&&(Kt.innerHTML=ga),Nt.forEach(l),R.forEach(l),to=r(e),g(it.$$.fragment,e),no=r(e),S=d(e,"DIV",{class:!0});var ae=v(S);g(lt.$$.fragment,ae),Bo=r(ae),en=d(ae,"P",{"data-svelte-h":!0}),u(en)!=="svelte-ov9hug"&&(en.innerHTML=_a),Lo=r(ae),tn=d(ae,"P",{"data-svelte-h":!0}),u(tn)!=="svelte-gxzj9w"&&(tn.innerHTML=ba),Xo=r(ae),pe=d(ae,"DIV",{class:!0});var be=v(pe);g(ct.$$.fragment,be),So=r(be),nn=d(be,"P",{"data-svelte-h":!0}),u(nn)!=="svelte-18v9wij"&&(nn.textContent=Ta),Ho=r(be),on=d(be,"UL",{"data-svelte-h":!0}),u(on)!=="svelte-xi6653"&&(on.innerHTML=ya),be.forEach(l),ae.forEach(l),oo=r(e),g(dt.$$.fragment,e),so=r(e),ge=d(e,"DIV",{class:!0});var Vt=v(ge);g(pt.$$.fragment,Vt),Po=r(Vt),sn=d(Vt,"P",{"data-svelte-h":!0}),u(sn)!=="svelte-1fdcsdp"&&(sn.innerHTML=Ma),Vt.forEach(l),ao=r(e),g(mt.$$.fragment,e),ro=r(e),F=d(e,"DIV",{class:!0});var P=v(F);g(ht.$$.fragment,P),Qo=r(P),an=d(P,"P",{"data-svelte-h":!0}),u(an)!=="svelte-59rglw"&&(an.textContent=ka),Ao=r(P),rn=d(P,"P",{"data-svelte-h":!0}),u(rn)!=="svelte-q52n56"&&(rn.innerHTML=wa),Yo=r(P),ln=d(P,"P",{"data-svelte-h":!0}),u(ln)!=="svelte-hswkmf"&&(ln.innerHTML=va),Oo=r(P),me=d(P,"DIV",{class:!0});var Te=v(me);g(ut.$$.fragment,Te),Do=r(Te),cn=d(Te,"P",{"data-svelte-h":!0}),u(cn)!=="svelte-iu2f3b"&&(cn.innerHTML=$a),Ko=r(Te),g(je.$$.fragment,Te),Te.forEach(l),P.forEach(l),io=r(e),g(ft.$$.fragment,e),lo=r(e),x=d(e,"DIV",{class:!0});var N=v(x);g(gt.$$.fragment,N),es=r(N),dn=d(N,"P",{"data-svelte-h":!0}),u(dn)!=="svelte-x8zbwo"&&(dn.textContent=Ea),ts=r(N),pn=d(N,"P",{"data-svelte-h":!0}),u(pn)!=="svelte-1q5ogri"&&(pn.textContent=ja),ns=r(N),mn=d(N,"P",{"data-svelte-h":!0}),u(mn)!=="svelte-q52n56"&&(mn.innerHTML=Ca),os=r(N),hn=d(N,"P",{"data-svelte-h":!0}),u(hn)!=="svelte-hswkmf"&&(hn.innerHTML=Ja),ss=r(N),O=d(N,"DIV",{class:!0});var re=v(O);g(_t.$$.fragment,re),as=r(re),un=d(re,"P",{"data-svelte-h":!0}),u(un)!=="svelte-1okvkx1"&&(un.innerHTML=xa),rs=r(re),g(Ce.$$.fragment,re),is=r(re),g(Je.$$.fragment,re),re.forEach(l),N.forEach(l),co=r(e),g(bt.$$.fragment,e),po=r(e),I=d(e,"DIV",{class:!0});var Q=v(I);g(Tt.$$.fragment,Q),ls=r(Q),fn=d(Q,"P",{"data-svelte-h":!0}),u(fn)!=="svelte-1ghsxj"&&(fn.innerHTML=za),cs=r(Q),gn=d(Q,"P",{"data-svelte-h":!0}),u(gn)!=="svelte-q52n56"&&(gn.innerHTML=Ua),ds=r(Q),_n=d(Q,"P",{"data-svelte-h":!0}),u(_n)!=="svelte-hswkmf"&&(_n.innerHTML=Fa),ps=r(Q),D=d(Q,"DIV",{class:!0});var ie=v(D);g(yt.$$.fragment,ie),ms=r(ie),bn=d(ie,"P",{"data-svelte-h":!0}),u(bn)!=="svelte-1tuxm8b"&&(bn.innerHTML=Ia),hs=r(ie),g(xe.$$.fragment,ie),us=r(ie),g(ze.$$.fragment,ie),ie.forEach(l),Q.forEach(l),mo=r(e),g(Mt.$$.fragment,e),ho=r(e),z=d(e,"DIV",{class:!0});var V=v(z);g(kt.$$.fragment,V),fs=r(V),Tn=d(V,"P",{"data-svelte-h":!0}),u(Tn)!=="svelte-tyzulr"&&(Tn.textContent=Wa),gs=r(V),yn=d(V,"P",{"data-svelte-h":!0}),u(yn)!=="svelte-1weuqko"&&(yn.textContent=Za),_s=r(V),Mn=d(V,"P",{"data-svelte-h":!0}),u(Mn)!=="svelte-q52n56"&&(Mn.innerHTML=qa),bs=r(V),kn=d(V,"P",{"data-svelte-h":!0}),u(kn)!=="svelte-hswkmf"&&(kn.innerHTML=Ra),Ts=r(V),K=d(V,"DIV",{class:!0});var le=v(K);g(wt.$$.fragment,le),ys=r(le),wn=d(le,"P",{"data-svelte-h":!0}),u(wn)!=="svelte-149b6fv"&&(wn.innerHTML=Na),Ms=r(le),g(Ue.$$.fragment,le),ks=r(le),g(Fe.$$.fragment,le),le.forEach(l),V.forEach(l),uo=r(e),g(vt.$$.fragment,e),fo=r(e),W=d(e,"DIV",{class:!0});var A=v(W);g($t.$$.fragment,A),ws=r(A),vn=d(A,"P",{"data-svelte-h":!0}),u(vn)!=="svelte-1jftmi8"&&(vn.textContent=Va),vs=r(A),$n=d(A,"P",{"data-svelte-h":!0}),u($n)!=="svelte-q52n56"&&($n.innerHTML=Ga),$s=r(A),En=d(A,"P",{"data-svelte-h":!0}),u(En)!=="svelte-hswkmf"&&(En.innerHTML=Ba),Es=r(A),B=d(A,"DIV",{class:!0});var he=v(B);g(Et.$$.fragment,he),js=r(he),jn=d(he,"P",{"data-svelte-h":!0}),u(jn)!=="svelte-1ee0tzp"&&(jn.innerHTML=La),Cs=r(he),g(Ie.$$.fragment,he),Js=r(he),g(We.$$.fragment,he),xs=r(he),g(Ze.$$.fragment,he),he.forEach(l),A.forEach(l),go=r(e),g(jt.$$.fragment,e),_o=r(e),Z=d(e,"DIV",{class:!0});var ue=v(Z);g(Ct.$$.fragment,ue),zs=r(ue),Cn=d(ue,"P",{"data-svelte-h":!0}),u(Cn)!=="svelte-mpo3mz"&&(Cn.textContent=Xa),Us=r(ue),Jn=d(ue,"P",{"data-svelte-h":!0}),u(Jn)!=="svelte-q52n56"&&(Jn.innerHTML=Sa),Fs=r(ue),xn=d(ue,"P",{"data-svelte-h":!0}),u(xn)!=="svelte-hswkmf"&&(xn.innerHTML=Ha),Is=r(ue),ee=d(ue,"DIV",{class:!0});var Le=v(ee);g(Jt.$$.fragment,Le),Ws=r(Le),zn=d(Le,"P",{"data-svelte-h":!0}),u(zn)!=="svelte-ykn0dl"&&(zn.innerHTML=Pa),Zs=r(Le),g(qe.$$.fragment,Le),qs=r(Le),g(Re.$$.fragment,Le),Le.forEach(l),ue.forEach(l),bo=r(e),g(xt.$$.fragment,e),To=r(e),U=d(e,"DIV",{class:!0});var oe=v(U);g(zt.$$.fragment,oe),Rs=r(oe),Un=d(oe,"P",{"data-svelte-h":!0}),u(Un)!=="svelte-1ilonbf"&&(Un.textContent=Qa),Ns=r(oe),Fn=d(oe,"P",{"data-svelte-h":!0}),u(Fn)!=="svelte-27p1hy"&&(Fn.textContent=Aa),Vs=r(oe),In=d(oe,"P",{"data-svelte-h":!0}),u(In)!=="svelte-q52n56"&&(In.innerHTML=Ya),Gs=r(oe),Wn=d(oe,"P",{"data-svelte-h":!0}),u(Wn)!=="svelte-hswkmf"&&(Wn.innerHTML=Oa),Bs=r(oe),te=d(oe,"DIV",{class:!0});var Xe=v(te);g(Ut.$$.fragment,Xe),Ls=r(Xe),Zn=d(Xe,"P",{"data-svelte-h":!0}),u(Zn)!=="svelte-bn628t"&&(Zn.innerHTML=Da),Xs=r(Xe),g(Ne.$$.fragment,Xe),Ss=r(Xe),g(Ve.$$.fragment,Xe),Xe.forEach(l),oe.forEach(l),yo=r(e),g(Ft.$$.fragment,e),Mo=r(e),q=d(e,"DIV",{class:!0});var fe=v(q);g(It.$$.fragment,fe),Hs=r(fe),qn=d(fe,"P",{"data-svelte-h":!0}),u(qn)!=="svelte-umcq3z"&&(qn.innerHTML=Ka),Ps=r(fe),Rn=d(fe,"P",{"data-svelte-h":!0}),u(Rn)!=="svelte-q52n56"&&(Rn.innerHTML=er),Qs=r(fe),Nn=d(fe,"P",{"data-svelte-h":!0}),u(Nn)!=="svelte-hswkmf"&&(Nn.innerHTML=tr),As=r(fe),ne=d(fe,"DIV",{class:!0});var Se=v(ne);g(Wt.$$.fragment,Se),Ys=r(Se),Vn=d(Se,"P",{"data-svelte-h":!0}),u(Vn)!=="svelte-1tmmkmp"&&(Vn.innerHTML=nr),Os=r(Se),g(Ge.$$.fragment,Se),Ds=r(Se),g(Be.$$.fragment,Se),Se.forEach(l),fe.forEach(l),ko=r(e),g(Zt.$$.fragment,e),wo=r(e),Bn=d(e,"P",{}),v(Bn).forEach(l),this.h()},h(){$(t,"name","hf:doc:metadata"),$(t,"content",Ir),lr(j,"float","right"),$(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$($e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,i){n(document.head,t),m(e,h,i),m(e,o,i),m(e,p,i),m(e,k,i),m(e,M,i),m(e,j,i),m(e,He,i),_(ce,e,i),m(e,Ln,i),m(e,Pe,i),m(e,Xn,i),m(e,Qe,i),m(e,Sn,i),m(e,Ae,i),m(e,Hn,i),_(ke,e,i),m(e,Pn,i),m(e,Ye,i),m(e,Qn,i),_(we,e,i),m(e,An,i),_(Oe,e,i),m(e,Yn,i),m(e,G,i),n(G,Gt),n(G,$o),n(G,Bt),n(G,Eo),n(G,Lt),n(G,jo),n(G,De),n(De,Xt),n(De,Co),_(Ke,De,null),n(G,Jo),n(G,St),m(e,On,i),_(et,e,i),m(e,Dn,i),m(e,X,i),_(tt,X,null),n(X,xo),n(X,Ht),n(X,zo),n(X,Pt),n(X,Uo),_(ve,X,null),m(e,Kn,i),_(nt,e,i),m(e,eo,i),m(e,J,i),_(ot,J,null),n(J,Fo),n(J,Qt),n(J,Io),n(J,At),n(J,Wo),n(J,de),_(st,de,null),n(de,Zo),n(de,Yt),n(de,qo),n(de,Ot),n(J,Ro),n(J,$e),_(at,$e,null),n($e,No),n($e,Dt),n(J,Vo),n(J,Ee),_(rt,Ee,null),n(Ee,Go),n(Ee,Kt),m(e,to,i),_(it,e,i),m(e,no,i),m(e,S,i),_(lt,S,null),n(S,Bo),n(S,en),n(S,Lo),n(S,tn),n(S,Xo),n(S,pe),_(ct,pe,null),n(pe,So),n(pe,nn),n(pe,Ho),n(pe,on),m(e,oo,i),_(dt,e,i),m(e,so,i),m(e,ge,i),_(pt,ge,null),n(ge,Po),n(ge,sn),m(e,ao,i),_(mt,e,i),m(e,ro,i),m(e,F,i),_(ht,F,null),n(F,Qo),n(F,an),n(F,Ao),n(F,rn),n(F,Yo),n(F,ln),n(F,Oo),n(F,me),_(ut,me,null),n(me,Do),n(me,cn),n(me,Ko),_(je,me,null),m(e,io,i),_(ft,e,i),m(e,lo,i),m(e,x,i),_(gt,x,null),n(x,es),n(x,dn),n(x,ts),n(x,pn),n(x,ns),n(x,mn),n(x,os),n(x,hn),n(x,ss),n(x,O),_(_t,O,null),n(O,as),n(O,un),n(O,rs),_(Ce,O,null),n(O,is),_(Je,O,null),m(e,co,i),_(bt,e,i),m(e,po,i),m(e,I,i),_(Tt,I,null),n(I,ls),n(I,fn),n(I,cs),n(I,gn),n(I,ds),n(I,_n),n(I,ps),n(I,D),_(yt,D,null),n(D,ms),n(D,bn),n(D,hs),_(xe,D,null),n(D,us),_(ze,D,null),m(e,mo,i),_(Mt,e,i),m(e,ho,i),m(e,z,i),_(kt,z,null),n(z,fs),n(z,Tn),n(z,gs),n(z,yn),n(z,_s),n(z,Mn),n(z,bs),n(z,kn),n(z,Ts),n(z,K),_(wt,K,null),n(K,ys),n(K,wn),n(K,Ms),_(Ue,K,null),n(K,ks),_(Fe,K,null),m(e,uo,i),_(vt,e,i),m(e,fo,i),m(e,W,i),_($t,W,null),n(W,ws),n(W,vn),n(W,vs),n(W,$n),n(W,$s),n(W,En),n(W,Es),n(W,B),_(Et,B,null),n(B,js),n(B,jn),n(B,Cs),_(Ie,B,null),n(B,Js),_(We,B,null),n(B,xs),_(Ze,B,null),m(e,go,i),_(jt,e,i),m(e,_o,i),m(e,Z,i),_(Ct,Z,null),n(Z,zs),n(Z,Cn),n(Z,Us),n(Z,Jn),n(Z,Fs),n(Z,xn),n(Z,Is),n(Z,ee),_(Jt,ee,null),n(ee,Ws),n(ee,zn),n(ee,Zs),_(qe,ee,null),n(ee,qs),_(Re,ee,null),m(e,bo,i),_(xt,e,i),m(e,To,i),m(e,U,i),_(zt,U,null),n(U,Rs),n(U,Un),n(U,Ns),n(U,Fn),n(U,Vs),n(U,In),n(U,Gs),n(U,Wn),n(U,Bs),n(U,te),_(Ut,te,null),n(te,Ls),n(te,Zn),n(te,Xs),_(Ne,te,null),n(te,Ss),_(Ve,te,null),m(e,yo,i),_(Ft,e,i),m(e,Mo,i),m(e,q,i),_(It,q,null),n(q,Hs),n(q,qn),n(q,Ps),n(q,Rn),n(q,Qs),n(q,Nn),n(q,As),n(q,ne),_(Wt,ne,null),n(ne,Ys),n(ne,Vn),n(ne,Os),_(Ge,ne,null),n(ne,Ds),_(Be,ne,null),m(e,ko,i),_(Zt,e,i),m(e,wo,i),m(e,Bn,i),vo=!0},p(e,[i]){const H={};i&2&&(H.$$scope={dirty:i,ctx:e}),ke.$set(H);const qt={};i&2&&(qt.$$scope={dirty:i,ctx:e}),we.$set(qt);const se={};i&2&&(se.$$scope={dirty:i,ctx:e}),ve.$set(se);const R={};i&2&&(R.$$scope={dirty:i,ctx:e}),je.$set(R);const _e={};i&2&&(_e.$$scope={dirty:i,ctx:e}),Ce.$set(_e);const Rt={};i&2&&(Rt.$$scope={dirty:i,ctx:e}),Je.$set(Rt);const Nt={};i&2&&(Nt.$$scope={dirty:i,ctx:e}),xe.$set(Nt);const ae={};i&2&&(ae.$$scope={dirty:i,ctx:e}),ze.$set(ae);const be={};i&2&&(be.$$scope={dirty:i,ctx:e}),Ue.$set(be);const Vt={};i&2&&(Vt.$$scope={dirty:i,ctx:e}),Fe.$set(Vt);const P={};i&2&&(P.$$scope={dirty:i,ctx:e}),Ie.$set(P);const Te={};i&2&&(Te.$$scope={dirty:i,ctx:e}),We.$set(Te);const N={};i&2&&(N.$$scope={dirty:i,ctx:e}),Ze.$set(N);const re={};i&2&&(re.$$scope={dirty:i,ctx:e}),qe.$set(re);const Q={};i&2&&(Q.$$scope={dirty:i,ctx:e}),Re.$set(Q);const ie={};i&2&&(ie.$$scope={dirty:i,ctx:e}),Ne.$set(ie);const V={};i&2&&(V.$$scope={dirty:i,ctx:e}),Ve.$set(V);const le={};i&2&&(le.$$scope={dirty:i,ctx:e}),Ge.$set(le);const A={};i&2&&(A.$$scope={dirty:i,ctx:e}),Be.$set(A)},i(e){vo||(b(ce.$$.fragment,e),b(ke.$$.fragment,e),b(we.$$.fragment,e),b(Oe.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(ve.$$.fragment,e),b(nt.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b(rt.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(ct.$$.fragment,e),b(dt.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(je.$$.fragment,e),b(ft.$$.fragment,e),b(gt.$$.fragment,e),b(_t.$$.fragment,e),b(Ce.$$.fragment,e),b(Je.$$.fragment,e),b(bt.$$.fragment,e),b(Tt.$$.fragment,e),b(yt.$$.fragment,e),b(xe.$$.fragment,e),b(ze.$$.fragment,e),b(Mt.$$.fragment,e),b(kt.$$.fragment,e),b(wt.$$.fragment,e),b(Ue.$$.fragment,e),b(Fe.$$.fragment,e),b(vt.$$.fragment,e),b($t.$$.fragment,e),b(Et.$$.fragment,e),b(Ie.$$.fragment,e),b(We.$$.fragment,e),b(Ze.$$.fragment,e),b(jt.$$.fragment,e),b(Ct.$$.fragment,e),b(Jt.$$.fragment,e),b(qe.$$.fragment,e),b(Re.$$.fragment,e),b(xt.$$.fragment,e),b(zt.$$.fragment,e),b(Ut.$$.fragment,e),b(Ne.$$.fragment,e),b(Ve.$$.fragment,e),b(Ft.$$.fragment,e),b(It.$$.fragment,e),b(Wt.$$.fragment,e),b(Ge.$$.fragment,e),b(Be.$$.fragment,e),b(Zt.$$.fragment,e),vo=!0)},o(e){T(ce.$$.fragment,e),T(ke.$$.fragment,e),T(we.$$.fragment,e),T(Oe.$$.fragment,e),T(Ke.$$.fragment,e),T(et.$$.fragment,e),T(tt.$$.fragment,e),T(ve.$$.fragment,e),T(nt.$$.fragment,e),T(ot.$$.fragment,e),T(st.$$.fragment,e),T(at.$$.fragment,e),T(rt.$$.fragment,e),T(it.$$.fragment,e),T(lt.$$.fragment,e),T(ct.$$.fragment,e),T(dt.$$.fragment,e),T(pt.$$.fragment,e),T(mt.$$.fragment,e),T(ht.$$.fragment,e),T(ut.$$.fragment,e),T(je.$$.fragment,e),T(ft.$$.fragment,e),T(gt.$$.fragment,e),T(_t.$$.fragment,e),T(Ce.$$.fragment,e),T(Je.$$.fragment,e),T(bt.$$.fragment,e),T(Tt.$$.fragment,e),T(yt.$$.fragment,e),T(xe.$$.fragment,e),T(ze.$$.fragment,e),T(Mt.$$.fragment,e),T(kt.$$.fragment,e),T(wt.$$.fragment,e),T(Ue.$$.fragment,e),T(Fe.$$.fragment,e),T(vt.$$.fragment,e),T($t.$$.fragment,e),T(Et.$$.fragment,e),T(Ie.$$.fragment,e),T(We.$$.fragment,e),T(Ze.$$.fragment,e),T(jt.$$.fragment,e),T(Ct.$$.fragment,e),T(Jt.$$.fragment,e),T(qe.$$.fragment,e),T(Re.$$.fragment,e),T(xt.$$.fragment,e),T(zt.$$.fragment,e),T(Ut.$$.fragment,e),T(Ne.$$.fragment,e),T(Ve.$$.fragment,e),T(Ft.$$.fragment,e),T(It.$$.fragment,e),T(Wt.$$.fragment,e),T(Ge.$$.fragment,e),T(Be.$$.fragment,e),T(Zt.$$.fragment,e),vo=!1},d(e){e&&(l(h),l(o),l(p),l(k),l(M),l(j),l(He),l(Ln),l(Pe),l(Xn),l(Qe),l(Sn),l(Ae),l(Hn),l(Pn),l(Ye),l(Qn),l(An),l(Yn),l(G),l(On),l(Dn),l(X),l(Kn),l(eo),l(J),l(to),l(no),l(S),l(oo),l(so),l(ge),l(ao),l(ro),l(F),l(io),l(lo),l(x),l(co),l(po),l(I),l(mo),l(ho),l(z),l(uo),l(fo),l(W),l(go),l(_o),l(Z),l(bo),l(To),l(U),l(yo),l(Mo),l(q),l(ko),l(wo),l(Bn)),l(t),y(ce,e),y(ke,e),y(we,e),y(Oe,e),y(Ke),y(et,e),y(tt),y(ve),y(nt,e),y(ot),y(st),y(at),y(rt),y(it,e),y(lt),y(ct),y(dt,e),y(pt),y(mt,e),y(ht),y(ut),y(je),y(ft,e),y(gt),y(_t),y(Ce),y(Je),y(bt,e),y(Tt),y(yt),y(xe),y(ze),y(Mt,e),y(kt),y(wt),y(Ue),y(Fe),y(vt,e),y($t),y(Et),y(Ie),y(We),y(Ze),y(jt,e),y(Ct),y(Jt),y(qe),y(Re),y(xt,e),y(zt),y(Ut),y(Ne),y(Ve),y(Ft,e),y(It),y(Wt),y(Ge),y(Be),y(Zt,e)}}}const Ir='{"title":"ELECTRA","local":"electra","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"ElectraConfig","local":"transformers.ElectraConfig","sections":[],"depth":2},{"title":"ElectraTokenizer","local":"transformers.ElectraTokenizer","sections":[],"depth":2},{"title":"ElectraTokenizerFast","local":"transformers.ElectraTokenizerFast","sections":[],"depth":2},{"title":"Electra specific outputs","local":"transformers.models.electra.modeling_electra.ElectraForPreTrainingOutput","sections":[],"depth":2},{"title":"ElectraModel","local":"transformers.ElectraModel","sections":[],"depth":2},{"title":"ElectraForPreTraining","local":"transformers.ElectraForPreTraining","sections":[],"depth":2},{"title":"ElectraForCausalLM","local":"transformers.ElectraForCausalLM","sections":[],"depth":2},{"title":"ElectraForMaskedLM","local":"transformers.ElectraForMaskedLM","sections":[],"depth":2},{"title":"ElectraForSequenceClassification","local":"transformers.ElectraForSequenceClassification","sections":[],"depth":2},{"title":"ElectraForMultipleChoice","local":"transformers.ElectraForMultipleChoice","sections":[],"depth":2},{"title":"ElectraForTokenClassification","local":"transformers.ElectraForTokenClassification","sections":[],"depth":2},{"title":"ElectraForQuestionAnswering","local":"transformers.ElectraForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Wr(w){return sr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Xr extends ar{constructor(t){super(),rr(this,t,Wr,Fr,or,{})}}export{Xr as component};
