import{s as bn,o as Mn,n as V}from"../chunks/scheduler.18a86fab.js";import{S as yn,i as vn,g as p,s as r,r as u,A as wn,h as m,f as a,c as l,j as H,x as y,u as f,k as Z,y as d,a as c,v as g,d as _,t as T,w as b}from"../chunks/index.98837b22.js";import{T as Yt}from"../chunks/Tip.77304350.js";import{D as X}from"../chunks/Docstring.a1ef7999.js";import{C as Le}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as je}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as S,E as kn}from"../chunks/getInferenceSnippets.06c2775f.js";function Cn(w){let t,h="Examples:",o,i,M;return i=new Le({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENUUkxDb25maWclMkMlMjBDVFJMTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwQ1RSTCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwQ1RSTENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBDVFJMTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CTRLConfig, CTRLModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a CTRL configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = CTRLConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CTRLModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=r(),u(i.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=l(n),f(i.$$.fragment,n)},m(n,v){c(n,t,v),c(n,o,v),g(i,n,v),M=!0},p:V,i(n){M||(_(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(a(t),a(o)),b(i,n)}}}function $n(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,i){c(o,t,i)},p:V,d(o){o&&a(t)}}}function jn(w){let t,h="Example:",o,i,M;return i=new Le({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBDVFJMTW9kZWwlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlNhbGVzZm9yY2UlMkZjdHJsJTIyKSUwQW1vZGVsJTIwJTNEJTIwQ1RSTE1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJTYWxlc2ZvcmNlJTJGY3RybCUyMiklMEElMEElMjMlMjBDVFJMJTIwd2FzJTIwdHJhaW5lZCUyMHdpdGglMjBjb250cm9sJTIwY29kZXMlMjBhcyUyMHRoZSUyMGZpcnN0JTIwdG9rZW4lMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyT3BpbmlvbiUyME15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFhc3NlcnQlMjBpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQlNUIwJTJDJTIwMCU1RC5pdGVtKCklMjBpbiUyMHRva2VuaXplci5jb250cm9sX2NvZGVzLnZhbHVlcygpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGUlMEFsaXN0KGxhc3RfaGlkZGVuX3N0YXRlcy5zaGFwZSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CTRLModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CTRLModel.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># CTRL was trained with control codes as the first token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Opinion My dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> inputs[<span class="hljs-string">&quot;input_ids&quot;</span>][<span class="hljs-number">0</span>, <span class="hljs-number">0</span>].item() <span class="hljs-keyword">in</span> tokenizer.control_codes.values()

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">5</span>, <span class="hljs-number">1280</span>]`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=r(),u(i.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=l(n),f(i.$$.fragment,n)},m(n,v){c(n,t,v),c(n,o,v),g(i,n,v),M=!0},p:V,i(n){M||(_(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(a(t),a(o)),b(i,n)}}}function Ln(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,i){c(o,t,i)},p:V,d(o){o&&a(t)}}}function Jn(w){let t,h="Example:",o,i,M;return i=new Le({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMENUUkxMTUhlYWRNb2RlbCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlNhbGVzZm9yY2UlMkZjdHJsJTIyKSUwQW1vZGVsJTIwJTNEJTIwQ1RSTExNSGVhZE1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJTYWxlc2ZvcmNlJTJGY3RybCUyMiklMEElMEElMjMlMjBDVFJMJTIwd2FzJTIwdHJhaW5lZCUyMHdpdGglMjBjb250cm9sJTIwY29kZXMlMjBhcyUyMHRoZSUyMGZpcnN0JTIwdG9rZW4lMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyV2lraXBlZGlhJTIwVGhlJTIwbGxhbWElMjBpcyUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBYXNzZXJ0JTIwaW5wdXRzJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTVCMCUyQyUyMDAlNUQuaXRlbSgpJTIwaW4lMjB0b2tlbml6ZXIuY29udHJvbF9jb2Rlcy52YWx1ZXMoKSUwQSUwQXNlcXVlbmNlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cyU1QiUyMmlucHV0X2lkcyUyMiU1RCklMEFzZXF1ZW5jZXMlMjAlM0QlMjB0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKHNlcXVlbmNlX2lkcyklMEFzZXF1ZW5jZXMlMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQpJTBBcm91bmQob3V0cHV0cy5sb3NzLml0ZW0oKSUyQyUyMDIpJTBBJTBBbGlzdChvdXRwdXRzLmxvZ2l0cy5zaGFwZSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CTRLLMHeadModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CTRLLMHeadModel.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># CTRL was trained with control codes as the first token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Wikipedia The llama is&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> inputs[<span class="hljs-string">&quot;input_ids&quot;</span>][<span class="hljs-number">0</span>, <span class="hljs-number">0</span>].item() <span class="hljs-keyword">in</span> tokenizer.control_codes.values()

<span class="hljs-meta">&gt;&gt;&gt; </span>sequence_ids = model.generate(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>sequences = tokenizer.batch_decode(sequence_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>sequences
[<span class="hljs-string">&#x27;Wikipedia The llama is a member of the family Bovidae. It is native to the Andes of Peru,&#x27;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(outputs.loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">9.21</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(outputs.logits.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">5</span>, <span class="hljs-number">246534</span>]`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=r(),u(i.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=l(n),f(i.$$.fragment,n)},m(n,v){c(n,t,v),c(n,o,v),g(i,n,v),M=!0},p:V,i(n){M||(_(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(a(t),a(o)),b(i,n)}}}function Rn(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,i){c(o,t,i)},p:V,d(o){o&&a(t)}}}function xn(w){let t,h="Example of single-label classification:",o,i,M;return i=new Le({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMENUUkxGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyU2FsZXNmb3JjZSUyRmN0cmwlMjIpJTBBbW9kZWwlMjAlM0QlMjBDVFJMRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyU2FsZXNmb3JjZSUyRmN0cmwlMjIpJTBBJTBBJTIzJTIwQ1RSTCUyMHdhcyUyMHRyYWluZWQlMjB3aXRoJTIwY29udHJvbCUyMGNvZGVzJTIwYXMlMjB0aGUlMjBmaXJzdCUyMHRva2VuJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMk9waW5pb24lMjBNeSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBYXNzZXJ0JTIwaW5wdXRzJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTVCMCUyQyUyMDAlNUQuaXRlbSgpJTIwaW4lMjB0b2tlbml6ZXIuY29udHJvbF9jb2Rlcy52YWx1ZXMoKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CTRLForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CTRLForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># CTRL was trained with control codes as the first token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Opinion My dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> inputs[<span class="hljs-string">&quot;input_ids&quot;</span>][<span class="hljs-number">0</span>, <span class="hljs-number">0</span>].item() <span class="hljs-keyword">in</span> tokenizer.control_codes.values()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
<span class="hljs-string">&#x27;LABEL_0&#x27;</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=r(),u(i.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-ykxpe4"&&(t.textContent=h),o=l(n),f(i.$$.fragment,n)},m(n,v){c(n,t,v),c(n,o,v),g(i,n,v),M=!0},p:V,i(n){M||(_(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(a(t),a(o)),b(i,n)}}}function zn(w){let t,h;return t=new Le({props:{code:"aW1wb3J0JTIwdG9yY2glMEElMEF0b3JjaC5tYW51YWxfc2VlZCg0MiklMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBDVFJMRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyU2FsZXNmb3JjZSUyRmN0cmwlMjIlMkMlMjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyklMEElMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMSklMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>torch.manual_seed(<span class="hljs-number">42</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CTRLForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">0.93</span>`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,i){g(t,o,i),h=!0},p:V,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){T(t.$$.fragment,o),h=!1},d(o){b(t,o)}}}function In(w){let t,h="Example of multi-label classification:",o,i,M;return i=new Le({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMENUUkxGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyU2FsZXNmb3JjZSUyRmN0cmwlMjIpJTBBbW9kZWwlMjAlM0QlMjBDVFJMRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyU2FsZXNmb3JjZSUyRmN0cmwlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiUwQSklMEElMEElMjMlMjBDVFJMJTIwd2FzJTIwdHJhaW5lZCUyMHdpdGglMjBjb250cm9sJTIwY29kZXMlMjBhcyUyMHRoZSUyMGZpcnN0JTIwdG9rZW4lMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyT3BpbmlvbiUyME15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFhc3NlcnQlMjBpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQlNUIwJTJDJTIwMCU1RC5pdGVtKCklMjBpbiUyMHRva2VuaXplci5jb250cm9sX2NvZGVzLnZhbHVlcygpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CTRLForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CTRLForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># CTRL was trained with control codes as the first token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Opinion My dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> inputs[<span class="hljs-string">&quot;input_ids&quot;</span>][<span class="hljs-number">0</span>, <span class="hljs-number">0</span>].item() <span class="hljs-keyword">in</span> tokenizer.control_codes.values()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
<span class="hljs-string">&#x27;LABEL_0&#x27;</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=r(),u(i.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-1l8e32d"&&(t.textContent=h),o=l(n),f(i.$$.fragment,n)},m(n,v){c(n,t,v),c(n,o,v),g(i,n,v),M=!0},p:V,i(n){M||(_(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(a(t),a(o)),b(i,n)}}}function Wn(w){let t,h;return t=new Le({props:{code:"JTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwQ1RSTEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMlNhbGVzZm9yY2UlMkZjdHJsJTIyJTJDJTIwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMpJTBBJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gubm4uZnVuY3Rpb25hbC5vbmVfaG90KHRvcmNoLnRlbnNvciglNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKS50byglMEElMjAlMjAlMjAlMjB0b3JjaC5mbG9hdCUwQSklMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFsb3NzLmJhY2t3YXJkKCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CTRLForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;Salesforce/ctrl&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
<span class="hljs-meta">... </span>    torch.<span class="hljs-built_in">float</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,i){g(t,o,i),h=!0},p:V,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){T(t.$$.fragment,o),h=!1},d(o){b(t,o)}}}function Un(w){let t,h,o,i,M,n="<em>This model was released on 2019-09-11 and added to Hugging Face Transformers on 2020-11-16.</em>",v,ne,Pe,G,Pt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Qe,se,De,oe,Qt=`CTRL model was proposed in <a href="https://huggingface.co/papers/1909.05858" rel="nofollow">CTRL: A Conditional Transformer Language Model for Controllable Generation</a> by Nitish Shirish Keskar<em>, Bryan McCann</em>, Lav R. Varshney, Caiming Xiong and
Richard Socher. It’s a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus
of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).`,Oe,ae,Dt="The abstract from the paper is the following:",Ae,re,Ot=`<em>Large-scale language models show promising text generation capabilities, but users cannot easily control particular
aspects of the generated text. We release CTRL, a 1.63 billion-parameter conditional transformer language model,
trained to condition on control codes that govern style, content, and task-specific behavior. Control codes were
derived from structure that naturally co-occurs with raw text, preserving the advantages of unsupervised learning while
providing more explicit control over text generation. These codes also allow CTRL to predict which parts of the
training data are most likely given a sequence. This provides a potential method for analyzing large amounts of data
via model-based source attribution.</em>`,Ke,le,At=`This model was contributed by <a href="https://huggingface.co/keskarnitishr" rel="nofollow">keskarnitishr</a>. The original code can be found
<a href="https://github.com/salesforce/ctrl" rel="nofollow">here</a>.`,et,ie,tt,ce,Kt=`<li>CTRL makes use of control codes to generate text: it requires generations to be started by certain words, sentences
or links to generate coherent text. Refer to the <a href="https://github.com/salesforce/ctrl" rel="nofollow">original implementation</a> for
more information.</li> <li>CTRL is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than
the left.</li> <li>CTRL was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
token in a sequence. Leveraging this feature allows CTRL to generate syntactically coherent text as it can be
observed in the <em>run_generation.py</em> example script.</li> <li>The PyTorch models can take the <code>past_key_values</code> as input, which is the previously computed key/value attention pairs.
Using the <code>past_key_values</code> value prevents the model from re-computing
pre-computed values in the context of text generation. See the <a href="model_doc/ctrl#transformers.CTRLModel.forward"><code>forward</code></a>
method for more information on the usage of this argument.</li>`,nt,de,st,pe,en='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li>',ot,me,at,J,he,_t,Je,tn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel">CTRLModel</a> or a <code>TFCTRLModel</code>. It is used to
instantiate a CTRL model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
<a href="https://huggingface.co/Salesforce/ctrl" rel="nofollow">Salesforce/ctrl</a> architecture from SalesForce.`,Tt,Re,nn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,bt,E,rt,ue,lt,R,fe,Mt,xe,sn="Construct a CTRL tokenizer. Based on Byte-Pair-Encoding.",yt,ze,on=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,vt,Ie,ge,it,_e,ct,$,Te,wt,We,an="The bare Ctrl Model outputting raw hidden-states without any specific head on top.",kt,Ue,rn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ct,Ze,ln=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$t,W,be,jt,Fe,cn='The <a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel">CTRLModel</a> forward method, overrides the <code>__call__</code> special method.',Lt,Y,Jt,P,dt,Me,pt,j,ye,Rt,Be,dn=`The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`,xt,qe,pn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zt,Ne,mn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,It,U,ve,Wt,He,hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLLMHeadModel">CTRLLMHeadModel</a> forward method, overrides the <code>__call__</code> special method.',Ut,Q,Zt,D,mt,we,ht,L,ke,Ft,Ve,un=`The CTRL Model transformer with a sequence classification head on top (linear layer).
<a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLForSequenceClassification">CTRLForSequenceClassification</a> uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last
token. If a <code>pad_token_id</code> is defined in the configuration, it finds the last token that is not a padding token in
each row. If no <code>pad_token_id</code> is defined, it simply takes the last value in each row of the batch. Since it cannot
guess the padding tokens when <code>inputs_embeds</code> are passed instead of <code>input_ids</code>, it does the same (take the last
value in each row of the batch).`,Bt,Xe,fn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,qt,Se,gn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Nt,k,Ce,Ht,Ge,_n='The <a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLForSequenceClassification">CTRLForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Vt,O,Xt,A,St,K,Gt,ee,Et,te,ut,$e,ft,Ee,gt;return ne=new S({props:{title:"CTRL",local:"ctrl",headingTag:"h1"}}),se=new S({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ie=new S({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),de=new S({props:{title:"Resources",local:"resources",headingTag:"h2"}}),me=new S({props:{title:"CTRLConfig",local:"transformers.CTRLConfig",headingTag:"h2"}}),he=new X({props:{name:"class transformers.CTRLConfig",anchor:"transformers.CTRLConfig",parameters:[{name:"vocab_size",val:" = 246534"},{name:"n_positions",val:" = 256"},{name:"n_embd",val:" = 1280"},{name:"dff",val:" = 8192"},{name:"n_layer",val:" = 48"},{name:"n_head",val:" = 16"},{name:"resid_pdrop",val:" = 0.1"},{name:"embd_pdrop",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-06"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CTRLConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 246534) &#x2014;
Vocabulary size of the CTRL model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel">CTRLModel</a> or <code>TFCTRLModel</code>.`,name:"vocab_size"},{anchor:"transformers.CTRLConfig.n_positions",description:`<strong>n_positions</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"n_positions"},{anchor:"transformers.CTRLConfig.n_embd",description:`<strong>n_embd</strong> (<code>int</code>, <em>optional</em>, defaults to 1280) &#x2014;
Dimensionality of the embeddings and hidden states.`,name:"n_embd"},{anchor:"transformers.CTRLConfig.dff",description:`<strong>dff</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimensionality of the inner dimension of the feed forward networks (FFN).`,name:"dff"},{anchor:"transformers.CTRLConfig.n_layer",description:`<strong>n_layer</strong> (<code>int</code>, <em>optional</em>, defaults to 48) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"n_layer"},{anchor:"transformers.CTRLConfig.n_head",description:`<strong>n_head</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"n_head"},{anchor:"transformers.CTRLConfig.resid_pdrop",description:`<strong>resid_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"resid_pdrop"},{anchor:"transformers.CTRLConfig.embd_pdrop",description:`<strong>embd_pdrop</strong> (<code>int</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the embeddings.`,name:"embd_pdrop"},{anchor:"transformers.CTRLConfig.layer_norm_epsilon",description:`<strong>layer_norm_epsilon</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon to use in the layer normalization layers`,name:"layer_norm_epsilon"},{anchor:"transformers.CTRLConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.CTRLConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/configuration_ctrl.py#L24"}}),E=new je({props:{anchor:"transformers.CTRLConfig.example",$$slots:{default:[Cn]},$$scope:{ctx:w}}}),ue=new S({props:{title:"CTRLTokenizer",local:"transformers.CTRLTokenizer",headingTag:"h2"}}),fe=new X({props:{name:"class transformers.CTRLTokenizer",anchor:"transformers.CTRLTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"unk_token",val:" = '<unk>'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CTRLTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.CTRLTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.CTRLTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/tokenization_ctrl.py#L110"}}),ge=new X({props:{name:"save_vocabulary",anchor:"transformers.CTRLTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/tokenization_ctrl.py#L215"}}),_e=new S({props:{title:"CTRLModel",local:"transformers.CTRLModel",headingTag:"h2"}}),Te=new X({props:{name:"class transformers.CTRLModel",anchor:"transformers.CTRLModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CTRLModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLModel">CTRLModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L234"}}),be=new X({props:{name:"forward",anchor:"transformers.CTRLModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CTRLModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0].shape[-2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only input IDs that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CTRLModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CTRLModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CTRLModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CTRLModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CTRLModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CTRLModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CTRLModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.CTRLModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CTRLModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CTRLModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.CTRLModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L270",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig"
>CTRLConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache"
>Cache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
<code>config.is_encoder_decoder=True</code> in the cross-attention blocks) that can be used (see <code>past_key_values</code>
input) to speed up sequential decoding.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Y=new Yt({props:{$$slots:{default:[$n]},$$scope:{ctx:w}}}),P=new je({props:{anchor:"transformers.CTRLModel.forward.example",$$slots:{default:[jn]},$$scope:{ctx:w}}}),Me=new S({props:{title:"CTRLLMHeadModel",local:"transformers.CTRLLMHeadModel",headingTag:"h2"}}),ye=new X({props:{name:"class transformers.CTRLLMHeadModel",anchor:"transformers.CTRLLMHeadModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CTRLLMHeadModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLLMHeadModel">CTRLLMHeadModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L444"}}),ve=new X({props:{name:"forward",anchor:"transformers.CTRLLMHeadModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CTRLLMHeadModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0].shape[-2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only input IDs that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CTRLLMHeadModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CTRLLMHeadModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CTRLLMHeadModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CTRLLMHeadModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CTRLLMHeadModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CTRLLMHeadModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CTRLLMHeadModel.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.CTRLLMHeadModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.CTRLLMHeadModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CTRLLMHeadModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CTRLLMHeadModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.CTRLLMHeadModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L455",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig"
>CTRLConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache"
>Cache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Q=new Yt({props:{$$slots:{default:[Ln]},$$scope:{ctx:w}}}),D=new je({props:{anchor:"transformers.CTRLLMHeadModel.forward.example",$$slots:{default:[Jn]},$$scope:{ctx:w}}}),we=new S({props:{title:"CTRLForSequenceClassification",local:"transformers.CTRLForSequenceClassification",headingTag:"h2"}}),ke=new X({props:{name:"class transformers.CTRLForSequenceClassification",anchor:"transformers.CTRLForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CTRLForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLForSequenceClassification">CTRLForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L587"}}),Ce=new X({props:{name:"forward",anchor:"transformers.CTRLForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.CTRLForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values[0].shape[-2]</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only input IDs that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CTRLForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CTRLForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CTRLForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CTRLForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CTRLForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CTRLForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CTRLForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.CTRLForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.CTRLForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CTRLForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CTRLForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ctrl/modeling_ctrl.py#L597",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ctrl#transformers.CTRLConfig"
>CTRLConfig</a>) and inputs.</p>
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
`}}),O=new Yt({props:{$$slots:{default:[Rn]},$$scope:{ctx:w}}}),A=new je({props:{anchor:"transformers.CTRLForSequenceClassification.forward.example",$$slots:{default:[xn]},$$scope:{ctx:w}}}),K=new je({props:{anchor:"transformers.CTRLForSequenceClassification.forward.example-2",$$slots:{default:[zn]},$$scope:{ctx:w}}}),ee=new je({props:{anchor:"transformers.CTRLForSequenceClassification.forward.example-3",$$slots:{default:[In]},$$scope:{ctx:w}}}),te=new je({props:{anchor:"transformers.CTRLForSequenceClassification.forward.example-4",$$slots:{default:[Wn]},$$scope:{ctx:w}}}),$e=new kn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/ctrl.md"}}),{c(){t=p("meta"),h=r(),o=p("p"),i=r(),M=p("p"),M.innerHTML=n,v=r(),u(ne.$$.fragment),Pe=r(),G=p("div"),G.innerHTML=Pt,Qe=r(),u(se.$$.fragment),De=r(),oe=p("p"),oe.innerHTML=Qt,Oe=r(),ae=p("p"),ae.textContent=Dt,Ae=r(),re=p("p"),re.innerHTML=Ot,Ke=r(),le=p("p"),le.innerHTML=At,et=r(),u(ie.$$.fragment),tt=r(),ce=p("ul"),ce.innerHTML=Kt,nt=r(),u(de.$$.fragment),st=r(),pe=p("ul"),pe.innerHTML=en,ot=r(),u(me.$$.fragment),at=r(),J=p("div"),u(he.$$.fragment),_t=r(),Je=p("p"),Je.innerHTML=tn,Tt=r(),Re=p("p"),Re.innerHTML=nn,bt=r(),u(E.$$.fragment),rt=r(),u(ue.$$.fragment),lt=r(),R=p("div"),u(fe.$$.fragment),Mt=r(),xe=p("p"),xe.textContent=sn,yt=r(),ze=p("p"),ze.innerHTML=on,vt=r(),Ie=p("div"),u(ge.$$.fragment),it=r(),u(_e.$$.fragment),ct=r(),$=p("div"),u(Te.$$.fragment),wt=r(),We=p("p"),We.textContent=an,kt=r(),Ue=p("p"),Ue.innerHTML=rn,Ct=r(),Ze=p("p"),Ze.innerHTML=ln,$t=r(),W=p("div"),u(be.$$.fragment),jt=r(),Fe=p("p"),Fe.innerHTML=cn,Lt=r(),u(Y.$$.fragment),Jt=r(),u(P.$$.fragment),dt=r(),u(Me.$$.fragment),pt=r(),j=p("div"),u(ye.$$.fragment),Rt=r(),Be=p("p"),Be.textContent=dn,xt=r(),qe=p("p"),qe.innerHTML=pn,zt=r(),Ne=p("p"),Ne.innerHTML=mn,It=r(),U=p("div"),u(ve.$$.fragment),Wt=r(),He=p("p"),He.innerHTML=hn,Ut=r(),u(Q.$$.fragment),Zt=r(),u(D.$$.fragment),mt=r(),u(we.$$.fragment),ht=r(),L=p("div"),u(ke.$$.fragment),Ft=r(),Ve=p("p"),Ve.innerHTML=un,Bt=r(),Xe=p("p"),Xe.innerHTML=fn,qt=r(),Se=p("p"),Se.innerHTML=gn,Nt=r(),k=p("div"),u(Ce.$$.fragment),Ht=r(),Ge=p("p"),Ge.innerHTML=_n,Vt=r(),u(O.$$.fragment),Xt=r(),u(A.$$.fragment),St=r(),u(K.$$.fragment),Gt=r(),u(ee.$$.fragment),Et=r(),u(te.$$.fragment),ut=r(),u($e.$$.fragment),ft=r(),Ee=p("p"),this.h()},l(e){const s=wn("svelte-u9bgzb",document.head);t=m(s,"META",{name:!0,content:!0}),s.forEach(a),h=l(e),o=m(e,"P",{}),H(o).forEach(a),i=l(e),M=m(e,"P",{"data-svelte-h":!0}),y(M)!=="svelte-5axdlm"&&(M.innerHTML=n),v=l(e),f(ne.$$.fragment,e),Pe=l(e),G=m(e,"DIV",{class:!0,"data-svelte-h":!0}),y(G)!=="svelte-13t8s2t"&&(G.innerHTML=Pt),Qe=l(e),f(se.$$.fragment,e),De=l(e),oe=m(e,"P",{"data-svelte-h":!0}),y(oe)!=="svelte-12sh13q"&&(oe.innerHTML=Qt),Oe=l(e),ae=m(e,"P",{"data-svelte-h":!0}),y(ae)!=="svelte-vfdo9a"&&(ae.textContent=Dt),Ae=l(e),re=m(e,"P",{"data-svelte-h":!0}),y(re)!=="svelte-f7zzo4"&&(re.innerHTML=Ot),Ke=l(e),le=m(e,"P",{"data-svelte-h":!0}),y(le)!=="svelte-3n853r"&&(le.innerHTML=At),et=l(e),f(ie.$$.fragment,e),tt=l(e),ce=m(e,"UL",{"data-svelte-h":!0}),y(ce)!=="svelte-j79wce"&&(ce.innerHTML=Kt),nt=l(e),f(de.$$.fragment,e),st=l(e),pe=m(e,"UL",{"data-svelte-h":!0}),y(pe)!=="svelte-17u5l9r"&&(pe.innerHTML=en),ot=l(e),f(me.$$.fragment,e),at=l(e),J=m(e,"DIV",{class:!0});var F=H(J);f(he.$$.fragment,F),_t=l(F),Je=m(F,"P",{"data-svelte-h":!0}),y(Je)!=="svelte-bw6jkb"&&(Je.innerHTML=tn),Tt=l(F),Re=m(F,"P",{"data-svelte-h":!0}),y(Re)!=="svelte-1ek1ss9"&&(Re.innerHTML=nn),bt=l(F),f(E.$$.fragment,F),F.forEach(a),rt=l(e),f(ue.$$.fragment,e),lt=l(e),R=m(e,"DIV",{class:!0});var B=H(R);f(fe.$$.fragment,B),Mt=l(B),xe=m(B,"P",{"data-svelte-h":!0}),y(xe)!=="svelte-1ry85wb"&&(xe.textContent=sn),yt=l(B),ze=m(B,"P",{"data-svelte-h":!0}),y(ze)!=="svelte-ntrhio"&&(ze.innerHTML=on),vt=l(B),Ie=m(B,"DIV",{class:!0});var Ye=H(Ie);f(ge.$$.fragment,Ye),Ye.forEach(a),B.forEach(a),it=l(e),f(_e.$$.fragment,e),ct=l(e),$=m(e,"DIV",{class:!0});var x=H($);f(Te.$$.fragment,x),wt=l(x),We=m(x,"P",{"data-svelte-h":!0}),y(We)!=="svelte-5qeehh"&&(We.textContent=an),kt=l(x),Ue=m(x,"P",{"data-svelte-h":!0}),y(Ue)!=="svelte-q52n56"&&(Ue.innerHTML=rn),Ct=l(x),Ze=m(x,"P",{"data-svelte-h":!0}),y(Ze)!=="svelte-hswkmf"&&(Ze.innerHTML=ln),$t=l(x),W=m(x,"DIV",{class:!0});var q=H(W);f(be.$$.fragment,q),jt=l(q),Fe=m(q,"P",{"data-svelte-h":!0}),y(Fe)!=="svelte-12ciudq"&&(Fe.innerHTML=cn),Lt=l(q),f(Y.$$.fragment,q),Jt=l(q),f(P.$$.fragment,q),q.forEach(a),x.forEach(a),dt=l(e),f(Me.$$.fragment,e),pt=l(e),j=m(e,"DIV",{class:!0});var z=H(j);f(ye.$$.fragment,z),Rt=l(z),Be=m(z,"P",{"data-svelte-h":!0}),y(Be)!=="svelte-ej2g0g"&&(Be.textContent=dn),xt=l(z),qe=m(z,"P",{"data-svelte-h":!0}),y(qe)!=="svelte-q52n56"&&(qe.innerHTML=pn),zt=l(z),Ne=m(z,"P",{"data-svelte-h":!0}),y(Ne)!=="svelte-hswkmf"&&(Ne.innerHTML=mn),It=l(z),U=m(z,"DIV",{class:!0});var N=H(U);f(ve.$$.fragment,N),Wt=l(N),He=m(N,"P",{"data-svelte-h":!0}),y(He)!=="svelte-1vxom8i"&&(He.innerHTML=hn),Ut=l(N),f(Q.$$.fragment,N),Zt=l(N),f(D.$$.fragment,N),N.forEach(a),z.forEach(a),mt=l(e),f(we.$$.fragment,e),ht=l(e),L=m(e,"DIV",{class:!0});var I=H(L);f(ke.$$.fragment,I),Ft=l(I),Ve=m(I,"P",{"data-svelte-h":!0}),y(Ve)!=="svelte-14m352y"&&(Ve.innerHTML=un),Bt=l(I),Xe=m(I,"P",{"data-svelte-h":!0}),y(Xe)!=="svelte-q52n56"&&(Xe.innerHTML=fn),qt=l(I),Se=m(I,"P",{"data-svelte-h":!0}),y(Se)!=="svelte-hswkmf"&&(Se.innerHTML=gn),Nt=l(I),k=m(I,"DIV",{class:!0});var C=H(k);f(Ce.$$.fragment,C),Ht=l(C),Ge=m(C,"P",{"data-svelte-h":!0}),y(Ge)!=="svelte-1jgbsu"&&(Ge.innerHTML=_n),Vt=l(C),f(O.$$.fragment,C),Xt=l(C),f(A.$$.fragment,C),St=l(C),f(K.$$.fragment,C),Gt=l(C),f(ee.$$.fragment,C),Et=l(C),f(te.$$.fragment,C),C.forEach(a),I.forEach(a),ut=l(e),f($e.$$.fragment,e),ft=l(e),Ee=m(e,"P",{}),H(Ee).forEach(a),this.h()},h(){Z(t,"name","hf:doc:metadata"),Z(t,"content",Zn),Z(G,"class","flex flex-wrap space-x-1"),Z(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(Ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){d(document.head,t),c(e,h,s),c(e,o,s),c(e,i,s),c(e,M,s),c(e,v,s),g(ne,e,s),c(e,Pe,s),c(e,G,s),c(e,Qe,s),g(se,e,s),c(e,De,s),c(e,oe,s),c(e,Oe,s),c(e,ae,s),c(e,Ae,s),c(e,re,s),c(e,Ke,s),c(e,le,s),c(e,et,s),g(ie,e,s),c(e,tt,s),c(e,ce,s),c(e,nt,s),g(de,e,s),c(e,st,s),c(e,pe,s),c(e,ot,s),g(me,e,s),c(e,at,s),c(e,J,s),g(he,J,null),d(J,_t),d(J,Je),d(J,Tt),d(J,Re),d(J,bt),g(E,J,null),c(e,rt,s),g(ue,e,s),c(e,lt,s),c(e,R,s),g(fe,R,null),d(R,Mt),d(R,xe),d(R,yt),d(R,ze),d(R,vt),d(R,Ie),g(ge,Ie,null),c(e,it,s),g(_e,e,s),c(e,ct,s),c(e,$,s),g(Te,$,null),d($,wt),d($,We),d($,kt),d($,Ue),d($,Ct),d($,Ze),d($,$t),d($,W),g(be,W,null),d(W,jt),d(W,Fe),d(W,Lt),g(Y,W,null),d(W,Jt),g(P,W,null),c(e,dt,s),g(Me,e,s),c(e,pt,s),c(e,j,s),g(ye,j,null),d(j,Rt),d(j,Be),d(j,xt),d(j,qe),d(j,zt),d(j,Ne),d(j,It),d(j,U),g(ve,U,null),d(U,Wt),d(U,He),d(U,Ut),g(Q,U,null),d(U,Zt),g(D,U,null),c(e,mt,s),g(we,e,s),c(e,ht,s),c(e,L,s),g(ke,L,null),d(L,Ft),d(L,Ve),d(L,Bt),d(L,Xe),d(L,qt),d(L,Se),d(L,Nt),d(L,k),g(Ce,k,null),d(k,Ht),d(k,Ge),d(k,Vt),g(O,k,null),d(k,Xt),g(A,k,null),d(k,St),g(K,k,null),d(k,Gt),g(ee,k,null),d(k,Et),g(te,k,null),c(e,ut,s),g($e,e,s),c(e,ft,s),c(e,Ee,s),gt=!0},p(e,[s]){const F={};s&2&&(F.$$scope={dirty:s,ctx:e}),E.$set(F);const B={};s&2&&(B.$$scope={dirty:s,ctx:e}),Y.$set(B);const Ye={};s&2&&(Ye.$$scope={dirty:s,ctx:e}),P.$set(Ye);const x={};s&2&&(x.$$scope={dirty:s,ctx:e}),Q.$set(x);const q={};s&2&&(q.$$scope={dirty:s,ctx:e}),D.$set(q);const z={};s&2&&(z.$$scope={dirty:s,ctx:e}),O.$set(z);const N={};s&2&&(N.$$scope={dirty:s,ctx:e}),A.$set(N);const I={};s&2&&(I.$$scope={dirty:s,ctx:e}),K.$set(I);const C={};s&2&&(C.$$scope={dirty:s,ctx:e}),ee.$set(C);const Tn={};s&2&&(Tn.$$scope={dirty:s,ctx:e}),te.$set(Tn)},i(e){gt||(_(ne.$$.fragment,e),_(se.$$.fragment,e),_(ie.$$.fragment,e),_(de.$$.fragment,e),_(me.$$.fragment,e),_(he.$$.fragment,e),_(E.$$.fragment,e),_(ue.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(Te.$$.fragment,e),_(be.$$.fragment,e),_(Y.$$.fragment,e),_(P.$$.fragment,e),_(Me.$$.fragment,e),_(ye.$$.fragment,e),_(ve.$$.fragment,e),_(Q.$$.fragment,e),_(D.$$.fragment,e),_(we.$$.fragment,e),_(ke.$$.fragment,e),_(Ce.$$.fragment,e),_(O.$$.fragment,e),_(A.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_($e.$$.fragment,e),gt=!0)},o(e){T(ne.$$.fragment,e),T(se.$$.fragment,e),T(ie.$$.fragment,e),T(de.$$.fragment,e),T(me.$$.fragment,e),T(he.$$.fragment,e),T(E.$$.fragment,e),T(ue.$$.fragment,e),T(fe.$$.fragment,e),T(ge.$$.fragment,e),T(_e.$$.fragment,e),T(Te.$$.fragment,e),T(be.$$.fragment,e),T(Y.$$.fragment,e),T(P.$$.fragment,e),T(Me.$$.fragment,e),T(ye.$$.fragment,e),T(ve.$$.fragment,e),T(Q.$$.fragment,e),T(D.$$.fragment,e),T(we.$$.fragment,e),T(ke.$$.fragment,e),T(Ce.$$.fragment,e),T(O.$$.fragment,e),T(A.$$.fragment,e),T(K.$$.fragment,e),T(ee.$$.fragment,e),T(te.$$.fragment,e),T($e.$$.fragment,e),gt=!1},d(e){e&&(a(h),a(o),a(i),a(M),a(v),a(Pe),a(G),a(Qe),a(De),a(oe),a(Oe),a(ae),a(Ae),a(re),a(Ke),a(le),a(et),a(tt),a(ce),a(nt),a(st),a(pe),a(ot),a(at),a(J),a(rt),a(lt),a(R),a(it),a(ct),a($),a(dt),a(pt),a(j),a(mt),a(ht),a(L),a(ut),a(ft),a(Ee)),a(t),b(ne,e),b(se,e),b(ie,e),b(de,e),b(me,e),b(he),b(E),b(ue,e),b(fe),b(ge),b(_e,e),b(Te),b(be),b(Y),b(P),b(Me,e),b(ye),b(ve),b(Q),b(D),b(we,e),b(ke),b(Ce),b(O),b(A),b(K),b(ee),b(te),b($e,e)}}}const Zn='{"title":"CTRL","local":"ctrl","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"CTRLConfig","local":"transformers.CTRLConfig","sections":[],"depth":2},{"title":"CTRLTokenizer","local":"transformers.CTRLTokenizer","sections":[],"depth":2},{"title":"CTRLModel","local":"transformers.CTRLModel","sections":[],"depth":2},{"title":"CTRLLMHeadModel","local":"transformers.CTRLLMHeadModel","sections":[],"depth":2},{"title":"CTRLForSequenceClassification","local":"transformers.CTRLForSequenceClassification","sections":[],"depth":2}],"depth":1}';function Fn(w){return Mn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Gn extends yn{constructor(t){super(),vn(this,t,Fn,Un,bn,{})}}export{Gn as component};
