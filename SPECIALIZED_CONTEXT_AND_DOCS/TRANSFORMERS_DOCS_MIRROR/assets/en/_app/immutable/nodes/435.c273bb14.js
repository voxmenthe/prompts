import{s as Fs,z as Us,o as Js,n as B}from"../chunks/scheduler.18a86fab.js";import{S as Ws,i as Zs,g as l,s as r,r as h,A as Ps,h as d,f as s,c as i,j as x,x as y,u as g,k,y as a,a as c,v as f,d as u,t as _,w as b}from"../chunks/index.98837b22.js";import{T as to}from"../chunks/Tip.77304350.js";import{D as V}from"../chunks/Docstring.a1ef7999.js";import{C as yt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as bt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as U,E as Bs}from"../chunks/getInferenceSnippets.06c2775f.js";function Rs(w){let t,T="Example:",m,p,v;return p=new yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFZpTFRNb2RlbCUyQyUyMFZpTFRDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwVmlMVCUyMGRhbmRlbGluJTJGdmlsdC1iMzItbWxtJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFZpTFRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBkYW5kZWxpbiUyRnZpbHQtYjMyLW1sbSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwVmlMVE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViLTModel, ViLTConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a ViLT dandelin/vilt-b32-mlm style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = ViLTConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the dandelin/vilt-b32-mlm style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViLTModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,m=r(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=T),m=i(o),g(p.$$.fragment,o)},m(o,M){c(o,t,M),c(o,m,M),f(p,o,M),v=!0},p:B,i(o){v||(u(p.$$.fragment,o),v=!0)},o(o){_(p.$$.fragment,o),v=!1},d(o){o&&(s(t),s(m)),b(p,o)}}}function Ns(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(m){t=d(m,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(m,p){c(m,t,p)},p:B,d(m){m&&s(t)}}}function Ls(w){let t,T="Examples:",m,p,v;return p=new yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFZpbHRQcm9jZXNzb3IlMkMlMjBWaWx0TW9kZWwlMEFmcm9tJTIwUElMJTIwaW1wb3J0JTIwSW1hZ2UlMEFpbXBvcnQlMjByZXF1ZXN0cyUwQSUwQSUyMyUyMHByZXBhcmUlMjBpbWFnZSUyMGFuZCUyMHRleHQlMEF1cmwlMjAlM0QlMjAlMjJodHRwJTNBJTJGJTJGaW1hZ2VzLmNvY29kYXRhc2V0Lm9yZyUyRnZhbDIwMTclMkYwMDAwMDAwMzk3NjkuanBnJTIyJTBBaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBdGV4dCUyMCUzRCUyMCUyMmhlbGxvJTIwd29ybGQlMjIlMEElMEFwcm9jZXNzb3IlMjAlM0QlMjBWaWx0UHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZCglMjJkYW5kZWxpbiUyRnZpbHQtYjMyLW1sbSUyMiklMEFtb2RlbCUyMCUzRCUyMFZpbHRNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyZGFuZGVsaW4lMkZ2aWx0LWIzMi1tbG0lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKGltYWdlJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare image and text</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;hello world&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltModel.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(image, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,m=r(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=T),m=i(o),g(p.$$.fragment,o)},m(o,M){c(o,t,M),c(o,m,M),f(p,o,M),v=!0},p:B,i(o){v||(u(p.$$.fragment,o),v=!0)},o(o){_(p.$$.fragment,o),v=!1},d(o){o&&(s(t),s(m)),b(p,o)}}}function As(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(m){t=d(m,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(m,p){c(m,t,p)},p:B,d(m){m&&s(t)}}}function Gs(w){let t,T="Examples:",m,p,v;return p=new yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFZpbHRQcm9jZXNzb3IlMkMlMjBWaWx0Rm9yTWFza2VkTE0lMEFpbXBvcnQlMjByZXF1ZXN0cyUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHJlJTBBaW1wb3J0JTIwdG9yY2glMEElMEF1cmwlMjAlM0QlMjAlMjJodHRwJTNBJTJGJTJGaW1hZ2VzLmNvY29kYXRhc2V0Lm9yZyUyRnZhbDIwMTclMkYwMDAwMDAwMzk3NjkuanBnJTIyJTBBaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBdGV4dCUyMCUzRCUyMCUyMmElMjBidW5jaCUyMG9mJTIwJTVCTUFTSyU1RCUyMGxheWluZyUyMG9uJTIwYSUyMCU1Qk1BU0slNUQuJTIyJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwVmlsdFByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyZGFuZGVsaW4lMkZ2aWx0LWIzMi1tbG0lMjIpJTBBbW9kZWwlMjAlM0QlMjBWaWx0Rm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmRhbmRlbGluJTJGdmlsdC1iMzItbWxtJTIyKSUwQSUwQSUyMyUyMHByZXBhcmUlMjBpbnB1dHMlMEFlbmNvZGluZyUyMCUzRCUyMHByb2Nlc3NvcihpbWFnZSUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMGZvcndhcmQlMjBwYXNzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmcpJTBBJTBBdGwlMjAlM0QlMjBsZW4ocmUuZmluZGFsbCglMjIlNUMlNUJNQVNLJTVDJTVEJTIyJTJDJTIwdGV4dCkpJTBBaW5mZXJyZWRfdG9rZW4lMjAlM0QlMjAlNUJ0ZXh0JTVEJTBBJTBBJTIzJTIwZ3JhZHVhbGx5JTIwZmlsbCUyMGluJTIwdGhlJTIwTUFTSyUyMHRva2VucyUyQyUyMG9uZSUyMGJ5JTIwb25lJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGZvciUyMGklMjBpbiUyMHJhbmdlKHRsKSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGVuY29kZWQlMjAlM0QlMjBwcm9jZXNzb3IudG9rZW5pemVyKGluZmVycmVkX3Rva2VuKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGlucHV0X2lkcyUyMCUzRCUyMHRvcmNoLnRlbnNvcihlbmNvZGVkLmlucHV0X2lkcyklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBlbmNvZGVkJTIwJTNEJTIwZW5jb2RlZCU1QiUyMmlucHV0X2lkcyUyMiU1RCU1QjAlNUQlNUIxJTNBLTElNUQlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwcGl4ZWxfdmFsdWVzJTNEZW5jb2RpbmcucGl4ZWxfdmFsdWVzKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMG1sbV9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyU1QjAlNUQlMjAlMjAlMjMlMjBzaGFwZSUyMChzZXFfbGVuJTJDJTIwdm9jYWJfc2l6ZSklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjMlMjBvbmx5JTIwdGFrZSUyMGludG8lMjBhY2NvdW50JTIwdGV4dCUyMGZlYXR1cmVzJTIwKG1pbnVzJTIwQ0xTJTIwYW5kJTIwU0VQJTIwdG9rZW4pJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwbWxtX2xvZ2l0cyUyMCUzRCUyMG1sbV9sb2dpdHMlNUIxJTIwJTNBJTIwaW5wdXRfaWRzLnNoYXBlJTVCMSU1RCUyMC0lMjAxJTJDJTIwJTNBJTVEJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwbWxtX3ZhbHVlcyUyQyUyMG1sbV9pZHMlMjAlM0QlMjBtbG1fbG9naXRzLnNvZnRtYXgoZGltJTNELTEpLm1heChkaW0lM0QtMSklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjMlMjBvbmx5JTIwdGFrZSUyMGludG8lMjBhY2NvdW50JTIwdGV4dCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMG1sbV92YWx1ZXMlNUJ0b3JjaC50ZW5zb3IoZW5jb2RlZCklMjAhJTNEJTIwMTAzJTVEJTIwJTNEJTIwMCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHNlbGVjdCUyMCUzRCUyMG1sbV92YWx1ZXMuYXJnbWF4KCkuaXRlbSgpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZW5jb2RlZCU1QnNlbGVjdCU1RCUyMCUzRCUyMG1sbV9pZHMlNUJzZWxlY3QlNUQuaXRlbSgpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwaW5mZXJyZWRfdG9rZW4lMjAlM0QlMjAlNUJwcm9jZXNzb3IuZGVjb2RlKGVuY29kZWQpJTVEJTBBJTBBc2VsZWN0ZWRfdG9rZW4lMjAlM0QlMjAlMjIlMjIlMEFlbmNvZGVkJTIwJTNEJTIwcHJvY2Vzc29yLnRva2VuaXplcihpbmZlcnJlZF90b2tlbiklMEFvdXRwdXQlMjAlM0QlMjBwcm9jZXNzb3IuZGVjb2RlKGVuY29kZWQuaW5wdXRfaWRzJTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQXByaW50KG91dHB1dCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> re
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;a bunch of [MASK] laying on a [MASK].&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForMaskedLM.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare inputs</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(image, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)

<span class="hljs-meta">&gt;&gt;&gt; </span>tl = <span class="hljs-built_in">len</span>(re.findall(<span class="hljs-string">&quot;\\[MASK\\]&quot;</span>, text))
<span class="hljs-meta">&gt;&gt;&gt; </span>inferred_token = [text]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># gradually fill in the MASK tokens, one by one</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(tl):
<span class="hljs-meta">... </span>        encoded = processor.tokenizer(inferred_token)
<span class="hljs-meta">... </span>        input_ids = torch.tensor(encoded.input_ids)
<span class="hljs-meta">... </span>        encoded = encoded[<span class="hljs-string">&quot;input_ids&quot;</span>][<span class="hljs-number">0</span>][<span class="hljs-number">1</span>:-<span class="hljs-number">1</span>]
<span class="hljs-meta">... </span>        outputs = model(input_ids=input_ids, pixel_values=encoding.pixel_values)
<span class="hljs-meta">... </span>        mlm_logits = outputs.logits[<span class="hljs-number">0</span>]  <span class="hljs-comment"># shape (seq_len, vocab_size)</span>
<span class="hljs-meta">... </span>        <span class="hljs-comment"># only take into account text features (minus CLS and SEP token)</span>
<span class="hljs-meta">... </span>        mlm_logits = mlm_logits[<span class="hljs-number">1</span> : input_ids.shape[<span class="hljs-number">1</span>] - <span class="hljs-number">1</span>, :]
<span class="hljs-meta">... </span>        mlm_values, mlm_ids = mlm_logits.softmax(dim=-<span class="hljs-number">1</span>).<span class="hljs-built_in">max</span>(dim=-<span class="hljs-number">1</span>)
<span class="hljs-meta">... </span>        <span class="hljs-comment"># only take into account text</span>
<span class="hljs-meta">... </span>        mlm_values[torch.tensor(encoded) != <span class="hljs-number">103</span>] = <span class="hljs-number">0</span>
<span class="hljs-meta">... </span>        select = mlm_values.argmax().item()
<span class="hljs-meta">... </span>        encoded[select] = mlm_ids[select].item()
<span class="hljs-meta">... </span>        inferred_token = [processor.decode(encoded)]

<span class="hljs-meta">&gt;&gt;&gt; </span>selected_token = <span class="hljs-string">&quot;&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoded = processor.tokenizer(inferred_token)
<span class="hljs-meta">&gt;&gt;&gt; </span>output = processor.decode(encoded.input_ids[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(output)
a bunch of cats laying on a couch.`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,m=r(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=T),m=i(o),g(p.$$.fragment,o)},m(o,M){c(o,t,M),c(o,m,M),f(p,o,M),v=!0},p:B,i(o){v||(u(p.$$.fragment,o),v=!0)},o(o){_(p.$$.fragment,o),v=!1},d(o){o&&(s(t),s(m)),b(p,o)}}}function Hs(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(m){t=d(m,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(m,p){c(m,t,p)},p:B,d(m){m&&s(t)}}}function qs(w){let t,T="Examples:",m,p,v;return p=new yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFZpbHRQcm9jZXNzb3IlMkMlMjBWaWx0Rm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjByZXF1ZXN0cyUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQSUwQXVybCUyMCUzRCUyMCUyMmh0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEF0ZXh0JTIwJTNEJTIwJTIySG93JTIwbWFueSUyMGNhdHMlMjBhcmUlMjB0aGVyZSUzRiUyMiUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMFZpbHRQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMmRhbmRlbGluJTJGdmlsdC1iMzItZmluZXR1bmVkLXZxYSUyMiklMEFtb2RlbCUyMCUzRCUyMFZpbHRGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyZGFuZGVsaW4lMkZ2aWx0LWIzMi1maW5ldHVuZWQtdnFhJTIyKSUwQSUwQSUyMyUyMHByZXBhcmUlMjBpbnB1dHMlMEFlbmNvZGluZyUyMCUzRCUyMHByb2Nlc3NvcihpbWFnZSUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMGZvcndhcmQlMjBwYXNzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmcpJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEFpZHglMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKS5pdGVtKCklMEFwcmludCglMjJQcmVkaWN0ZWQlMjBhbnN3ZXIlM0ElMjIlMkMlMjBtb2RlbC5jb25maWcuaWQybGFiZWwlNUJpZHglNUQp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;How many cats are there?&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-vqa&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-vqa&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare inputs</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(image, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>idx = logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Predicted answer:&quot;</span>, model.config.id2label[idx])
Predicted answer: <span class="hljs-number">2</span>`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,m=r(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=T),m=i(o),g(p.$$.fragment,o)},m(o,M){c(o,t,M),c(o,m,M),f(p,o,M),v=!0},p:B,i(o){v||(u(p.$$.fragment,o),v=!0)},o(o){_(p.$$.fragment,o),v=!1},d(o){o&&(s(t),s(m)),b(p,o)}}}function Ss(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(m){t=d(m,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(m,p){c(m,t,p)},p:B,d(m){m&&s(t)}}}function Es(w){let t,T="Examples:",m,p,v;return p=new yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFZpbHRQcm9jZXNzb3IlMkMlMjBWaWx0Rm9ySW1hZ2VzQW5kVGV4dENsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwcmVxdWVzdHMlMEFmcm9tJTIwUElMJTIwaW1wb3J0JTIwSW1hZ2UlMEElMEFpbWFnZTElMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCglMjJodHRwcyUzQSUyRiUyRmxpbC5ubHAuY29ybmVsbC5lZHUlMkZubHZyJTJGZXhzJTJGZXgwXzAuanBnJTIyJTJDJTIwc3RyZWFtJTNEVHJ1ZSkucmF3KSUwQWltYWdlMiUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KCUyMmh0dHBzJTNBJTJGJTJGbGlsLm5scC5jb3JuZWxsLmVkdSUyRm5sdnIlMkZleHMlMkZleDBfMS5qcGclMjIlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBdGV4dCUyMCUzRCUyMCUyMlRoZSUyMGxlZnQlMjBpbWFnZSUyMGNvbnRhaW5zJTIwdHdpY2UlMjB0aGUlMjBudW1iZXIlMjBvZiUyMGRvZ3MlMjBhcyUyMHRoZSUyMHJpZ2h0JTIwaW1hZ2UuJTIyJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwVmlsdFByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyZGFuZGVsaW4lMkZ2aWx0LWIzMi1maW5ldHVuZWQtbmx2cjIlMjIpJTBBbW9kZWwlMjAlM0QlMjBWaWx0Rm9ySW1hZ2VzQW5kVGV4dENsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJkYW5kZWxpbiUyRnZpbHQtYjMyLWZpbmV0dW5lZC1ubHZyMiUyMiklMEElMEElMjMlMjBwcmVwYXJlJTIwaW5wdXRzJTBBZW5jb2RpbmclMjAlM0QlMjBwcm9jZXNzb3IoJTVCaW1hZ2UxJTJDJTIwaW1hZ2UyJTVEJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwZm9yd2FyZCUyMHBhc3MlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEZW5jb2RpbmcuaW5wdXRfaWRzJTJDJTIwcGl4ZWxfdmFsdWVzJTNEZW5jb2RpbmcucGl4ZWxfdmFsdWVzLnVuc3F1ZWV6ZSgwKSklMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQWlkeCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpLml0ZW0oKSUwQXByaW50KCUyMlByZWRpY3RlZCUyMGFuc3dlciUzQSUyMiUyQyUyMG1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QmlkeCU1RCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltForImagesAndTextClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span>image1 = Image.<span class="hljs-built_in">open</span>(requests.get(<span class="hljs-string">&quot;https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg&quot;</span>, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>image2 = Image.<span class="hljs-built_in">open</span>(requests.get(<span class="hljs-string">&quot;https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg&quot;</span>, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;The left image contains twice the number of dogs as the right image.&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-nlvr2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForImagesAndTextClassification.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-nlvr2&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare inputs</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor([image1, image2], text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(<span class="hljs-number">0</span>))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>idx = logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Predicted answer:&quot;</span>, model.config.id2label[idx])
Predicted answer: <span class="hljs-literal">True</span>`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,m=r(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=T),m=i(o),g(p.$$.fragment,o)},m(o,M){c(o,t,M),c(o,m,M),f(p,o,M),v=!0},p:B,i(o){v||(u(p.$$.fragment,o),v=!0)},o(o){_(p.$$.fragment,o),v=!1},d(o){o&&(s(t),s(m)),b(p,o)}}}function Qs(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(m){t=d(m,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(m,p){c(m,t,p)},p:B,d(m){m&&s(t)}}}function Xs(w){let t,T="Examples:",m,p,v;return p=new yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFZpbHRQcm9jZXNzb3IlMkMlMjBWaWx0Rm9ySW1hZ2VBbmRUZXh0UmV0cmlldmFsJTBBaW1wb3J0JTIwcmVxdWVzdHMlMEFmcm9tJTIwUElMJTIwaW1wb3J0JTIwSW1hZ2UlMEElMEF1cmwlMjAlM0QlMjAlMjJodHRwJTNBJTJGJTJGaW1hZ2VzLmNvY29kYXRhc2V0Lm9yZyUyRnZhbDIwMTclMkYwMDAwMDAwMzk3NjkuanBnJTIyJTBBaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBdGV4dHMlMjAlM0QlMjAlNUIlMjJBbiUyMGltYWdlJTIwb2YlMjB0d28lMjBjYXRzJTIwY2hpbGxpbmclMjBvbiUyMGElMjBjb3VjaCUyMiUyQyUyMCUyMkElMjBmb290YmFsbCUyMHBsYXllciUyMHNjb3JpbmclMjBhJTIwZ29hbCUyMiU1RCUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMFZpbHRQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMmRhbmRlbGluJTJGdmlsdC1iMzItZmluZXR1bmVkLWNvY28lMjIpJTBBbW9kZWwlMjAlM0QlMjBWaWx0Rm9ySW1hZ2VBbmRUZXh0UmV0cmlldmFsLmZyb21fcHJldHJhaW5lZCglMjJkYW5kZWxpbiUyRnZpbHQtYjMyLWZpbmV0dW5lZC1jb2NvJTIyKSUwQSUwQSUyMyUyMGZvcndhcmQlMjBwYXNzJTBBc2NvcmVzJTIwJTNEJTIwZGljdCgpJTBBZm9yJTIwdGV4dCUyMGluJTIwdGV4dHMlM0ElMEElMjAlMjAlMjAlMjAlMjMlMjBwcmVwYXJlJTIwaW5wdXRzJTBBJTIwJTIwJTIwJTIwZW5jb2RpbmclMjAlM0QlMjBwcm9jZXNzb3IoaW1hZ2UlMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiplbmNvZGluZyklMEElMjAlMjAlMjAlMjBzY29yZXMlNUJ0ZXh0JTVEJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlNUIwJTJDJTIwJTNBJTVELml0ZW0oKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ViltProcessor, ViltForImageAndTextRetrieval
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>texts = [<span class="hljs-string">&quot;An image of two cats chilling on a couch&quot;</span>, <span class="hljs-string">&quot;A football player scoring a goal&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = ViltProcessor.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-coco&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForImageAndTextRetrieval.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-finetuned-coco&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>scores = <span class="hljs-built_in">dict</span>()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> text <span class="hljs-keyword">in</span> texts:
<span class="hljs-meta">... </span>    <span class="hljs-comment"># prepare inputs</span>
<span class="hljs-meta">... </span>    encoding = processor(image, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">... </span>    outputs = model(**encoding)
<span class="hljs-meta">... </span>    scores[text] = outputs.logits[<span class="hljs-number">0</span>, :].item()`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,m=r(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=T),m=i(o),g(p.$$.fragment,o)},m(o,M){c(o,t,M),c(o,m,M),f(p,o,M),v=!0},p:B,i(o){v||(u(p.$$.fragment,o),v=!0)},o(o){_(p.$$.fragment,o),v=!1},d(o){o&&(s(t),s(m)),b(p,o)}}}function Os(w){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(m){t=d(m,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(m,p){c(m,t,p)},p:B,d(m){m&&s(t)}}}function Ys(w){let t,T="Example:",m,p,v;return p=new yt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBWaWx0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZGFuZGVsaW4lMkZ2aWx0LWIzMi1tbG0lMjIpJTBBbW9kZWwlMjAlM0QlMjBWaWx0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZGFuZGVsaW4lMkZ2aWx0LWIzMi1tbG0lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ViltForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ViltForTokenClassification.from_pretrained(<span class="hljs-string">&quot;dandelin/vilt-b32-mlm&quot;</span>)

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
...`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,m=r(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=T),m=i(o),g(p.$$.fragment,o)},m(o,M){c(o,t,M),c(o,m,M),f(p,o,M),v=!0},p:B,i(o){v||(u(p.$$.fragment,o),v=!0)},o(o){_(p.$$.fragment,o),v=!1},d(o){o&&(s(t),s(m)),b(p,o)}}}function Ds(w){let t,T,m,p,v,o="<em>This model was released on 2021-02-05 and added to Hugging Face Transformers on 2022-01-19.</em>",M,je,so,re,Hn='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',ao,$e,ro,Ie,qn=`The ViLT model was proposed in <a href="https://huggingface.co/papers/2102.03334" rel="nofollow">ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision</a>
by Wonjae Kim, Bokyung Son, Ildoo Kim. ViLT incorporates text embeddings into a Vision Transformer (ViT), allowing it to have a minimal design
for Vision-and-Language Pre-training (VLP).`,io,Ce,Sn="The abstract from the paper is the following:",lo,ze,En=`<em>Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks.
Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision
(e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we
find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more
computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive
power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model,
Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically
simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of
times faster than previous VLP models, yet with competitive or better downstream task performance.</em>`,co,ie,Qn,mo,Fe,Xn='ViLT architecture. Taken from the <a href="https://huggingface.co/papers/2102.03334">original paper</a>.',po,Ue,On='This model was contributed by <a href="https://huggingface.co/nielsr" rel="nofollow">nielsr</a>. The original code can be found <a href="https://github.com/dandelin/ViLT" rel="nofollow">here</a>.',ho,Je,go,We,Yn=`<li>The quickest way to get started with ViLT is by checking the <a href="https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViLT" rel="nofollow">example notebooks</a>
(which showcase both inference and fine-tuning on custom data).</li> <li>ViLT is a model that takes both <code>pixel_values</code> and <code>input_ids</code> as input. One can use <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> to prepare data for the model.
This processor wraps a image processor (for the image modality) and a tokenizer (for the language modality) into one.</li> <li>ViLT is trained with images of various sizes: the authors resize the shorter edge of input images to 384 and limit the longer edge to
under 640 while preserving the aspect ratio. To make batching of images possible, the authors use a <code>pixel_mask</code> that indicates
which pixel values are real and which are padding. <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> automatically creates this for you.</li> <li>The design of ViLT is very similar to that of a standard Vision Transformer (ViT). The only difference is that the model includes
additional embedding layers for the language modality.</li> <li>The PyTorch version of this model is only available in torch 1.10 and higher.</li>`,fo,Ze,uo,J,Pe,No,Tt,Dn=`This is the configuration class to store the configuration of a <code>ViLTModel</code>. It is used to instantiate an ViLT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the ViLT
<a href="https://huggingface.co/dandelin/vilt-b32-mlm" rel="nofollow">dandelin/vilt-b32-mlm</a> architecture.`,Lo,vt,Kn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ao,le,_o,Be,bo,oe,Re,Go,de,Ne,Ho,Mt,es="Preprocess an image or a batch of images.",yo,Le,To,q,Ae,qo,wt,ts="Constructs a ViLT image processor.",So,ce,Ge,Eo,kt,os="Preprocess an image or batch of images.",vo,He,Mo,S,qe,Qo,xt,ns="Constructs a fast Vilt image processor.",Xo,Vt,Se,wo,Ee,ko,W,Qe,Oo,jt,ss="Constructs a ViLT processor which wraps a BERT tokenizer and ViLT image processor into a single processor.",Yo,$t,as=`<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> offers all the functionalities of <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast">BertTokenizerFast</a>. See the
docstring of <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor.__call__"><strong>call</strong>()</a> and <a href="/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode">decode()</a> for more information.`,Do,Y,Xe,Ko,It,rs=`This method uses <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ViltImageProcessor.<strong>call</strong>()</a> method to prepare image(s) for the model, and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">BertTokenizerFast.<strong>call</strong>()</a> to prepare text for the model.`,en,Ct,is="Please refer to the docstring of the above two methods for more information.",xo,Oe,Vo,j,Ye,tn,zt,ls="The bare Vilt Model outputting raw hidden-states without any specific head on top.",on,Ft,ds=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,nn,Ut,cs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,sn,R,De,an,Jt,ms='The <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltModel">ViltModel</a> forward method, overrides the <code>__call__</code> special method.',rn,me,ln,pe,jo,Ke,$o,$,et,dn,Wt,ps="ViLT Model with a language modeling head on top as done during pretraining.",cn,Zt,hs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,mn,Pt,gs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,pn,N,tt,hn,Bt,fs='The <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForMaskedLM">ViltForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',gn,he,fn,ge,Io,ot,Co,I,nt,un,Rt,us=`Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
token) for visual question answering, e.g. for VQAv2.`,_n,Nt,_s=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,bn,Lt,bs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,yn,L,st,Tn,At,ys='The <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForQuestionAnswering">ViltForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',vn,fe,Mn,ue,zo,at,Fo,C,rt,wn,Gt,Ts="Vilt Model transformer with a classifier head on top for natural language visual reasoning, e.g. NLVR2.",kn,Ht,vs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xn,qt,Ms=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Vn,A,it,jn,St,ws='The <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForImagesAndTextClassification">ViltForImagesAndTextClassification</a> forward method, overrides the <code>__call__</code> special method.',$n,_e,In,be,Uo,lt,Jo,z,dt,Cn,Et,ks=`Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
token) for image-to-text or text-to-image retrieval, e.g. MSCOCO and F30K.`,zn,Qt,xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Fn,Xt,Vs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Un,G,ct,Jn,Ot,js='The <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForImageAndTextRetrieval">ViltForImageAndTextRetrieval</a> forward method, overrides the <code>__call__</code> special method.',Wn,ye,Zn,Te,Wo,mt,Zo,F,pt,Pn,Yt,$s=`The Vilt transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Bn,Dt,Is=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rn,Kt,Cs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Nn,H,ht,Ln,eo,zs='The <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForTokenClassification">ViltForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',An,ve,Gn,Me,Po,gt,Bo,oo,Ro;return je=new U({props:{title:"ViLT",local:"vilt",headingTag:"h1"}}),$e=new U({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Je=new U({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Ze=new U({props:{title:"ViltConfig",local:"transformers.ViltConfig",headingTag:"h2"}}),Pe=new V({props:{name:"class transformers.ViltConfig",anchor:"transformers.ViltConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"type_vocab_size",val:" = 2"},{name:"modality_type_vocab_size",val:" = 2"},{name:"max_position_embeddings",val:" = 40"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.0"},{name:"attention_probs_dropout_prob",val:" = 0.0"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"image_size",val:" = 384"},{name:"patch_size",val:" = 32"},{name:"num_channels",val:" = 3"},{name:"qkv_bias",val:" = True"},{name:"max_image_length",val:" = -1"},{name:"tie_word_embeddings",val:" = False"},{name:"num_images",val:" = -1"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ViltConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the text part of the model. Defines the number of different tokens that can be
represented by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltModel">ViltModel</a>.`,name:"vocab_size"},{anchor:"transformers.ViltConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltModel">ViltModel</a>. This is used when encoding
text.`,name:"type_vocab_size"},{anchor:"transformers.ViltConfig.modality_type_vocab_size",description:`<strong>modality_type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the modalities passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltModel">ViltModel</a>. This is used after concatenating the
embeddings of the text and image modalities.`,name:"modality_type_vocab_size"},{anchor:"transformers.ViltConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 40) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.ViltConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.ViltConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.ViltConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.ViltConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.ViltConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.ViltConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.ViltConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.ViltConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.ViltConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.ViltConfig.image_size",description:`<strong>image_size</strong> (<code>int</code>, <em>optional</em>, defaults to 384) &#x2014;
The size (resolution) of each image.`,name:"image_size"},{anchor:"transformers.ViltConfig.patch_size",description:`<strong>patch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The size (resolution) of each patch.`,name:"patch_size"},{anchor:"transformers.ViltConfig.num_channels",description:`<strong>num_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The number of input channels.`,name:"num_channels"},{anchor:"transformers.ViltConfig.qkv_bias",description:`<strong>qkv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a bias to the queries, keys and values.`,name:"qkv_bias"},{anchor:"transformers.ViltConfig.max_image_length",description:`<strong>max_image_length</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
The maximum number of patches to take as input for the Transformer encoder. If set to a positive integer,
the encoder will sample <code>max_image_length</code> patches at maximum. If set to -1, will not be taken into
account.`,name:"max_image_length"},{anchor:"transformers.ViltConfig.num_images",description:`<strong>num_images</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
The number of images to use for natural language visual reasoning. If set to a positive integer, will be
used by <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForImagesAndTextClassification">ViltForImagesAndTextClassification</a> for defining the classifier head.`,name:"num_images"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/configuration_vilt.py#L24"}}),le=new bt({props:{anchor:"transformers.ViltConfig.example",$$slots:{default:[Rs]},$$scope:{ctx:w}}}),Be=new U({props:{title:"ViltFeatureExtractor",local:"transformers.ViltFeatureExtractor",headingTag:"h2"}}),Re=new V({props:{name:"class transformers.ViltFeatureExtractor",anchor:"transformers.ViltFeatureExtractor",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/feature_extraction_vilt.py#L28"}}),Ne=new V({props:{name:"__call__",anchor:"transformers.ViltFeatureExtractor.__call__",parameters:[{name:"images",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49"}}),Le=new U({props:{title:"ViltImageProcessor",local:"transformers.ViltImageProcessor",headingTag:"h2"}}),Ae=new V({props:{name:"class transformers.ViltImageProcessor",anchor:"transformers.ViltImageProcessor",parameters:[{name:"do_resize",val:": bool = True"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"size_divisor",val:": int = 32"},{name:"resample",val:": Resampling = <Resampling.BICUBIC: 3>"},{name:"do_rescale",val:": bool = True"},{name:"rescale_factor",val:": typing.Union[int, float] = 0.00392156862745098"},{name:"do_normalize",val:": bool = True"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"do_pad",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ViltImageProcessor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the image&#x2019;s (height, width) dimensions to the specified <code>size</code>. Can be overridden by the
<code>do_resize</code> parameter in the <code>preprocess</code> method.`,name:"do_resize"},{anchor:"transformers.ViltImageProcessor.size",description:`<strong>size</strong> (<code>dict[str, int]</code> <em>optional</em>, defaults to <code>{&quot;shortest_edge&quot; -- 384}</code>):
Resize the shorter side of the input to <code>size[&quot;shortest_edge&quot;]</code>. The longer side will be limited to under
<code>int((1333 / 800) * size[&quot;shortest_edge&quot;])</code> while preserving the aspect ratio. Only has an effect if
<code>do_resize</code> is set to <code>True</code>. Can be overridden by the <code>size</code> parameter in the <code>preprocess</code> method.`,name:"size"},{anchor:"transformers.ViltImageProcessor.size_divisor",description:`<strong>size_divisor</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The size by which to make sure both the height and width can be divided. Only has an effect if <code>do_resize</code>
is set to <code>True</code>. Can be overridden by the <code>size_divisor</code> parameter in the <code>preprocess</code> method.`,name:"size_divisor"},{anchor:"transformers.ViltImageProcessor.resample",description:`<strong>resample</strong> (<code>PILImageResampling</code>, <em>optional</em>, defaults to <code>Resampling.BICUBIC</code>) &#x2014;
Resampling filter to use if resizing the image. Only has an effect if <code>do_resize</code> is set to <code>True</code>. Can be
overridden by the <code>resample</code> parameter in the <code>preprocess</code> method.`,name:"resample"},{anchor:"transformers.ViltImageProcessor.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Wwhether to rescale the image by the specified scale <code>rescale_factor</code>. Can be overridden by the
<code>do_rescale</code> parameter in the <code>preprocess</code> method.`,name:"do_rescale"},{anchor:"transformers.ViltImageProcessor.rescale_factor",description:`<strong>rescale_factor</strong> (<code>int</code> or <code>float</code>, <em>optional</em>, defaults to <code>1/255</code>) &#x2014;
Scale factor to use if rescaling the image. Only has an effect if <code>do_rescale</code> is set to <code>True</code>. Can be
overridden by the <code>rescale_factor</code> parameter in the <code>preprocess</code> method.`,name:"rescale_factor"},{anchor:"transformers.ViltImageProcessor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize the image. Can be overridden by the <code>do_normalize</code> parameter in the <code>preprocess</code>
method. Can be overridden by the <code>do_normalize</code> parameter in the <code>preprocess</code> method.`,name:"do_normalize"},{anchor:"transformers.ViltImageProcessor.image_mean",description:`<strong>image_mean</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>IMAGENET_STANDARD_MEAN</code>) &#x2014;
Mean to use if normalizing the image. This is a float or list of floats the length of the number of
channels in the image. Can be overridden by the <code>image_mean</code> parameter in the <code>preprocess</code> method. Can be
overridden by the <code>image_mean</code> parameter in the <code>preprocess</code> method.`,name:"image_mean"},{anchor:"transformers.ViltImageProcessor.image_std",description:`<strong>image_std</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>IMAGENET_STANDARD_STD</code>) &#x2014;
Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
number of channels in the image. Can be overridden by the <code>image_std</code> parameter in the <code>preprocess</code> method.
Can be overridden by the <code>image_std</code> parameter in the <code>preprocess</code> method.`,name:"image_std"},{anchor:"transformers.ViltImageProcessor.do_pad",description:`<strong>do_pad</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to pad the image to the <code>(max_height, max_width)</code> of the images in the batch. Can be overridden by
the <code>do_pad</code> parameter in the <code>preprocess</code> method.`,name:"do_pad"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/image_processing_vilt.py#L124"}}),Ge=new V({props:{name:"preprocess",anchor:"transformers.ViltImageProcessor.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"do_resize",val:": typing.Optional[bool] = None"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"size_divisor",val:": typing.Optional[int] = None"},{name:"resample",val:": Resampling = None"},{name:"do_rescale",val:": typing.Optional[bool] = None"},{name:"rescale_factor",val:": typing.Optional[float] = None"},{name:"do_normalize",val:": typing.Optional[bool] = None"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"do_pad",val:": typing.Optional[bool] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"data_format",val:": ChannelDimension = <ChannelDimension.FIRST: 'channels_first'>"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}],parametersDescription:[{anchor:"transformers.ViltImageProcessor.preprocess.images",description:`<strong>images</strong> (<code>ImageInput</code>) &#x2014;
Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
passing in images with pixel values between 0 and 1, set <code>do_rescale=False</code>.`,name:"images"},{anchor:"transformers.ViltImageProcessor.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_resize</code>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.ViltImageProcessor.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>, defaults to <code>self.size</code>) &#x2014;
Controls the size of the image after <code>resize</code>. The shortest edge of the image is resized to
<code>size[&quot;shortest_edge&quot;]</code> whilst preserving the aspect ratio. If the longest edge of this resized image
is &gt; <code>int(size[&quot;shortest_edge&quot;] * (1333 / 800))</code>, then the image is resized again to make the longest
edge equal to <code>int(size[&quot;shortest_edge&quot;] * (1333 / 800))</code>.`,name:"size"},{anchor:"transformers.ViltImageProcessor.preprocess.size_divisor",description:`<strong>size_divisor</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.size_divisor</code>) &#x2014;
The image is resized to a size that is a multiple of this value.`,name:"size_divisor"},{anchor:"transformers.ViltImageProcessor.preprocess.resample",description:`<strong>resample</strong> (<code>PILImageResampling</code>, <em>optional</em>, defaults to <code>self.resample</code>) &#x2014;
Resampling filter to use if resizing the image. Only has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.ViltImageProcessor.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_rescale</code>) &#x2014;
Whether to rescale the image values between [0 - 1].`,name:"do_rescale"},{anchor:"transformers.ViltImageProcessor.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>float</code>, <em>optional</em>, defaults to <code>self.rescale_factor</code>) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.ViltImageProcessor.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_normalize</code>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.ViltImageProcessor.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>self.image_mean</code>) &#x2014;
Image mean to normalize the image by if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_mean"},{anchor:"transformers.ViltImageProcessor.preprocess.image_std",description:`<strong>image_std</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>self.image_std</code>) &#x2014;
Image standard deviation to normalize the image by if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_std"},{anchor:"transformers.ViltImageProcessor.preprocess.do_pad",description:`<strong>do_pad</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_pad</code>) &#x2014;
Whether to pad the image to the (max_height, max_width) in the batch. If <code>True</code>, a pixel mask is also
created and returned.`,name:"do_pad"},{anchor:"transformers.ViltImageProcessor.preprocess.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <code>TensorType</code>, <em>optional</em>) &#x2014;
The type of tensors to return. Can be one of:<ul>
<li>Unset: Return a list of <code>np.ndarray</code>.</li>
<li><code>TensorType.TENSORFLOW</code> or <code>&apos;tf&apos;</code>: Return a batch of type <code>tf.Tensor</code>.</li>
<li><code>TensorType.PYTORCH</code> or <code>&apos;pt&apos;</code>: Return a batch of type <code>torch.Tensor</code>.</li>
<li><code>TensorType.NUMPY</code> or <code>&apos;np&apos;</code>: Return a batch of type <code>np.ndarray</code>.</li>
<li><code>TensorType.JAX</code> or <code>&apos;jax&apos;</code>: Return a batch of type <code>jax.numpy.ndarray</code>.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.ViltImageProcessor.preprocess.data_format",description:`<strong>data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>, defaults to <code>ChannelDimension.FIRST</code>) &#x2014;
The channel dimension format for the output image. Can be one of:<ul>
<li><code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
</ul>`,name:"data_format"},{anchor:"transformers.ViltImageProcessor.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/image_processing_vilt.py#L340"}}),He=new U({props:{title:"ViltImageProcessorFast",local:"transformers.ViltImageProcessorFast",headingTag:"h2"}}),qe=new V({props:{name:"class transformers.ViltImageProcessorFast",anchor:"transformers.ViltImageProcessorFast",parameters:[{name:"**kwargs",val:": typing_extensions.Unpack[transformers.image_processing_utils_fast.DefaultFastImageProcessorKwargs]"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/image_processing_vilt_fast.py#L69"}}),Se=new V({props:{name:"preprocess",anchor:"transformers.ViltImageProcessorFast.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"*args",val:""},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.image_processing_utils_fast.DefaultFastImageProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.ViltImageProcessorFast.preprocess.images",description:`<strong>images</strong> (<code>Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[&apos;PIL.Image.Image&apos;], list[numpy.ndarray], list[&apos;torch.Tensor&apos;]]</code>) &#x2014;
Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
passing in images with pixel values between 0 and 1, set <code>do_rescale=False</code>.`,name:"images"},{anchor:"transformers.ViltImageProcessorFast.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.ViltImageProcessorFast.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
Describes the maximum input dimensions to the model.`,name:"size"},{anchor:"transformers.ViltImageProcessorFast.preprocess.default_to_square",description:`<strong>default_to_square</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to default to a square image when resizing, if size is an int.`,name:"default_to_square"},{anchor:"transformers.ViltImageProcessorFast.preprocess.resample",description:`<strong>resample</strong> (<code>Union[PILImageResampling, F.InterpolationMode, NoneType]</code>) &#x2014;
Resampling filter to use if resizing the image. This can be one of the enum <code>PILImageResampling</code>. Only
has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.ViltImageProcessorFast.preprocess.do_center_crop",description:`<strong>do_center_crop</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to center crop the image.`,name:"do_center_crop"},{anchor:"transformers.ViltImageProcessorFast.preprocess.crop_size",description:`<strong>crop_size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
Size of the output image after applying <code>center_crop</code>.`,name:"crop_size"},{anchor:"transformers.ViltImageProcessorFast.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.ViltImageProcessorFast.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>Union[int, float, NoneType]</code>) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.ViltImageProcessorFast.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.ViltImageProcessorFast.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>Union[float, list[float], NoneType]</code>) &#x2014;
Image mean to use for normalization. Only has an effect if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_mean"},{anchor:"transformers.ViltImageProcessorFast.preprocess.image_std",description:`<strong>image_std</strong> (<code>Union[float, list[float], NoneType]</code>) &#x2014;
Image standard deviation to use for normalization. Only has an effect if <code>do_normalize</code> is set to
<code>True</code>.`,name:"image_std"},{anchor:"transformers.ViltImageProcessorFast.preprocess.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.ViltImageProcessorFast.preprocess.return_tensors",description:"<strong>return_tensors</strong> (<code>Union[str, ~utils.generic.TensorType, NoneType]</code>) &#x2014;\nReturns stacked tensors if set to `pt, otherwise returns a list of tensors.",name:"return_tensors"},{anchor:"transformers.ViltImageProcessorFast.preprocess.data_format",description:`<strong>data_format</strong> (<code>~image_utils.ChannelDimension</code>, <em>optional</em>) &#x2014;
Only <code>ChannelDimension.FIRST</code> is supported. Added for compatibility with slow processors.`,name:"data_format"},{anchor:"transformers.ViltImageProcessorFast.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>Union[str, ~image_utils.ChannelDimension, NoneType]</code>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"},{anchor:"transformers.ViltImageProcessorFast.preprocess.device",description:`<strong>device</strong> (<code>torch.device</code>, <em>optional</em>) &#x2014;
The device to process the images on. If unset, the device is inferred from the input images.`,name:"device"},{anchor:"transformers.ViltImageProcessorFast.preprocess.disable_grouping",description:`<strong>disable_grouping</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to disable grouping of images by size to process them individually and not in batches.
If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
empirical observations, as detailed here: <a href="https://github.com/huggingface/transformers/pull/38157" rel="nofollow">https://github.com/huggingface/transformers/pull/38157</a>`,name:"disable_grouping"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils_fast.py#L639",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<ul>
<li><strong>data</strong> (<code>dict</code>) — Dictionary of lists/arrays/tensors returned by the <strong>call</strong> method (‘pixel_values’, etc.).</li>
<li><strong>tensor_type</strong> (<code>Union[None, str, TensorType]</code>, <em>optional</em>) — You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
initialization.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>&lt;class 'transformers.image_processing_base.BatchFeature'&gt;</code></p>
`}}),Ee=new U({props:{title:"ViltProcessor",local:"transformers.ViltProcessor",headingTag:"h2"}}),Qe=new V({props:{name:"class transformers.ViltProcessor",anchor:"transformers.ViltProcessor",parameters:[{name:"image_processor",val:" = None"},{name:"tokenizer",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ViltProcessor.image_processor",description:`<strong>image_processor</strong> (<code>ViltImageProcessor</code>, <em>optional</em>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a>. The image processor is a required input.`,name:"image_processor"},{anchor:"transformers.ViltProcessor.tokenizer",description:"<strong>tokenizer</strong> (<code>BertTokenizerFast</code>, <em>optional</em>) &#x2014;\nAn instance of [&#x2018;BertTokenizerFast`]. The tokenizer is a required input.",name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/processing_vilt.py#L27"}}),Xe=new V({props:{name:"__call__",anchor:"transformers.ViltProcessor.__call__",parameters:[{name:"images",val:""},{name:"text",val:": typing.Union[str, list[str], list[list[str]]] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/processing_vilt.py#L64"}}),Oe=new U({props:{title:"ViltModel",local:"transformers.ViltModel",headingTag:"h2"}}),Ye=new V({props:{name:"class transformers.ViltModel",anchor:"transformers.ViltModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.ViltModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltModel">ViltModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.ViltModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L567"}}),De=new V({props:{name:"forward",anchor:"transformers.ViltModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_token_type_idx",val:": typing.Optional[int] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ViltModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ViltModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ViltImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ViltModel.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ViltModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltModel.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltModel.forward.image_token_type_idx",description:`<strong>image_token_type_idx</strong> (<code>int</code>, <em>optional</em>) &#x2014;</p>
<ul>
<li>The token type ids for images.</li>
</ul>`,name:"image_token_type_idx"},{anchor:"transformers.ViltModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L599",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
`}}),me=new to({props:{$$slots:{default:[Ns]},$$scope:{ctx:w}}}),pe=new bt({props:{anchor:"transformers.ViltModel.forward.example",$$slots:{default:[Ls]},$$scope:{ctx:w}}}),Ke=new U({props:{title:"ViltForMaskedLM",local:"transformers.ViltForMaskedLM",headingTag:"h2"}}),et=new V({props:{name:"class transformers.ViltForMaskedLM",anchor:"transformers.ViltForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForMaskedLM">ViltForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L739"}}),tt=new V({props:{name:"forward",anchor:"transformers.ViltForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ViltForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ViltForMaskedLM.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ViltImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ViltForMaskedLM.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ViltForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForMaskedLM.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>*torch.LongTensor*</code> of shape <em>(batch_size, sequence_length)</em>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <em>[-100, 0, &#x2026;,
config.vocab_size]</em> (see <em>input_ids</em> docstring) Tokens with indices set to <em>-100</em> are ignored (masked), the
loss is only computed for the tokens with labels in <em>[0, &#x2026;, config.vocab_size]</em>`,name:"labels"},{anchor:"transformers.ViltForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L758",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
`}}),he=new to({props:{$$slots:{default:[As]},$$scope:{ctx:w}}}),ge=new bt({props:{anchor:"transformers.ViltForMaskedLM.forward.example",$$slots:{default:[Gs]},$$scope:{ctx:w}}}),ot=new U({props:{title:"ViltForQuestionAnswering",local:"transformers.ViltForQuestionAnswering",headingTag:"h2"}}),nt=new V({props:{name:"class transformers.ViltForQuestionAnswering",anchor:"transformers.ViltForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForQuestionAnswering">ViltForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L918"}}),st=new V({props:{name:"forward",anchor:"transformers.ViltForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ViltForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ViltForQuestionAnswering.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ViltImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ViltForQuestionAnswering.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ViltForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForQuestionAnswering.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForQuestionAnswering.forward.labels",description:`<strong>labels</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_labels)</code>, <em>optional</em>) &#x2014;
Labels for computing the visual question answering loss. This tensor must be either a one-hot encoding of
all answers that are applicable for a given example in the batch, or a soft encoding indicating which
answers are applicable, where 1.0 is the highest score.`,name:"labels"},{anchor:"transformers.ViltForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L936",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
`}}),fe=new to({props:{$$slots:{default:[Hs]},$$scope:{ctx:w}}}),ue=new bt({props:{anchor:"transformers.ViltForQuestionAnswering.forward.example",$$slots:{default:[qs]},$$scope:{ctx:w}}}),at=new U({props:{title:"ViltForImagesAndTextClassification",local:"transformers.ViltForImagesAndTextClassification",headingTag:"h2"}}),rt=new V({props:{name:"class transformers.ViltForImagesAndTextClassification",anchor:"transformers.ViltForImagesAndTextClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForImagesAndTextClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForImagesAndTextClassification">ViltForImagesAndTextClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L1128"}}),it=new V({props:{name:"forward",anchor:"transformers.ViltForImagesAndTextClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForImagesAndTextClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ViltImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Binary classification labels.`,name:"labels"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForImagesAndTextClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L1147",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.vilt.modeling_vilt.ViltForImagesAndTextClassificationOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>list[tuple(torch.FloatTensor)]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — List of tuples of <code>torch.FloatTensor</code> (one for each image-text pair, each tuple containing the output of
the embeddings + one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.
Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>list[tuple[torch.FloatTensor]]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.vilt.modeling_vilt.ViltForImagesAndTextClassificationOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),_e=new to({props:{$$slots:{default:[Ss]},$$scope:{ctx:w}}}),be=new bt({props:{anchor:"transformers.ViltForImagesAndTextClassification.forward.example",$$slots:{default:[Es]},$$scope:{ctx:w}}}),lt=new U({props:{title:"ViltForImageAndTextRetrieval",local:"transformers.ViltForImageAndTextRetrieval",headingTag:"h2"}}),dt=new V({props:{name:"class transformers.ViltForImageAndTextRetrieval",anchor:"transformers.ViltForImageAndTextRetrieval",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForImageAndTextRetrieval.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForImageAndTextRetrieval">ViltForImageAndTextRetrieval</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L1030"}}),ct=new V({props:{name:"forward",anchor:"transformers.ViltForImageAndTextRetrieval.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForImageAndTextRetrieval.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ViltImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels are currently not supported.`,name:"labels"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForImageAndTextRetrieval.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L1042",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
`}}),ye=new to({props:{$$slots:{default:[Qs]},$$scope:{ctx:w}}}),Te=new bt({props:{anchor:"transformers.ViltForImageAndTextRetrieval.forward.example",$$slots:{default:[Xs]},$$scope:{ctx:w}}}),mt=new U({props:{title:"ViltForTokenClassification",local:"transformers.ViltForTokenClassification",headingTag:"h2"}}),pt=new V({props:{name:"class transformers.ViltForTokenClassification",anchor:"transformers.ViltForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ViltForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltForTokenClassification">ViltForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L1264"}}),ht=new V({props:{name:"forward",anchor:"transformers.ViltForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"image_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ViltForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ViltForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ViltForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ViltForTokenClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ViltImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltProcessor">ViltProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltImageProcessor">ViltImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ViltForTokenClassification.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ViltForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ViltForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ViltForTokenClassification.forward.image_embeds",description:`<strong>image_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_patches, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>pixel_values</code>, you can choose to directly pass an embedded representation.
This is useful if you want more control over how to convert <code>pixel_values</code> into patch embeddings.`,name:"image_embeds"},{anchor:"transformers.ViltForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, text_sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.ViltForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ViltForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ViltForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vilt/modeling_vilt.py#L1277",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/vilt#transformers.ViltConfig"
>ViltConfig</a>) and inputs.</p>
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
`}}),ve=new to({props:{$$slots:{default:[Os]},$$scope:{ctx:w}}}),Me=new bt({props:{anchor:"transformers.ViltForTokenClassification.forward.example",$$slots:{default:[Ys]},$$scope:{ctx:w}}}),gt=new Bs({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vilt.md"}}),{c(){t=l("meta"),T=r(),m=l("p"),p=r(),v=l("p"),v.innerHTML=o,M=r(),h(je.$$.fragment),so=r(),re=l("div"),re.innerHTML=Hn,ao=r(),h($e.$$.fragment),ro=r(),Ie=l("p"),Ie.innerHTML=qn,io=r(),Ce=l("p"),Ce.textContent=Sn,lo=r(),ze=l("p"),ze.innerHTML=En,co=r(),ie=l("img"),mo=r(),Fe=l("small"),Fe.innerHTML=Xn,po=r(),Ue=l("p"),Ue.innerHTML=On,ho=r(),h(Je.$$.fragment),go=r(),We=l("ul"),We.innerHTML=Yn,fo=r(),h(Ze.$$.fragment),uo=r(),J=l("div"),h(Pe.$$.fragment),No=r(),Tt=l("p"),Tt.innerHTML=Dn,Lo=r(),vt=l("p"),vt.innerHTML=Kn,Ao=r(),h(le.$$.fragment),_o=r(),h(Be.$$.fragment),bo=r(),oe=l("div"),h(Re.$$.fragment),Go=r(),de=l("div"),h(Ne.$$.fragment),Ho=r(),Mt=l("p"),Mt.textContent=es,yo=r(),h(Le.$$.fragment),To=r(),q=l("div"),h(Ae.$$.fragment),qo=r(),wt=l("p"),wt.textContent=ts,So=r(),ce=l("div"),h(Ge.$$.fragment),Eo=r(),kt=l("p"),kt.textContent=os,vo=r(),h(He.$$.fragment),Mo=r(),S=l("div"),h(qe.$$.fragment),Qo=r(),xt=l("p"),xt.textContent=ns,Xo=r(),Vt=l("div"),h(Se.$$.fragment),wo=r(),h(Ee.$$.fragment),ko=r(),W=l("div"),h(Qe.$$.fragment),Oo=r(),jt=l("p"),jt.textContent=ss,Yo=r(),$t=l("p"),$t.innerHTML=as,Do=r(),Y=l("div"),h(Xe.$$.fragment),Ko=r(),It=l("p"),It.innerHTML=rs,en=r(),Ct=l("p"),Ct.textContent=is,xo=r(),h(Oe.$$.fragment),Vo=r(),j=l("div"),h(Ye.$$.fragment),tn=r(),zt=l("p"),zt.textContent=ls,on=r(),Ft=l("p"),Ft.innerHTML=ds,nn=r(),Ut=l("p"),Ut.innerHTML=cs,sn=r(),R=l("div"),h(De.$$.fragment),an=r(),Jt=l("p"),Jt.innerHTML=ms,rn=r(),h(me.$$.fragment),ln=r(),h(pe.$$.fragment),jo=r(),h(Ke.$$.fragment),$o=r(),$=l("div"),h(et.$$.fragment),dn=r(),Wt=l("p"),Wt.textContent=ps,cn=r(),Zt=l("p"),Zt.innerHTML=hs,mn=r(),Pt=l("p"),Pt.innerHTML=gs,pn=r(),N=l("div"),h(tt.$$.fragment),hn=r(),Bt=l("p"),Bt.innerHTML=fs,gn=r(),h(he.$$.fragment),fn=r(),h(ge.$$.fragment),Io=r(),h(ot.$$.fragment),Co=r(),I=l("div"),h(nt.$$.fragment),un=r(),Rt=l("p"),Rt.textContent=us,_n=r(),Nt=l("p"),Nt.innerHTML=_s,bn=r(),Lt=l("p"),Lt.innerHTML=bs,yn=r(),L=l("div"),h(st.$$.fragment),Tn=r(),At=l("p"),At.innerHTML=ys,vn=r(),h(fe.$$.fragment),Mn=r(),h(ue.$$.fragment),zo=r(),h(at.$$.fragment),Fo=r(),C=l("div"),h(rt.$$.fragment),wn=r(),Gt=l("p"),Gt.textContent=Ts,kn=r(),Ht=l("p"),Ht.innerHTML=vs,xn=r(),qt=l("p"),qt.innerHTML=Ms,Vn=r(),A=l("div"),h(it.$$.fragment),jn=r(),St=l("p"),St.innerHTML=ws,$n=r(),h(_e.$$.fragment),In=r(),h(be.$$.fragment),Uo=r(),h(lt.$$.fragment),Jo=r(),z=l("div"),h(dt.$$.fragment),Cn=r(),Et=l("p"),Et.textContent=ks,zn=r(),Qt=l("p"),Qt.innerHTML=xs,Fn=r(),Xt=l("p"),Xt.innerHTML=Vs,Un=r(),G=l("div"),h(ct.$$.fragment),Jn=r(),Ot=l("p"),Ot.innerHTML=js,Wn=r(),h(ye.$$.fragment),Zn=r(),h(Te.$$.fragment),Wo=r(),h(mt.$$.fragment),Zo=r(),F=l("div"),h(pt.$$.fragment),Pn=r(),Yt=l("p"),Yt.textContent=$s,Bn=r(),Dt=l("p"),Dt.innerHTML=Is,Rn=r(),Kt=l("p"),Kt.innerHTML=Cs,Nn=r(),H=l("div"),h(ht.$$.fragment),Ln=r(),eo=l("p"),eo.innerHTML=zs,An=r(),h(ve.$$.fragment),Gn=r(),h(Me.$$.fragment),Po=r(),h(gt.$$.fragment),Bo=r(),oo=l("p"),this.h()},l(e){const n=Ps("svelte-u9bgzb",document.head);t=d(n,"META",{name:!0,content:!0}),n.forEach(s),T=i(e),m=d(e,"P",{}),x(m).forEach(s),p=i(e),v=d(e,"P",{"data-svelte-h":!0}),y(v)!=="svelte-11jsv1b"&&(v.innerHTML=o),M=i(e),g(je.$$.fragment,e),so=i(e),re=d(e,"DIV",{class:!0,"data-svelte-h":!0}),y(re)!=="svelte-13t8s2t"&&(re.innerHTML=Hn),ao=i(e),g($e.$$.fragment,e),ro=i(e),Ie=d(e,"P",{"data-svelte-h":!0}),y(Ie)!=="svelte-glols5"&&(Ie.innerHTML=qn),io=i(e),Ce=d(e,"P",{"data-svelte-h":!0}),y(Ce)!=="svelte-vfdo9a"&&(Ce.textContent=Sn),lo=i(e),ze=d(e,"P",{"data-svelte-h":!0}),y(ze)!=="svelte-44uf49"&&(ze.innerHTML=En),co=i(e),ie=d(e,"IMG",{src:!0,alt:!0,width:!0}),mo=i(e),Fe=d(e,"SMALL",{"data-svelte-h":!0}),y(Fe)!=="svelte-182lijk"&&(Fe.innerHTML=Xn),po=i(e),Ue=d(e,"P",{"data-svelte-h":!0}),y(Ue)!=="svelte-1u1ca4l"&&(Ue.innerHTML=On),ho=i(e),g(Je.$$.fragment,e),go=i(e),We=d(e,"UL",{"data-svelte-h":!0}),y(We)!=="svelte-1k6vqpo"&&(We.innerHTML=Yn),fo=i(e),g(Ze.$$.fragment,e),uo=i(e),J=d(e,"DIV",{class:!0});var E=x(J);g(Pe.$$.fragment,E),No=i(E),Tt=d(E,"P",{"data-svelte-h":!0}),y(Tt)!=="svelte-u7ypkj"&&(Tt.innerHTML=Dn),Lo=i(E),vt=d(E,"P",{"data-svelte-h":!0}),y(vt)!=="svelte-1ek1ss9"&&(vt.innerHTML=Kn),Ao=i(E),g(le.$$.fragment,E),E.forEach(s),_o=i(e),g(Be.$$.fragment,e),bo=i(e),oe=d(e,"DIV",{class:!0});var ft=x(oe);g(Re.$$.fragment,ft),Go=i(ft),de=d(ft,"DIV",{class:!0});var ut=x(de);g(Ne.$$.fragment,ut),Ho=i(ut),Mt=d(ut,"P",{"data-svelte-h":!0}),y(Mt)!=="svelte-khengj"&&(Mt.textContent=es),ut.forEach(s),ft.forEach(s),yo=i(e),g(Le.$$.fragment,e),To=i(e),q=d(e,"DIV",{class:!0});var ne=x(q);g(Ae.$$.fragment,ne),qo=i(ne),wt=d(ne,"P",{"data-svelte-h":!0}),y(wt)!=="svelte-1mh06yd"&&(wt.textContent=ts),So=i(ne),ce=d(ne,"DIV",{class:!0});var _t=x(ce);g(Ge.$$.fragment,_t),Eo=i(_t),kt=d(_t,"P",{"data-svelte-h":!0}),y(kt)!=="svelte-1x3yxsa"&&(kt.textContent=os),_t.forEach(s),ne.forEach(s),vo=i(e),g(He.$$.fragment,e),Mo=i(e),S=d(e,"DIV",{class:!0});var se=x(S);g(qe.$$.fragment,se),Qo=i(se),xt=d(se,"P",{"data-svelte-h":!0}),y(xt)!=="svelte-jia723"&&(xt.textContent=ns),Xo=i(se),Vt=d(se,"DIV",{class:!0});var no=x(Vt);g(Se.$$.fragment,no),no.forEach(s),se.forEach(s),wo=i(e),g(Ee.$$.fragment,e),ko=i(e),W=d(e,"DIV",{class:!0});var Q=x(W);g(Qe.$$.fragment,Q),Oo=i(Q),jt=d(Q,"P",{"data-svelte-h":!0}),y(jt)!=="svelte-1w20qql"&&(jt.textContent=ss),Yo=i(Q),$t=d(Q,"P",{"data-svelte-h":!0}),y($t)!=="svelte-lk3xno"&&($t.innerHTML=as),Do=i(Q),Y=d(Q,"DIV",{class:!0});var ae=x(Y);g(Xe.$$.fragment,ae),Ko=i(ae),It=d(ae,"P",{"data-svelte-h":!0}),y(It)!=="svelte-1ihzwx0"&&(It.innerHTML=rs),en=i(ae),Ct=d(ae,"P",{"data-svelte-h":!0}),y(Ct)!=="svelte-ws0hzs"&&(Ct.textContent=is),ae.forEach(s),Q.forEach(s),xo=i(e),g(Oe.$$.fragment,e),Vo=i(e),j=d(e,"DIV",{class:!0});var Z=x(j);g(Ye.$$.fragment,Z),tn=i(Z),zt=d(Z,"P",{"data-svelte-h":!0}),y(zt)!=="svelte-mmtjy1"&&(zt.textContent=ls),on=i(Z),Ft=d(Z,"P",{"data-svelte-h":!0}),y(Ft)!=="svelte-q52n56"&&(Ft.innerHTML=ds),nn=i(Z),Ut=d(Z,"P",{"data-svelte-h":!0}),y(Ut)!=="svelte-hswkmf"&&(Ut.innerHTML=cs),sn=i(Z),R=d(Z,"DIV",{class:!0});var X=x(R);g(De.$$.fragment,X),an=i(X),Jt=d(X,"P",{"data-svelte-h":!0}),y(Jt)!=="svelte-18zxa2y"&&(Jt.innerHTML=ms),rn=i(X),g(me.$$.fragment,X),ln=i(X),g(pe.$$.fragment,X),X.forEach(s),Z.forEach(s),jo=i(e),g(Ke.$$.fragment,e),$o=i(e),$=d(e,"DIV",{class:!0});var P=x($);g(et.$$.fragment,P),dn=i(P),Wt=d(P,"P",{"data-svelte-h":!0}),y(Wt)!=="svelte-fi8ziq"&&(Wt.textContent=ps),cn=i(P),Zt=d(P,"P",{"data-svelte-h":!0}),y(Zt)!=="svelte-q52n56"&&(Zt.innerHTML=hs),mn=i(P),Pt=d(P,"P",{"data-svelte-h":!0}),y(Pt)!=="svelte-hswkmf"&&(Pt.innerHTML=gs),pn=i(P),N=d(P,"DIV",{class:!0});var O=x(N);g(tt.$$.fragment,O),hn=i(O),Bt=d(O,"P",{"data-svelte-h":!0}),y(Bt)!=="svelte-i56dm2"&&(Bt.innerHTML=fs),gn=i(O),g(he.$$.fragment,O),fn=i(O),g(ge.$$.fragment,O),O.forEach(s),P.forEach(s),Io=i(e),g(ot.$$.fragment,e),Co=i(e),I=d(e,"DIV",{class:!0});var D=x(I);g(nt.$$.fragment,D),un=i(D),Rt=d(D,"P",{"data-svelte-h":!0}),y(Rt)!=="svelte-14c41nz"&&(Rt.textContent=us),_n=i(D),Nt=d(D,"P",{"data-svelte-h":!0}),y(Nt)!=="svelte-q52n56"&&(Nt.innerHTML=_s),bn=i(D),Lt=d(D,"P",{"data-svelte-h":!0}),y(Lt)!=="svelte-hswkmf"&&(Lt.innerHTML=bs),yn=i(D),L=d(D,"DIV",{class:!0});var we=x(L);g(st.$$.fragment,we),Tn=i(we),At=d(we,"P",{"data-svelte-h":!0}),y(At)!=="svelte-1m1anyi"&&(At.innerHTML=ys),vn=i(we),g(fe.$$.fragment,we),Mn=i(we),g(ue.$$.fragment,we),we.forEach(s),D.forEach(s),zo=i(e),g(at.$$.fragment,e),Fo=i(e),C=d(e,"DIV",{class:!0});var K=x(C);g(rt.$$.fragment,K),wn=i(K),Gt=d(K,"P",{"data-svelte-h":!0}),y(Gt)!=="svelte-1uhjqed"&&(Gt.textContent=Ts),kn=i(K),Ht=d(K,"P",{"data-svelte-h":!0}),y(Ht)!=="svelte-q52n56"&&(Ht.innerHTML=vs),xn=i(K),qt=d(K,"P",{"data-svelte-h":!0}),y(qt)!=="svelte-hswkmf"&&(qt.innerHTML=Ms),Vn=i(K),A=d(K,"DIV",{class:!0});var ke=x(A);g(it.$$.fragment,ke),jn=i(ke),St=d(ke,"P",{"data-svelte-h":!0}),y(St)!=="svelte-5cver6"&&(St.innerHTML=ws),$n=i(ke),g(_e.$$.fragment,ke),In=i(ke),g(be.$$.fragment,ke),ke.forEach(s),K.forEach(s),Uo=i(e),g(lt.$$.fragment,e),Jo=i(e),z=d(e,"DIV",{class:!0});var ee=x(z);g(dt.$$.fragment,ee),Cn=i(ee),Et=d(ee,"P",{"data-svelte-h":!0}),y(Et)!=="svelte-1mcrono"&&(Et.textContent=ks),zn=i(ee),Qt=d(ee,"P",{"data-svelte-h":!0}),y(Qt)!=="svelte-q52n56"&&(Qt.innerHTML=xs),Fn=i(ee),Xt=d(ee,"P",{"data-svelte-h":!0}),y(Xt)!=="svelte-hswkmf"&&(Xt.innerHTML=Vs),Un=i(ee),G=d(ee,"DIV",{class:!0});var xe=x(G);g(ct.$$.fragment,xe),Jn=i(xe),Ot=d(xe,"P",{"data-svelte-h":!0}),y(Ot)!=="svelte-jd66z8"&&(Ot.innerHTML=js),Wn=i(xe),g(ye.$$.fragment,xe),Zn=i(xe),g(Te.$$.fragment,xe),xe.forEach(s),ee.forEach(s),Wo=i(e),g(mt.$$.fragment,e),Zo=i(e),F=d(e,"DIV",{class:!0});var te=x(F);g(pt.$$.fragment,te),Pn=i(te),Yt=d(te,"P",{"data-svelte-h":!0}),y(Yt)!=="svelte-w73uj"&&(Yt.textContent=$s),Bn=i(te),Dt=d(te,"P",{"data-svelte-h":!0}),y(Dt)!=="svelte-q52n56"&&(Dt.innerHTML=Is),Rn=i(te),Kt=d(te,"P",{"data-svelte-h":!0}),y(Kt)!=="svelte-hswkmf"&&(Kt.innerHTML=Cs),Nn=i(te),H=d(te,"DIV",{class:!0});var Ve=x(H);g(ht.$$.fragment,Ve),Ln=i(Ve),eo=d(Ve,"P",{"data-svelte-h":!0}),y(eo)!=="svelte-1wb1dkc"&&(eo.innerHTML=zs),An=i(Ve),g(ve.$$.fragment,Ve),Gn=i(Ve),g(Me.$$.fragment,Ve),Ve.forEach(s),te.forEach(s),Po=i(e),g(gt.$$.fragment,e),Bo=i(e),oo=d(e,"P",{}),x(oo).forEach(s),this.h()},h(){k(t,"name","hf:doc:metadata"),k(t,"content",Ks),k(re,"class","flex flex-wrap space-x-1"),Us(ie.src,Qn="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vilt_architecture.jpg")||k(ie,"src",Qn),k(ie,"alt","drawing"),k(ie,"width","600"),k(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){a(document.head,t),c(e,T,n),c(e,m,n),c(e,p,n),c(e,v,n),c(e,M,n),f(je,e,n),c(e,so,n),c(e,re,n),c(e,ao,n),f($e,e,n),c(e,ro,n),c(e,Ie,n),c(e,io,n),c(e,Ce,n),c(e,lo,n),c(e,ze,n),c(e,co,n),c(e,ie,n),c(e,mo,n),c(e,Fe,n),c(e,po,n),c(e,Ue,n),c(e,ho,n),f(Je,e,n),c(e,go,n),c(e,We,n),c(e,fo,n),f(Ze,e,n),c(e,uo,n),c(e,J,n),f(Pe,J,null),a(J,No),a(J,Tt),a(J,Lo),a(J,vt),a(J,Ao),f(le,J,null),c(e,_o,n),f(Be,e,n),c(e,bo,n),c(e,oe,n),f(Re,oe,null),a(oe,Go),a(oe,de),f(Ne,de,null),a(de,Ho),a(de,Mt),c(e,yo,n),f(Le,e,n),c(e,To,n),c(e,q,n),f(Ae,q,null),a(q,qo),a(q,wt),a(q,So),a(q,ce),f(Ge,ce,null),a(ce,Eo),a(ce,kt),c(e,vo,n),f(He,e,n),c(e,Mo,n),c(e,S,n),f(qe,S,null),a(S,Qo),a(S,xt),a(S,Xo),a(S,Vt),f(Se,Vt,null),c(e,wo,n),f(Ee,e,n),c(e,ko,n),c(e,W,n),f(Qe,W,null),a(W,Oo),a(W,jt),a(W,Yo),a(W,$t),a(W,Do),a(W,Y),f(Xe,Y,null),a(Y,Ko),a(Y,It),a(Y,en),a(Y,Ct),c(e,xo,n),f(Oe,e,n),c(e,Vo,n),c(e,j,n),f(Ye,j,null),a(j,tn),a(j,zt),a(j,on),a(j,Ft),a(j,nn),a(j,Ut),a(j,sn),a(j,R),f(De,R,null),a(R,an),a(R,Jt),a(R,rn),f(me,R,null),a(R,ln),f(pe,R,null),c(e,jo,n),f(Ke,e,n),c(e,$o,n),c(e,$,n),f(et,$,null),a($,dn),a($,Wt),a($,cn),a($,Zt),a($,mn),a($,Pt),a($,pn),a($,N),f(tt,N,null),a(N,hn),a(N,Bt),a(N,gn),f(he,N,null),a(N,fn),f(ge,N,null),c(e,Io,n),f(ot,e,n),c(e,Co,n),c(e,I,n),f(nt,I,null),a(I,un),a(I,Rt),a(I,_n),a(I,Nt),a(I,bn),a(I,Lt),a(I,yn),a(I,L),f(st,L,null),a(L,Tn),a(L,At),a(L,vn),f(fe,L,null),a(L,Mn),f(ue,L,null),c(e,zo,n),f(at,e,n),c(e,Fo,n),c(e,C,n),f(rt,C,null),a(C,wn),a(C,Gt),a(C,kn),a(C,Ht),a(C,xn),a(C,qt),a(C,Vn),a(C,A),f(it,A,null),a(A,jn),a(A,St),a(A,$n),f(_e,A,null),a(A,In),f(be,A,null),c(e,Uo,n),f(lt,e,n),c(e,Jo,n),c(e,z,n),f(dt,z,null),a(z,Cn),a(z,Et),a(z,zn),a(z,Qt),a(z,Fn),a(z,Xt),a(z,Un),a(z,G),f(ct,G,null),a(G,Jn),a(G,Ot),a(G,Wn),f(ye,G,null),a(G,Zn),f(Te,G,null),c(e,Wo,n),f(mt,e,n),c(e,Zo,n),c(e,F,n),f(pt,F,null),a(F,Pn),a(F,Yt),a(F,Bn),a(F,Dt),a(F,Rn),a(F,Kt),a(F,Nn),a(F,H),f(ht,H,null),a(H,Ln),a(H,eo),a(H,An),f(ve,H,null),a(H,Gn),f(Me,H,null),c(e,Po,n),f(gt,e,n),c(e,Bo,n),c(e,oo,n),Ro=!0},p(e,[n]){const E={};n&2&&(E.$$scope={dirty:n,ctx:e}),le.$set(E);const ft={};n&2&&(ft.$$scope={dirty:n,ctx:e}),me.$set(ft);const ut={};n&2&&(ut.$$scope={dirty:n,ctx:e}),pe.$set(ut);const ne={};n&2&&(ne.$$scope={dirty:n,ctx:e}),he.$set(ne);const _t={};n&2&&(_t.$$scope={dirty:n,ctx:e}),ge.$set(_t);const se={};n&2&&(se.$$scope={dirty:n,ctx:e}),fe.$set(se);const no={};n&2&&(no.$$scope={dirty:n,ctx:e}),ue.$set(no);const Q={};n&2&&(Q.$$scope={dirty:n,ctx:e}),_e.$set(Q);const ae={};n&2&&(ae.$$scope={dirty:n,ctx:e}),be.$set(ae);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),ye.$set(Z);const X={};n&2&&(X.$$scope={dirty:n,ctx:e}),Te.$set(X);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),ve.$set(P);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),Me.$set(O)},i(e){Ro||(u(je.$$.fragment,e),u($e.$$.fragment,e),u(Je.$$.fragment,e),u(Ze.$$.fragment,e),u(Pe.$$.fragment,e),u(le.$$.fragment,e),u(Be.$$.fragment,e),u(Re.$$.fragment,e),u(Ne.$$.fragment,e),u(Le.$$.fragment,e),u(Ae.$$.fragment,e),u(Ge.$$.fragment,e),u(He.$$.fragment,e),u(qe.$$.fragment,e),u(Se.$$.fragment,e),u(Ee.$$.fragment,e),u(Qe.$$.fragment,e),u(Xe.$$.fragment,e),u(Oe.$$.fragment,e),u(Ye.$$.fragment,e),u(De.$$.fragment,e),u(me.$$.fragment,e),u(pe.$$.fragment,e),u(Ke.$$.fragment,e),u(et.$$.fragment,e),u(tt.$$.fragment,e),u(he.$$.fragment,e),u(ge.$$.fragment,e),u(ot.$$.fragment,e),u(nt.$$.fragment,e),u(st.$$.fragment,e),u(fe.$$.fragment,e),u(ue.$$.fragment,e),u(at.$$.fragment,e),u(rt.$$.fragment,e),u(it.$$.fragment,e),u(_e.$$.fragment,e),u(be.$$.fragment,e),u(lt.$$.fragment,e),u(dt.$$.fragment,e),u(ct.$$.fragment,e),u(ye.$$.fragment,e),u(Te.$$.fragment,e),u(mt.$$.fragment,e),u(pt.$$.fragment,e),u(ht.$$.fragment,e),u(ve.$$.fragment,e),u(Me.$$.fragment,e),u(gt.$$.fragment,e),Ro=!0)},o(e){_(je.$$.fragment,e),_($e.$$.fragment,e),_(Je.$$.fragment,e),_(Ze.$$.fragment,e),_(Pe.$$.fragment,e),_(le.$$.fragment,e),_(Be.$$.fragment,e),_(Re.$$.fragment,e),_(Ne.$$.fragment,e),_(Le.$$.fragment,e),_(Ae.$$.fragment,e),_(Ge.$$.fragment,e),_(He.$$.fragment,e),_(qe.$$.fragment,e),_(Se.$$.fragment,e),_(Ee.$$.fragment,e),_(Qe.$$.fragment,e),_(Xe.$$.fragment,e),_(Oe.$$.fragment,e),_(Ye.$$.fragment,e),_(De.$$.fragment,e),_(me.$$.fragment,e),_(pe.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(he.$$.fragment,e),_(ge.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(st.$$.fragment,e),_(fe.$$.fragment,e),_(ue.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(it.$$.fragment,e),_(_e.$$.fragment,e),_(be.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(ye.$$.fragment,e),_(Te.$$.fragment,e),_(mt.$$.fragment,e),_(pt.$$.fragment,e),_(ht.$$.fragment,e),_(ve.$$.fragment,e),_(Me.$$.fragment,e),_(gt.$$.fragment,e),Ro=!1},d(e){e&&(s(T),s(m),s(p),s(v),s(M),s(so),s(re),s(ao),s(ro),s(Ie),s(io),s(Ce),s(lo),s(ze),s(co),s(ie),s(mo),s(Fe),s(po),s(Ue),s(ho),s(go),s(We),s(fo),s(uo),s(J),s(_o),s(bo),s(oe),s(yo),s(To),s(q),s(vo),s(Mo),s(S),s(wo),s(ko),s(W),s(xo),s(Vo),s(j),s(jo),s($o),s($),s(Io),s(Co),s(I),s(zo),s(Fo),s(C),s(Uo),s(Jo),s(z),s(Wo),s(Zo),s(F),s(Po),s(Bo),s(oo)),s(t),b(je,e),b($e,e),b(Je,e),b(Ze,e),b(Pe),b(le),b(Be,e),b(Re),b(Ne),b(Le,e),b(Ae),b(Ge),b(He,e),b(qe),b(Se),b(Ee,e),b(Qe),b(Xe),b(Oe,e),b(Ye),b(De),b(me),b(pe),b(Ke,e),b(et),b(tt),b(he),b(ge),b(ot,e),b(nt),b(st),b(fe),b(ue),b(at,e),b(rt),b(it),b(_e),b(be),b(lt,e),b(dt),b(ct),b(ye),b(Te),b(mt,e),b(pt),b(ht),b(ve),b(Me),b(gt,e)}}}const Ks='{"title":"ViLT","local":"vilt","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"ViltConfig","local":"transformers.ViltConfig","sections":[],"depth":2},{"title":"ViltFeatureExtractor","local":"transformers.ViltFeatureExtractor","sections":[],"depth":2},{"title":"ViltImageProcessor","local":"transformers.ViltImageProcessor","sections":[],"depth":2},{"title":"ViltImageProcessorFast","local":"transformers.ViltImageProcessorFast","sections":[],"depth":2},{"title":"ViltProcessor","local":"transformers.ViltProcessor","sections":[],"depth":2},{"title":"ViltModel","local":"transformers.ViltModel","sections":[],"depth":2},{"title":"ViltForMaskedLM","local":"transformers.ViltForMaskedLM","sections":[],"depth":2},{"title":"ViltForQuestionAnswering","local":"transformers.ViltForQuestionAnswering","sections":[],"depth":2},{"title":"ViltForImagesAndTextClassification","local":"transformers.ViltForImagesAndTextClassification","sections":[],"depth":2},{"title":"ViltForImageAndTextRetrieval","local":"transformers.ViltForImageAndTextRetrieval","sections":[],"depth":2},{"title":"ViltForTokenClassification","local":"transformers.ViltForTokenClassification","sections":[],"depth":2}],"depth":1}';function ea(w){return Js(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class la extends Ws{constructor(t){super(),Zs(this,t,ea,Ds,Fs,{})}}export{la as component};
