import{s as gr,o as fr,n as j}from"../chunks/scheduler.18a86fab.js";import{S as _r,i as br,g as l,s,r as u,A as kr,h as d,f as r,c as a,j as x,x as h,u as g,k as v,y as o,a as m,v as f,d as _,t as b,w as k}from"../chunks/index.98837b22.js";import{T as Yt}from"../chunks/Tip.77304350.js";import{D as N}from"../chunks/Docstring.a1ef7999.js";import{C as ce}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as de}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as W,E as Tr}from"../chunks/getInferenceSnippets.06c2775f.js";function yr(w){let n,y="Examples:",p,c,T;return c=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFhMTmV0Q29uZmlnJTJDJTIwWExOZXRNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBYTE5ldCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwWExOZXRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwWExOZXRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> XLNetConfig, XLNetModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a XLNet configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = XLNetConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-kvfsh7"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function Mr(w){let n,y="sequence pair mask has the following format:",p,c,T;return c=new ce({props:{code:"MCUyMDAlMjAwJTIwMCUyMDAlMjAwJTIwMCUyMDAlMjAwJTIwMCUyMDAlMjAxJTIwMSUyMDElMjAxJTIwMSUyMDElMjAxJTIwMSUyMDElMEElN0MlMjBmaXJzdCUyMHNlcXVlbmNlJTIwJTIwJTIwJTIwJTdDJTIwc2Vjb25kJTIwc2VxdWVuY2UlMjAlN0M=",highlighted:`0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1 1
| first sequence    | second sequence |`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-16klr56"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function wr(w){let n,y="sequence pair mask has the following format:",p,c,T;return c=new ce({props:{code:"MCUyMDAlMjAwJTIwMCUyMDAlMjAwJTIwMCUyMDAlMjAwJTIwMCUyMDAlMjAxJTIwMSUyMDElMjAxJTIwMSUyMDElMjAxJTIwMSUyMDElMEElN0MlMjBmaXJzdCUyMHNlcXVlbmNlJTIwJTIwJTIwJTIwJTdDJTIwc2Vjb25kJTIwc2VxdWVuY2UlMjAlN0M=",highlighted:`0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1 1
| first sequence    | second sequence |`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-16klr56"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function vr(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=y},l(p){n=d(p,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(p,c){m(p,n,c)},p:j,d(p){p&&r(n)}}}function xr(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=y},l(p){n=d(p,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(p,c){m(p,n,c)},p:j,d(p){p&&r(n)}}}function Nr(w){let n,y="Examples:",p,c,T;return c=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE5ldExNSGVhZE1vZGVsJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ4bG5ldCUyRnhsbmV0LWxhcmdlLWNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExOZXRMTUhlYWRNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyeGxuZXQlMkZ4bG5ldC1sYXJnZS1jYXNlZCUyMiklMEElMEElMjMlMjBXZSUyMHNob3clMjBob3clMjB0byUyMHNldHVwJTIwaW5wdXRzJTIwdG8lMjBwcmVkaWN0JTIwYSUyMG5leHQlMjB0b2tlbiUyMHVzaW5nJTIwYSUyMGJpLWRpcmVjdGlvbmFsJTIwY29udGV4dC4lMEFpbnB1dF9pZHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTBBJTIwJTIwJTIwJTIwdG9rZW5pemVyLmVuY29kZSglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjB2ZXJ5JTIwJTNDbWFzayUzRSUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlKSUwQSkudW5zcXVlZXplKCUwQSUyMCUyMCUyMCUyMDAlMEEpJTIwJTIwJTIzJTIwV2UlMjB3aWxsJTIwcHJlZGljdCUyMHRoZSUyMG1hc2tlZCUyMHRva2VuJTBBcGVybV9tYXNrJTIwJTNEJTIwdG9yY2guemVyb3MoKDElMkMlMjBpbnB1dF9pZHMuc2hhcGUlNUIxJTVEJTJDJTIwaW5wdXRfaWRzLnNoYXBlJTVCMSU1RCklMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0KSUwQXBlcm1fbWFzayU1QiUzQSUyQyUyMCUzQSUyQyUyMC0xJTVEJTIwJTNEJTIwMS4wJTIwJTIwJTIzJTIwUHJldmlvdXMlMjB0b2tlbnMlMjBkb24ndCUyMHNlZSUyMGxhc3QlMjB0b2tlbiUwQXRhcmdldF9tYXBwaW5nJTIwJTNEJTIwdG9yY2guemVyb3MoJTBBJTIwJTIwJTIwJTIwKDElMkMlMjAxJTJDJTIwaW5wdXRfaWRzLnNoYXBlJTVCMSU1RCklMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0JTBBKSUyMCUyMCUyMyUyMFNoYXBlJTIwJTVCMSUyQyUyMDElMkMlMjBzZXFfbGVuZ3RoJTVEJTIwJTNEJTNFJTIwbGV0J3MlMjBwcmVkaWN0JTIwb25lJTIwdG9rZW4lMEF0YXJnZXRfbWFwcGluZyU1QiUwQSUyMCUyMCUyMCUyMDAlMkMlMjAwJTJDJTIwLTElMEElNUQlMjAlM0QlMjAxLjAlMjAlMjAlMjMlMjBPdXIlMjBmaXJzdCUyMChhbmQlMjBvbmx5KSUyMHByZWRpY3Rpb24lMjB3aWxsJTIwYmUlMjB0aGUlMjBsYXN0JTIwdG9rZW4lMjBvZiUyMHRoZSUyMHNlcXVlbmNlJTIwKHRoZSUyMG1hc2tlZCUyMHRva2VuKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlMkMlMjBwZXJtX21hc2slM0RwZXJtX21hc2slMkMlMjB0YXJnZXRfbWFwcGluZyUzRHRhcmdldF9tYXBwaW5nKSUwQW5leHRfdG9rZW5fbG9naXRzJTIwJTNEJTIwb3V0cHV0cyU1QiUwQSUyMCUyMCUyMCUyMDAlMEElNUQlMjAlMjAlMjMlMjBPdXRwdXQlMjBoYXMlMjBzaGFwZSUyMCU1QnRhcmdldF9tYXBwaW5nLnNpemUoMCklMkMlMjB0YXJnZXRfbWFwcGluZy5zaXplKDEpJTJDJTIwY29uZmlnLnZvY2FiX3NpemUlNUQlMEElMEElMjMlMjBUaGUlMjBzYW1lJTIwd2F5JTIwY2FuJTIwdGhlJTIwWExOZXRMTUhlYWRNb2RlbCUyMGJlJTIwdXNlZCUyMHRvJTIwYmUlMjB0cmFpbmVkJTIwYnklMjBzdGFuZGFyZCUyMGF1dG8tcmVncmVzc2l2ZSUyMGxhbmd1YWdlJTIwbW9kZWxpbmcuJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9yY2gudGVuc29yKCUwQSUyMCUyMCUyMCUyMHRva2VuaXplci5lbmNvZGUoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwdmVyeSUyMCUzQ21hc2slM0UlMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSklMEEpLnVuc3F1ZWV6ZSglMEElMjAlMjAlMjAlMjAwJTBBKSUyMCUyMCUyMyUyMFdlJTIwd2lsbCUyMHByZWRpY3QlMjB0aGUlMjBtYXNrZWQlMjB0b2tlbiUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcih0b2tlbml6ZXIuZW5jb2RlKCUyMmN1dGUlMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSkpLnVuc3F1ZWV6ZSgwKSUwQWFzc2VydCUyMGxhYmVscy5zaGFwZSU1QjAlNUQlMjAlM0QlM0QlMjAxJTJDJTIwJTIyb25seSUyMG9uZSUyMHdvcmQlMjB3aWxsJTIwYmUlMjBwcmVkaWN0ZWQlMjIlMEFwZXJtX21hc2slMjAlM0QlMjB0b3JjaC56ZXJvcygoMSUyQyUyMGlucHV0X2lkcy5zaGFwZSU1QjElNUQlMkMlMjBpbnB1dF9pZHMuc2hhcGUlNUIxJTVEKSUyQyUyMGR0eXBlJTNEdG9yY2guZmxvYXQpJTBBcGVybV9tYXNrJTVCJTBBJTIwJTIwJTIwJTIwJTNBJTJDJTIwJTNBJTJDJTIwLTElMEElNUQlMjAlM0QlMjAxLjAlMjAlMjAlMjMlMjBQcmV2aW91cyUyMHRva2VucyUyMGRvbid0JTIwc2VlJTIwbGFzdCUyMHRva2VuJTIwYXMlMjBpcyUyMGRvbmUlMjBpbiUyMHN0YW5kYXJkJTIwYXV0by1yZWdyZXNzaXZlJTIwbG0lMjB0cmFpbmluZyUwQXRhcmdldF9tYXBwaW5nJTIwJTNEJTIwdG9yY2guemVyb3MoJTBBJTIwJTIwJTIwJTIwKDElMkMlMjAxJTJDJTIwaW5wdXRfaWRzLnNoYXBlJTVCMSU1RCklMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0JTBBKSUyMCUyMCUyMyUyMFNoYXBlJTIwJTVCMSUyQyUyMDElMkMlMjBzZXFfbGVuZ3RoJTVEJTIwJTNEJTNFJTIwbGV0J3MlMjBwcmVkaWN0JTIwb25lJTIwdG9rZW4lMEF0YXJnZXRfbWFwcGluZyU1QiUwQSUyMCUyMCUyMCUyMDAlMkMlMjAwJTJDJTIwLTElMEElNUQlMjAlM0QlMjAxLjAlMjAlMjAlMjMlMjBPdXIlMjBmaXJzdCUyMChhbmQlMjBvbmx5KSUyMHByZWRpY3Rpb24lMjB3aWxsJTIwYmUlMjB0aGUlMjBsYXN0JTIwdG9rZW4lMjBvZiUyMHRoZSUyMHNlcXVlbmNlJTIwKHRoZSUyMG1hc2tlZCUyMHRva2VuKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlMkMlMjBwZXJtX21hc2slM0RwZXJtX21hc2slMkMlMjB0YXJnZXRfbWFwcGluZyUzRHRhcmdldF9tYXBwaW5nJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFuZXh0X3Rva2VuX2xvZ2l0cyUyMCUzRCUyMCglMEElMjAlMjAlMjAlMjBvdXRwdXRzLmxvZ2l0cyUwQSklMjAlMjAlMjMlMjBMb2dpdHMlMjBoYXZlJTIwc2hhcGUlMjAlNUJ0YXJnZXRfbWFwcGluZy5zaXplKDApJTJDJTIwdGFyZ2V0X21hcHBpbmcuc2l6ZSgxKSUyQyUyMGNvbmZpZy52b2NhYl9zaXplJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLNetLMHeadModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetLMHeadModel.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># We show how to setup inputs to predict a next token using a bi-directional context.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = torch.tensor(
<span class="hljs-meta">... </span>    tokenizer.encode(<span class="hljs-string">&quot;Hello, my dog is very &lt;mask&gt;&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>)
<span class="hljs-meta">... </span>).unsqueeze(
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># We will predict the masked token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>perm_mask = torch.zeros((<span class="hljs-number">1</span>, input_ids.shape[<span class="hljs-number">1</span>], input_ids.shape[<span class="hljs-number">1</span>]), dtype=torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>perm_mask[:, :, -<span class="hljs-number">1</span>] = <span class="hljs-number">1.0</span>  <span class="hljs-comment"># Previous tokens don&#x27;t see last token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_mapping = torch.zeros(
<span class="hljs-meta">... </span>    (<span class="hljs-number">1</span>, <span class="hljs-number">1</span>, input_ids.shape[<span class="hljs-number">1</span>]), dtype=torch.<span class="hljs-built_in">float</span>
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># Shape [1, 1, seq_length] =&gt; let&#x27;s predict one token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_mapping[
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, -<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>] = <span class="hljs-number">1.0</span>  <span class="hljs-comment"># Our first (and only) prediction will be the last token of the sequence (the masked token)</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
<span class="hljs-meta">&gt;&gt;&gt; </span>next_token_logits = outputs[
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>
<span class="hljs-meta">... </span>]  <span class="hljs-comment"># Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = torch.tensor(
<span class="hljs-meta">... </span>    tokenizer.encode(<span class="hljs-string">&quot;Hello, my dog is very &lt;mask&gt;&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>)
<span class="hljs-meta">... </span>).unsqueeze(
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># We will predict the masked token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(tokenizer.encode(<span class="hljs-string">&quot;cute&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>)).unsqueeze(<span class="hljs-number">0</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> labels.shape[<span class="hljs-number">0</span>] == <span class="hljs-number">1</span>, <span class="hljs-string">&quot;only one word will be predicted&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>perm_mask = torch.zeros((<span class="hljs-number">1</span>, input_ids.shape[<span class="hljs-number">1</span>], input_ids.shape[<span class="hljs-number">1</span>]), dtype=torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>perm_mask[
<span class="hljs-meta">... </span>    :, :, -<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>] = <span class="hljs-number">1.0</span>  <span class="hljs-comment"># Previous tokens don&#x27;t see last token as is done in standard auto-regressive lm training</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_mapping = torch.zeros(
<span class="hljs-meta">... </span>    (<span class="hljs-number">1</span>, <span class="hljs-number">1</span>, input_ids.shape[<span class="hljs-number">1</span>]), dtype=torch.<span class="hljs-built_in">float</span>
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># Shape [1, 1, seq_length] =&gt; let&#x27;s predict one token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_mapping[
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>, <span class="hljs-number">0</span>, -<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>] = <span class="hljs-number">1.0</span>  <span class="hljs-comment"># Our first (and only) prediction will be the last token of the sequence (the masked token)</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>next_token_logits = (
<span class="hljs-meta">... </span>    outputs.logits
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]</span>`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-kvfsh7"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function Lr(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=y},l(p){n=d(p,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(p,c){m(p,n,c)},p:j,d(p){p&&r(n)}}}function Xr(w){let n,y="Example of single-label classification:",p,c,T;return c=new ce({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFhMTmV0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnhsbmV0JTJGeGxuZXQtbGFyZ2UtY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBYTE5ldEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnhsbmV0JTJGeGxuZXQtbGFyZ2UtY2FzZWQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFhMTmV0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyeGxuZXQlMkZ4bG5ldC1sYXJnZS1jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLNetForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-ykxpe4"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function jr(w){let n,y="Example of multi-label classification:",p,c,T;return c=new ce({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFhMTmV0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnhsbmV0JTJGeGxuZXQtbGFyZ2UtY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBYTE5ldEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnhsbmV0JTJGeGxuZXQtbGFyZ2UtY2FzZWQlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFhMTmV0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyeGxuZXQlMkZ4bG5ldC1sYXJnZS1jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLNetForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-1l8e32d"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function $r(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=y},l(p){n=d(p,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(p,c){m(p,n,c)},p:j,d(p){p&&r(n)}}}function Fr(w){let n,y="Example:",p,c,T;return c=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE5ldEZvck11bHRpcGxlQ2hvaWNlJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ4bG5ldCUyRnhsbmV0LWxhcmdlLWNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExOZXRGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIyeGxuZXQlMkZ4bG5ldC1sYXJnZS1jYXNlZCUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBY2hvaWNlMCUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdpdGglMjBhJTIwZm9yayUyMGFuZCUyMGElMjBrbmlmZS4lMjIlMEFjaG9pY2UxJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2hpbGUlMjBoZWxkJTIwaW4lMjB0aGUlMjBoYW5kLiUyMiUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigwKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBjaG9pY2UwJTIwaXMlMjBjb3JyZWN0JTIwKGFjY29yZGluZyUyMHRvJTIwV2lraXBlZGlhJTIwJTNCKSklMkMlMjBiYXRjaCUyMHNpemUlMjAxJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTVCcHJvbXB0JTJDJTIwcHJvbXB0JTVEJTJDJTIwJTVCY2hvaWNlMCUyQyUyMGNob2ljZTElNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqJTdCayUzQSUyMHYudW5zcXVlZXplKDApJTIwZm9yJTIwayUyQyUyMHYlMjBpbiUyMGVuY29kaW5nLml0ZW1zKCklN0QlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTIwJTIwJTIzJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBJTBBJTIzJTIwdGhlJTIwbGluZWFyJTIwY2xhc3NpZmllciUyMHN0aWxsJTIwbmVlZHMlMjB0byUyMGJlJTIwdHJhaW5lZCUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLNetForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-11lpom8"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function Cr(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=y},l(p){n=d(p,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(p,c){m(p,n,c)},p:j,d(p){p&&r(n)}}}function Jr(w){let n,y="Example:",p,c,T;return c=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE5ldEZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnhsbmV0JTJGeGxuZXQtbGFyZ2UtY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBYTE5ldEZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnhsbmV0JTJGeGxuZXQtbGFyZ2UtY2FzZWQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLNetForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetForTokenClassification.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)

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
...`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-11lpom8"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function zr(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=y},l(p){n=d(p,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(p,c){m(p,n,c)},p:j,d(p){p&&r(n)}}}function Ur(w){let n,y="Example:",p,c,T;return c=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE5ldEZvclF1ZXN0aW9uQW5zd2VyaW5nU2ltcGxlJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ4bG5ldCUyRnhsbmV0LWxhcmdlLWNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExOZXRGb3JRdWVzdGlvbkFuc3dlcmluZ1NpbXBsZS5mcm9tX3ByZXRyYWluZWQoJTIyeGxuZXQlMkZ4bG5ldC1sYXJnZS1jYXNlZCUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLNetForQuestionAnsweringSimple
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetForQuestionAnsweringSimple.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-large-cased&quot;</span>)

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
...`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-11lpom8"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function qr(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=y},l(p){n=d(p,"P",{"data-svelte-h":!0}),h(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(p,c){m(p,n,c)},p:j,d(p){p&&r(n)}}}function Ir(w){let n,y="Example:",p,c,T;return c=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE5ldEZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ4bG5ldCUyRnhsbmV0LWJhc2UtY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBYTE5ldEZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJ4bG5ldCUyRnhsbmV0LWJhc2UtY2FzZWQlMjIpJTBBJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9yY2gudGVuc29yKHRva2VuaXplci5lbmNvZGUoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKS51bnNxdWVlemUoJTBBJTIwJTIwJTIwJTIwMCUwQSklMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBc3RhcnRfcG9zaXRpb25zJTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjElNUQpJTBBZW5kX3Bvc2l0aW9ucyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIzJTVEKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0RzdGFydF9wb3NpdGlvbnMlMkMlMjBlbmRfcG9zaXRpb25zJTNEZW5kX3Bvc2l0aW9ucyklMEElMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3Nz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLNetForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-base-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLNetForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;xlnet/xlnet-base-cased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = torch.tensor(tokenizer.encode(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, add_special_tokens=<span class="hljs-literal">True</span>)).unsqueeze(
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>start_positions = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>end_positions = torch.tensor([<span class="hljs-number">3</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`,wrap:!1}}),{c(){n=l("p"),n.textContent=y,p=s(),u(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),h(n)!=="svelte-11lpom8"&&(n.textContent=y),p=a(t),g(c.$$.fragment,t)},m(t,M){m(t,n,M),m(t,p,M),f(c,t,M),T=!0},p:j,i(t){T||(_(c.$$.fragment,t),T=!0)},o(t){b(c.$$.fragment,t),T=!1},d(t){t&&(r(n),r(p)),k(c,t)}}}function Wr(w){let n,y,p,c,T,t="<em>This model was released on 2019-06-19 and added to Hugging Face Transformers on 2020-11-16.</em>",M,Ve,Dn,Te,ma='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Kn,Qe,eo,Ae,ha=`The XLNet model was proposed in <a href="https://huggingface.co/papers/1906.08237" rel="nofollow">XLNet: Generalized Autoregressive Pretraining for Language Understanding</a> by Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov,
Quoc V. Le. XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn
bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization
order.`,to,Re,ua="The abstract from the paper is the following:",no,Oe,ga=`<em>With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves
better performance than pretraining approaches based on autoregressive language modeling. However, relying on
corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a
pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive
pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all
permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive
formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into
pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large
margin, including question answering, natural language inference, sentiment analysis, and document ranking.</em>`,oo,Ee,fa='This model was contributed by <a href="https://huggingface.co/thomwolf" rel="nofollow">thomwolf</a>. The original code can be found <a href="https://github.com/zihangdai/xlnet/" rel="nofollow">here</a>.',so,Ye,ao,Pe,_a=`<li>The specific attention pattern can be controlled at training and test time using the <code>perm_mask</code> input.</li> <li>Due to the difficulty of training a fully auto-regressive model over various factorization order, XLNet is pretrained
using only a sub-set of the output tokens as target which are selected with the <code>target_mapping</code> input.</li> <li>To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the <code>perm_mask</code> and
<code>target_mapping</code> inputs to control the attention span and outputs (see examples in
<em>examples/pytorch/text-generation/run_generation.py</em>)</li> <li>XLNet is one of the few models that has no sequence length limit.</li> <li>XLNet is not a traditional autoregressive model but uses a training strategy that builds on that. It permutes the tokens in the sentence, then allows the model to use the last n tokens to predict the token n+1. Since this is all done with a mask, the sentence is actually fed in the model in the right order, but instead of masking the first n tokens for n+1, XLNet uses a mask that hides the previous tokens in some given permutation of 1,…,sequence length.</li> <li>XLNet also uses the same recurrence mechanism as Transformer-XL to build long-term dependencies.</li>`,ro,De,io,Ke,ba='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',lo,et,co,B,tt,Bo,Pt,ka=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel">XLNetModel</a> or a <code>TFXLNetModel</code>. It is used to
instantiate a XLNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
<a href="https://huggingface.co/xlnet/xlnet-large-cased" rel="nofollow">xlnet/xlnet-large-cased</a> architecture.`,Go,Dt,Ta=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ho,ye,po,nt,mo,L,ot,So,Kt,ya='Construct an XLNet tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',Vo,en,Ma=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,Qo,K,st,Ao,tn,wa=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLNet sequence has the following format:`,Ro,nn,va="<li>single sequence: <code>X &lt;sep&gt; &lt;cls&gt;</code></li> <li>pair of sequences: <code>A &lt;sep&gt; B &lt;sep&gt; &lt;cls&gt;</code></li>",Oo,Me,at,Eo,on,xa=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Yo,S,rt,Po,sn,Na="Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet",Do,we,Ko,an,La="If <code>token_ids_1</code> is <code>None</code>, this method only returns the first portion of the mask (0s).",es,rn,it,ho,lt,uo,$,dt,ts,ln,Xa=`Construct a “fast” XLNet tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models" rel="nofollow">Unigram</a>.`,ns,dn,ja=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,os,ee,ct,ss,cn,$a=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLNet sequence has the following format:`,as,pn,Fa="<li>single sequence: <code>X &lt;sep&gt; &lt;cls&gt;</code></li> <li>pair of sequences: <code>A &lt;sep&gt; B &lt;sep&gt; &lt;cls&gt;</code></li>",rs,V,pt,is,mn,Ca="Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet",ls,ve,ds,hn,Ja="If <code>token_ids_1</code> is <code>None</code>, this method only returns the first portion of the mask (0s).",go,mt,fo,pe,ht,cs,un,za='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel">XLNetModel</a>.',_o,me,ut,ps,gn,Ua='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel">XLNetLMHeadModel</a>.',bo,he,gt,ms,fn,qa='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForSequenceClassification">XLNetForSequenceClassification</a>.',ko,ue,ft,hs,_n,Ia='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForMultipleChoice">XLNetForMultipleChoice</a>.',To,ge,_t,us,bn,Wa="Output type of <code>XLNetForTokenClassificationOutput</code>.",yo,fe,bt,gs,kn,Za='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple">XLNetForQuestionAnsweringSimple</a>.',Mo,_e,kt,fs,Tn,Ba='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering">XLNetForQuestionAnswering</a>.',wo,Tt,vo,F,yt,_s,yn,Ga="The bare Xlnet Model outputting raw hidden-states without any specific head on top.",bs,Mn,Ha=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ks,wn,Sa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ts,te,Mt,ys,vn,Va='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel">XLNetModel</a> forward method, overrides the <code>__call__</code> special method.',Ms,xe,xo,wt,No,C,vt,ws,xn,Qa="XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).",vs,Nn,Aa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xs,Ln,Ra=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ns,Q,xt,Ls,Xn,Oa='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel">XLNetLMHeadModel</a> forward method, overrides the <code>__call__</code> special method.',Xs,Ne,js,Le,Lo,Nt,Xo,J,Lt,$s,jn,Ea=`XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
for GLUE tasks.`,Fs,$n,Ya=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Cs,Fn,Pa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Js,Z,Xt,zs,Cn,Da='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForSequenceClassification">XLNetForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Us,Xe,qs,je,Is,$e,jo,jt,$o,z,$t,Ws,Jn,Ka=`The Xlnet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Zs,zn,er=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bs,Un,tr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Gs,A,Ft,Hs,qn,nr='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForMultipleChoice">XLNetForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Ss,Fe,Vs,Ce,Fo,Ct,Co,U,Jt,Qs,In,or=`The Xlnet transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,As,Wn,sr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rs,Zn,ar=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Os,R,zt,Es,Bn,rr='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForTokenClassification">XLNetForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Ys,Je,Ps,ze,Jo,Ut,zo,q,qt,Ds,Gn,ir=`XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Ks,Hn,lr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ea,Sn,dr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ta,O,It,na,Vn,cr='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple">XLNetForQuestionAnsweringSimple</a> forward method, overrides the <code>__call__</code> special method.',oa,Ue,sa,qe,Uo,Wt,qo,I,Zt,aa,Qn,pr=`The Xlnet transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ra,An,mr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ia,Rn,hr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,la,E,Bt,da,On,ur='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering">XLNetForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',ca,Ie,pa,We,Io,Gt,Wo,Yn,Zo;return Ve=new W({props:{title:"XLNet",local:"xlnet",headingTag:"h1"}}),Qe=new W({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ye=new W({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),De=new W({props:{title:"Resources",local:"resources",headingTag:"h2"}}),et=new W({props:{title:"XLNetConfig",local:"transformers.XLNetConfig",headingTag:"h2"}}),tt=new N({props:{name:"class transformers.XLNetConfig",anchor:"transformers.XLNetConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"d_model",val:" = 1024"},{name:"n_layer",val:" = 24"},{name:"n_head",val:" = 16"},{name:"d_inner",val:" = 4096"},{name:"ff_activation",val:" = 'gelu'"},{name:"untie_r",val:" = True"},{name:"attn_type",val:" = 'bi'"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"dropout",val:" = 0.1"},{name:"mem_len",val:" = 512"},{name:"reuse_len",val:" = None"},{name:"use_mems_eval",val:" = True"},{name:"use_mems_train",val:" = False"},{name:"bi_data",val:" = False"},{name:"clamp_len",val:" = -1"},{name:"same_length",val:" = False"},{name:"summary_type",val:" = 'last'"},{name:"summary_use_proj",val:" = True"},{name:"summary_activation",val:" = 'tanh'"},{name:"summary_last_dropout",val:" = 0.1"},{name:"start_n_top",val:" = 5"},{name:"end_n_top",val:" = 5"},{name:"pad_token_id",val:" = 5"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the XLNet model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel">XLNetModel</a> or <code>TFXLNetModel</code>.`,name:"vocab_size"},{anchor:"transformers.XLNetConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.XLNetConfig.n_layer",description:`<strong>n_layer</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"n_layer"},{anchor:"transformers.XLNetConfig.n_head",description:`<strong>n_head</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"n_head"},{anchor:"transformers.XLNetConfig.d_inner",description:`<strong>d_inner</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"d_inner"},{anchor:"transformers.XLNetConfig.ff_activation",description:`<strong>ff_activation</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the If string, <code>&quot;gelu&quot;</code>, <code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and
<code>&quot;gelu_new&quot;</code> are supported.`,name:"ff_activation"},{anchor:"transformers.XLNetConfig.untie_r",description:`<strong>untie_r</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to untie relative position biases`,name:"untie_r"},{anchor:"transformers.XLNetConfig.attn_type",description:`<strong>attn_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;bi&quot;</code>) &#x2014;
The attention type used by the model. Set <code>&quot;bi&quot;</code> for XLNet, <code>&quot;uni&quot;</code> for Transformer-XL.`,name:"attn_type"},{anchor:"transformers.XLNetConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.XLNetConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.XLNetConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.XLNetConfig.mem_len",description:`<strong>mem_len</strong> (<code>int</code> or <code>None</code>, <em>optional</em>) &#x2014;
The number of tokens to cache. The key/value pairs that have already been pre-computed in a previous
forward pass won&#x2019;t be re-computed. See the
<a href="https://huggingface.co/transformers/quickstart.html#using-the-past" rel="nofollow">quickstart</a> for more information.`,name:"mem_len"},{anchor:"transformers.XLNetConfig.reuse_len",description:`<strong>reuse_len</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The number of tokens in the current batch to be cached and reused in the future.`,name:"reuse_len"},{anchor:"transformers.XLNetConfig.bi_data",description:`<strong>bi_data</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use bidirectional input pipeline. Usually set to <code>True</code> during pretraining and <code>False</code>
during finetuning.`,name:"bi_data"},{anchor:"transformers.XLNetConfig.clamp_len",description:`<strong>clamp_len</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no clamping.`,name:"clamp_len"},{anchor:"transformers.XLNetConfig.same_length",description:`<strong>same_length</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use the same attention length for each token.`,name:"same_length"},{anchor:"transformers.XLNetConfig.summary_type",description:`<strong>summary_type</strong> (<code>str</code>, <em>optional</em>, defaults to &#x201C;last&#x201D;) &#x2014;
Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.</p>
<p>Has to be one of the following options:</p>
<ul>
<li><code>&quot;last&quot;</code>: Take the last token hidden state (like XLNet).</li>
<li><code>&quot;first&quot;</code>: Take the first token hidden state (like BERT).</li>
<li><code>&quot;mean&quot;</code>: Take the mean of all tokens hidden states.</li>
<li><code>&quot;cls_index&quot;</code>: Supply a Tensor of classification token position (like GPT/GPT-2).</li>
<li><code>&quot;attn&quot;</code>: Not implemented now, use multi-head attention.</li>
</ul>`,name:"summary_type"},{anchor:"transformers.XLNetConfig.summary_use_proj",description:`<strong>summary_use_proj</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.</p>
<p>Whether or not to add a projection after the vector extraction.`,name:"summary_use_proj"},{anchor:"transformers.XLNetConfig.summary_activation",description:`<strong>summary_activation</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.</p>
<p>Pass <code>&quot;tanh&quot;</code> for a tanh activation to the output, any other value will result in no activation.`,name:"summary_activation"},{anchor:"transformers.XLNetConfig.summary_proj_to_labels",description:`<strong>summary_proj_to_labels</strong> (<code>boo</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Used in the sequence classification and multiple choice models.</p>
<p>Whether the projection outputs should have <code>config.num_labels</code> or <code>config.hidden_size</code> classes.`,name:"summary_proj_to_labels"},{anchor:"transformers.XLNetConfig.summary_last_dropout",description:`<strong>summary_last_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Used in the sequence classification and multiple choice models.</p>
<p>The dropout ratio to be used after the projection and activation.`,name:"summary_last_dropout"},{anchor:"transformers.XLNetConfig.start_n_top",description:`<strong>start_n_top</strong> (<code>int</code>, <em>optional</em>, defaults to 5) &#x2014;
Used in the SQuAD evaluation script.`,name:"start_n_top"},{anchor:"transformers.XLNetConfig.end_n_top",description:`<strong>end_n_top</strong> (<code>int</code>, <em>optional</em>, defaults to 5) &#x2014;
Used in the SQuAD evaluation script.`,name:"end_n_top"},{anchor:"transformers.XLNetConfig.use_mems_eval",description:`<strong>use_mems_eval</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.`,name:"use_mems_eval"},{anchor:"transformers.XLNetConfig.use_mems_train",description:`<strong>use_mems_train</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the model should make use of the recurrent memory mechanism in train mode.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>For pretraining, it is recommended to set <code>use_mems_train</code> to <code>True</code>. For fine-tuning, it is recommended to
set <code>use_mems_train</code> to <code>False</code> as discussed
<a href="https://github.com/zihangdai/xlnet/issues/41#issuecomment-505102587" rel="nofollow">here</a>. If <code>use_mems_train</code> is set to
<code>True</code>, one has to make sure that the train batches are correctly pre-processed, <em>e.g.</em> <code>batch_1 = [[This line is], [This is the]]</code> and <code>batch_2 = [[ the first line], [ second line]]</code> and that all batches are of
equal size.</p>

					</div>`,name:"use_mems_train"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/configuration_xlnet.py#L27"}}),ye=new de({props:{anchor:"transformers.XLNetConfig.example",$$slots:{default:[yr]},$$scope:{ctx:w}}}),nt=new W({props:{title:"XLNetTokenizer",local:"transformers.XLNetTokenizer",headingTag:"h2"}}),ot=new N({props:{name:"class transformers.XLNetTokenizer",anchor:"transformers.XLNetTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = False"},{name:"remove_space",val:" = True"},{name:"keep_accents",val:" = False"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"sep_token",val:" = '<sep>'"},{name:"pad_token",val:" = '<pad>'"},{name:"cls_token",val:" = '<cls>'"},{name:"mask_token",val:" = '<mask>'"},{name:"additional_special_tokens",val:" = ['<eop>', '<eod>']"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a .spm extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.XLNetTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.XLNetTokenizer.remove_space",description:`<strong>remove_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to strip the text when tokenizing (removing excess spaces before and after the string).`,name:"remove_space"},{anchor:"transformers.XLNetTokenizer.keep_accents",description:`<strong>keep_accents</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to keep accents when tokenizing.`,name:"keep_accents"},{anchor:"transformers.XLNetTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.XLNetTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.XLNetTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.XLNetTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;sep&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.XLNetTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.XLNetTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;cls&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.XLNetTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.XLNetTokenizer.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&apos;&lt;eop&gt;&apos;, &apos;&lt;eod&gt;&apos;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"},{anchor:"transformers.XLNetTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.XLNetTokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L43"}}),st=new N({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.XLNetTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLNetTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.XLNetTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L286",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),at=new N({props:{name:"get_special_tokens_mask",anchor:"transformers.XLNetTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.XLNetTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XLNetTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.XLNetTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L311",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),rt=new N({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.XLNetTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLNetTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XLNetTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L339",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#token-type-ids">token type IDs</a> according to the given sequence(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),we=new de({props:{anchor:"transformers.XLNetTokenizer.create_token_type_ids_from_sequences.example",$$slots:{default:[Mr]},$$scope:{ctx:w}}}),it=new N({props:{name:"save_vocabulary",anchor:"transformers.XLNetTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet.py#L369"}}),lt=new W({props:{title:"XLNetTokenizerFast",local:"transformers.XLNetTokenizerFast",headingTag:"h2"}}),dt=new N({props:{name:"class transformers.XLNetTokenizerFast",anchor:"transformers.XLNetTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = False"},{name:"remove_space",val:" = True"},{name:"keep_accents",val:" = False"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"sep_token",val:" = '<sep>'"},{name:"pad_token",val:" = '<pad>'"},{name:"cls_token",val:" = '<cls>'"},{name:"mask_token",val:" = '<mask>'"},{name:"additional_special_tokens",val:" = ['<eop>', '<eod>']"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a .spm extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.XLNetTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.XLNetTokenizerFast.remove_space",description:`<strong>remove_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to strip the text when tokenizing (removing excess spaces before and after the string).`,name:"remove_space"},{anchor:"transformers.XLNetTokenizerFast.keep_accents",description:`<strong>keep_accents</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to keep accents when tokenizing.`,name:"keep_accents"},{anchor:"transformers.XLNetTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.XLNetTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.XLNetTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.XLNetTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;sep&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.XLNetTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.XLNetTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;cls&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.XLNetTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.XLNetTokenizerFast.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&quot;&lt;eop&gt;&quot;, &quot;&lt;eod&gt;&quot;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"},{anchor:"transformers.XLNetTokenizerFast.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L47"}}),ct=new N({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.XLNetTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLNetTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.XLNetTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L155",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),pt=new N({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.XLNetTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLNetTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XLNetTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/tokenization_xlnet_fast.py#L180",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#token-type-ids">token type IDs</a> according to the given sequence(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ve=new de({props:{anchor:"transformers.XLNetTokenizerFast.create_token_type_ids_from_sequences.example",$$slots:{default:[wr]},$$scope:{ctx:w}}}),mt=new W({props:{title:"XLNet specific outputs",local:"transformers.models.xlnet.modeling_xlnet.XLNetModelOutput",headingTag:"h2"}}),ht=new N({props:{name:"class transformers.models.xlnet.modeling_xlnet.XLNetModelOutput",anchor:"transformers.models.xlnet.modeling_xlnet.XLNetModelOutput",parameters:[{name:"last_hidden_state",val:": FloatTensor"},{name:"mems",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetModelOutput.last_hidden_state",description:`<strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, hidden_size)</code>) &#x2014;
Sequence of hidden-states at the last layer of the model.</p>
<p><code>num_predict</code> corresponds to <code>target_mapping.shape[1]</code>. If <code>target_mapping</code> is <code>None</code>, then <code>num_predict</code>
corresponds to <code>sequence_length</code>.`,name:"last_hidden_state"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetModelOutput.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.`,name:"mems"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetModelOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetModelOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L841"}}),ut=new N({props:{name:"class transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput",anchor:"transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mems",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) &#x2014;
Language modeling loss (for next-token prediction).`,name:"loss"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
<p><code>num_predict</code> corresponds to <code>target_mapping.shape[1]</code>. If <code>target_mapping</code> is <code>None</code>, then <code>num_predict</code>
corresponds to <code>sequence_length</code>.`,name:"logits"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.`,name:"mems"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L866"}}),gt=new N({props:{name:"class transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput",anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mems",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>label</code> is provided) &#x2014;
Classification (or regression if config.num_labels==1) loss.`,name:"loss"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) &#x2014;
Classification (or regression if config.num_labels==1) scores (before SoftMax).`,name:"logits"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.`,name:"mems"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L894"}}),ft=new N({props:{name:"class transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput",anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mems",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) &#x2014;
Classification loss.`,name:"loss"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) &#x2014;
<em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).`,name:"logits"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.`,name:"mems"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L944"}}),_t=new N({props:{name:"class transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput",anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mems",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) &#x2014;
Classification loss.`,name:"loss"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) &#x2014;
Classification scores (before SoftMax).`,name:"logits"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.`,name:"mems"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L919"}}),bt=new N({props:{name:"class transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput",anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"end_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mems",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) &#x2014;
Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.`,name:"loss"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput.start_logits",description:`<strong>start_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length,)</code>) &#x2014;
Span-start scores (before SoftMax).`,name:"start_logits"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput.end_logits",description:`<strong>end_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length,)</code>) &#x2014;
Span-end scores (before SoftMax).`,name:"end_logits"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.`,name:"mems"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L971"}}),kt=new N({props:{name:"class transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput",anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_top_log_probs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_top_index",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_top_log_probs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"end_top_index",val:": typing.Optional[torch.LongTensor] = None"},{name:"cls_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mems",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned if both <code>start_positions</code> and <code>end_positions</code> are provided) &#x2014;
Classification loss as the sum of start token, end token (and is_impossible if provided) classification
losses.`,name:"loss"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.start_top_log_probs",description:`<strong>start_top_log_probs</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.start_n_top)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) &#x2014;
Log probabilities for the top config.start_n_top start token possibilities (beam-search).`,name:"start_top_log_probs"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.start_top_index",description:`<strong>start_top_index</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.start_n_top)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) &#x2014;
Indices for the top config.start_n_top start token possibilities (beam-search).`,name:"start_top_index"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.end_top_log_probs",description:`<strong>end_top_log_probs</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.start_n_top * config.end_n_top)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) &#x2014;
Log probabilities for the top <code>config.start_n_top * config.end_n_top</code> end token possibilities
(beam-search).`,name:"end_top_log_probs"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.end_top_index",description:`<strong>end_top_index</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.start_n_top * config.end_n_top)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) &#x2014;
Indices for the top <code>config.start_n_top * config.end_n_top</code> end token possibilities (beam-search).`,name:"end_top_index"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.cls_logits",description:`<strong>cls_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) &#x2014;
Log probabilities for the <code>is_impossible</code> label of the answers.`,name:"cls_logits"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.`,name:"mems"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L999"}}),Tt=new W({props:{title:"XLNetModel",local:"transformers.XLNetModel",headingTag:"h2"}}),yt=new N({props:{name:"class transformers.XLNetModel",anchor:"transformers.XLNetModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLNetModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetModel">XLNetModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1033"}}),Mt=new N({props:{name:"forward",anchor:"transformers.XLNetModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mems",val:": typing.Optional[torch.Tensor] = None"},{name:"perm_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"target_mapping",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_mems",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLNetModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLNetModel.forward.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states (see <code>mems</code> output below) . Can be used to speed up sequential
decoding. The token ids which have their past given to this model should not be passed as <code>input_ids</code> as
they have already been computed.</p>
<p><code>use_mems</code> has to be set to <code>True</code> to make use of <code>mems</code>.`,name:"mems"},{anchor:"transformers.XLNetModel.forward.perm_mask",description:`<strong>perm_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the attention pattern for each input token with values selected in <code>[0, 1]</code>:</p>
<ul>
<li>if <code>perm_mask[k, i, j] = 0</code>, i attend to j in batch k;</li>
<li>if <code>perm_mask[k, i, j] = 1</code>, i does not attend to j in batch k.</li>
</ul>
<p>If not set, each token attends to all the others (full bidirectional attention). Only used during
pretraining (to define factorization order) or for sequential decoding (generation).`,name:"perm_mask"},{anchor:"transformers.XLNetModel.forward.target_mapping",description:`<strong>target_mapping</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the output tokens to use. If <code>target_mapping[k, i, j] = 1</code>, the i-th predict in batch k is
on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
(generation).`,name:"target_mapping"},{anchor:"transformers.XLNetModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLNetModel.forward.input_mask",description:`<strong>input_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Negative of <code>attention_mask</code>, i.e. with 0 for
real tokens and 1 for padding which is kept for compatibility with the original code base.</p>
<p>Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>masked</strong>,</li>
<li>0 for tokens that are <strong>not masked</strong>.</li>
</ul>
<p>You can only uses one of <code>input_mask</code> and <code>attention_mask</code>.`,name:"input_mask"},{anchor:"transformers.XLNetModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLNetModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLNetModel.forward.use_mems",description:`<strong>use_mems</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to use memory states to speed up sequential decoding. If set to <code>True</code>, the model will use the hidden
states from previous forward passes to compute attention, which can significantly improve performance for
sequential decoding tasks.`,name:"use_mems"},{anchor:"transformers.XLNetModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLNetModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLNetModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1161",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig"
>XLNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, hidden_size)</code>) — Sequence of hidden-states at the last layer of the model.</p>
<p><code>num_predict</code> corresponds to <code>target_mapping.shape[1]</code>. If <code>target_mapping</code> is <code>None</code>, then <code>num_predict</code>
corresponds to <code>sequence_length</code>.</p>
</li>
<li>
<p><strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) — Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetModelOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),xe=new Yt({props:{$$slots:{default:[vr]},$$scope:{ctx:w}}}),wt=new W({props:{title:"XLNetLMHeadModel",local:"transformers.XLNetLMHeadModel",headingTag:"h2"}}),vt=new N({props:{name:"class transformers.XLNetLMHeadModel",anchor:"transformers.XLNetLMHeadModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLNetLMHeadModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetLMHeadModel">XLNetLMHeadModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1424"}}),xt=new N({props:{name:"forward",anchor:"transformers.XLNetLMHeadModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mems",val:": typing.Optional[torch.Tensor] = None"},{name:"perm_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"target_mapping",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_mems",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetLMHeadModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLNetLMHeadModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLNetLMHeadModel.forward.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states (see <code>mems</code> output below) . Can be used to speed up sequential
decoding. The token ids which have their past given to this model should not be passed as <code>input_ids</code> as
they have already been computed.</p>
<p><code>use_mems</code> has to be set to <code>True</code> to make use of <code>mems</code>.`,name:"mems"},{anchor:"transformers.XLNetLMHeadModel.forward.perm_mask",description:`<strong>perm_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the attention pattern for each input token with values selected in <code>[0, 1]</code>:</p>
<ul>
<li>if <code>perm_mask[k, i, j] = 0</code>, i attend to j in batch k;</li>
<li>if <code>perm_mask[k, i, j] = 1</code>, i does not attend to j in batch k.</li>
</ul>
<p>If not set, each token attends to all the others (full bidirectional attention). Only used during
pretraining (to define factorization order) or for sequential decoding (generation).`,name:"perm_mask"},{anchor:"transformers.XLNetLMHeadModel.forward.target_mapping",description:`<strong>target_mapping</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the output tokens to use. If <code>target_mapping[k, i, j] = 1</code>, the i-th predict in batch k is
on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
(generation).`,name:"target_mapping"},{anchor:"transformers.XLNetLMHeadModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLNetLMHeadModel.forward.input_mask",description:`<strong>input_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Negative of <code>attention_mask</code>, i.e. with 0 for
real tokens and 1 for padding which is kept for compatibility with the original code base.</p>
<p>Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>masked</strong>,</li>
<li>0 for tokens that are <strong>not masked</strong>.</li>
</ul>
<p>You can only uses one of <code>input_mask</code> and <code>attention_mask</code>.`,name:"input_mask"},{anchor:"transformers.XLNetLMHeadModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLNetLMHeadModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLNetLMHeadModel.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_predict)</code>, <em>optional</em>) &#x2014;
Labels for masked language modeling. <code>num_predict</code> corresponds to <code>target_mapping.shape[1]</code>. If
<code>target_mapping</code> is <code>None</code>, then <code>num_predict</code> corresponds to <code>sequence_length</code>.</p>
<p>The labels should correspond to the masked input words that should be predicted and depends on
<code>target_mapping</code>. Note in order to perform standard auto-regressive language modeling a <em><mask></mask></em> token has
to be added to the <code>input_ids</code> (see the <code>prepare_inputs_for_generation</code> function and examples below)</p>
<p>Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code> are ignored, the loss
is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.XLNetLMHeadModel.forward.use_mems",description:`<strong>use_mems</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to use memory states to speed up sequential decoding. If set to <code>True</code>, the model will use the hidden
states from previous forward passes to compute attention, which can significantly improve performance for
sequential decoding tasks.`,name:"use_mems"},{anchor:"transformers.XLNetLMHeadModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLNetLMHeadModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLNetLMHeadModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1488",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig"
>XLNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
<p><code>num_predict</code> corresponds to <code>target_mapping.shape[1]</code>. If <code>target_mapping</code> is <code>None</code>, then <code>num_predict</code>
corresponds to <code>sequence_length</code>.</p>
</li>
<li>
<p><strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) — Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ne=new Yt({props:{$$slots:{default:[xr]},$$scope:{ctx:w}}}),Le=new de({props:{anchor:"transformers.XLNetLMHeadModel.forward.example",$$slots:{default:[Nr]},$$scope:{ctx:w}}}),Nt=new W({props:{title:"XLNetForSequenceClassification",local:"transformers.XLNetForSequenceClassification",headingTag:"h2"}}),Lt=new N({props:{name:"class transformers.XLNetForSequenceClassification",anchor:"transformers.XLNetForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLNetForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForSequenceClassification">XLNetForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1660"}}),Xt=new N({props:{name:"forward",anchor:"transformers.XLNetForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mems",val:": typing.Optional[torch.Tensor] = None"},{name:"perm_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"target_mapping",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_mems",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLNetForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLNetForSequenceClassification.forward.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states (see <code>mems</code> output below) . Can be used to speed up sequential
decoding. The token ids which have their past given to this model should not be passed as <code>input_ids</code> as
they have already been computed.</p>
<p><code>use_mems</code> has to be set to <code>True</code> to make use of <code>mems</code>.`,name:"mems"},{anchor:"transformers.XLNetForSequenceClassification.forward.perm_mask",description:`<strong>perm_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the attention pattern for each input token with values selected in <code>[0, 1]</code>:</p>
<ul>
<li>if <code>perm_mask[k, i, j] = 0</code>, i attend to j in batch k;</li>
<li>if <code>perm_mask[k, i, j] = 1</code>, i does not attend to j in batch k.</li>
</ul>
<p>If not set, each token attends to all the others (full bidirectional attention). Only used during
pretraining (to define factorization order) or for sequential decoding (generation).`,name:"perm_mask"},{anchor:"transformers.XLNetForSequenceClassification.forward.target_mapping",description:`<strong>target_mapping</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the output tokens to use. If <code>target_mapping[k, i, j] = 1</code>, the i-th predict in batch k is
on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
(generation).`,name:"target_mapping"},{anchor:"transformers.XLNetForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLNetForSequenceClassification.forward.input_mask",description:`<strong>input_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Negative of <code>attention_mask</code>, i.e. with 0 for
real tokens and 1 for padding which is kept for compatibility with the original code base.</p>
<p>Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>masked</strong>,</li>
<li>0 for tokens that are <strong>not masked</strong>.</li>
</ul>
<p>You can only uses one of <code>input_mask</code> and <code>attention_mask</code>.`,name:"input_mask"},{anchor:"transformers.XLNetForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLNetForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLNetForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.XLNetForSequenceClassification.forward.use_mems",description:`<strong>use_mems</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to use memory states to speed up sequential decoding. If set to <code>True</code>, the model will use the hidden
states from previous forward passes to compute attention, which can significantly improve performance for
sequential decoding tasks.`,name:"use_mems"},{anchor:"transformers.XLNetForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLNetForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLNetForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1673",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig"
>XLNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>label</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) — Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForSequenceClassificationOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Xe=new Yt({props:{$$slots:{default:[Lr]},$$scope:{ctx:w}}}),je=new de({props:{anchor:"transformers.XLNetForSequenceClassification.forward.example",$$slots:{default:[Xr]},$$scope:{ctx:w}}}),$e=new de({props:{anchor:"transformers.XLNetForSequenceClassification.forward.example-2",$$slots:{default:[jr]},$$scope:{ctx:w}}}),jt=new W({props:{title:"XLNetForMultipleChoice",local:"transformers.XLNetForMultipleChoice",headingTag:"h2"}}),$t=new N({props:{name:"class transformers.XLNetForMultipleChoice",anchor:"transformers.XLNetForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLNetForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForMultipleChoice">XLNetForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1900"}}),Ft=new N({props:{name:"forward",anchor:"transformers.XLNetForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mems",val:": typing.Optional[torch.Tensor] = None"},{name:"perm_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"target_mapping",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_mems",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLNetForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLNetForMultipleChoice.forward.input_mask",description:`<strong>input_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, num_choices, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Negative of <code>attention_mask</code>, i.e. with 0 for
real tokens and 1 for padding which is kept for compatibility with the original code base.</p>
<p>Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>masked</strong>,</li>
<li>0 for tokens that are <strong>not masked</strong>.</li>
</ul>
<p>You can only uses one of <code>input_mask</code> and <code>attention_mask</code>.`,name:"input_mask"},{anchor:"transformers.XLNetForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLNetForMultipleChoice.forward.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states (see <code>mems</code> output below) . Can be used to speed up sequential
decoding. The token ids which have their past given to this model should not be passed as <code>input_ids</code> as
they have already been computed.</p>
<p><code>use_mems</code> has to be set to <code>True</code> to make use of <code>mems</code>.`,name:"mems"},{anchor:"transformers.XLNetForMultipleChoice.forward.perm_mask",description:`<strong>perm_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the attention pattern for each input token with values selected in <code>[0, 1]</code>:</p>
<ul>
<li>if <code>perm_mask[k, i, j] = 0</code>, i attend to j in batch k;</li>
<li>if <code>perm_mask[k, i, j] = 1</code>, i does not attend to j in batch k.</li>
</ul>
<p>If not set, each token attends to all the others (full bidirectional attention). Only used during
pretraining (to define factorization order) or for sequential decoding (generation).`,name:"perm_mask"},{anchor:"transformers.XLNetForMultipleChoice.forward.target_mapping",description:`<strong>target_mapping</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the output tokens to use. If <code>target_mapping[k, i, j] = 1</code>, the i-th predict in batch k is
on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
(generation).`,name:"target_mapping"},{anchor:"transformers.XLNetForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLNetForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLNetForMultipleChoice.forward.labels",description:"<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;\nLabels for computing the multiple choice classification loss. Indices should be in `[0, &#x2026;,",name:"labels"},{anchor:"transformers.XLNetForMultipleChoice.forward.use_mems",description:`<strong>use_mems</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to use memory states to speed up sequential decoding. If set to <code>True</code>, the model will use the hidden
states from previous forward passes to compute attention, which can significantly improve performance for
sequential decoding tasks.`,name:"use_mems"},{anchor:"transformers.XLNetForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLNetForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLNetForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1911",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig"
>XLNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) — <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) — Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Fe=new Yt({props:{$$slots:{default:[$r]},$$scope:{ctx:w}}}),Ce=new de({props:{anchor:"transformers.XLNetForMultipleChoice.forward.example",$$slots:{default:[Fr]},$$scope:{ctx:w}}}),Ct=new W({props:{title:"XLNetForTokenClassification",local:"transformers.XLNetForTokenClassification",headingTag:"h2"}}),Jt=new N({props:{name:"class transformers.XLNetForTokenClassification",anchor:"transformers.XLNetForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLNetForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForTokenClassification">XLNetForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1790"}}),zt=new N({props:{name:"forward",anchor:"transformers.XLNetForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mems",val:": typing.Optional[torch.Tensor] = None"},{name:"perm_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"target_mapping",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_mems",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLNetForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLNetForTokenClassification.forward.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states (see <code>mems</code> output below) . Can be used to speed up sequential
decoding. The token ids which have their past given to this model should not be passed as <code>input_ids</code> as
they have already been computed.</p>
<p><code>use_mems</code> has to be set to <code>True</code> to make use of <code>mems</code>.`,name:"mems"},{anchor:"transformers.XLNetForTokenClassification.forward.perm_mask",description:`<strong>perm_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the attention pattern for each input token with values selected in <code>[0, 1]</code>:</p>
<ul>
<li>if <code>perm_mask[k, i, j] = 0</code>, i attend to j in batch k;</li>
<li>if <code>perm_mask[k, i, j] = 1</code>, i does not attend to j in batch k.</li>
</ul>
<p>If not set, each token attends to all the others (full bidirectional attention). Only used during
pretraining (to define factorization order) or for sequential decoding (generation).`,name:"perm_mask"},{anchor:"transformers.XLNetForTokenClassification.forward.target_mapping",description:`<strong>target_mapping</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the output tokens to use. If <code>target_mapping[k, i, j] = 1</code>, the i-th predict in batch k is
on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
(generation).`,name:"target_mapping"},{anchor:"transformers.XLNetForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLNetForTokenClassification.forward.input_mask",description:`<strong>input_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Negative of <code>attention_mask</code>, i.e. with 0 for
real tokens and 1 for padding which is kept for compatibility with the original code base.</p>
<p>Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>masked</strong>,</li>
<li>0 for tokens that are <strong>not masked</strong>.</li>
</ul>
<p>You can only uses one of <code>input_mask</code> and <code>attention_mask</code>.`,name:"input_mask"},{anchor:"transformers.XLNetForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLNetForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLNetForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices]</code>
where <em>num_choices</em> is the size of the second dimension of the input tensors. (see <em>input_ids</em> above)`,name:"labels"},{anchor:"transformers.XLNetForTokenClassification.forward.use_mems",description:`<strong>use_mems</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to use memory states to speed up sequential decoding. If set to <code>True</code>, the model will use the hidden
states from previous forward passes to compute attention, which can significantly improve performance for
sequential decoding tasks.emory states to speed up sequential decoding. If set to <code>True</code>, the model will use the hidden
states from previous forward passes to compute attention, which can significantly improve performance for
sequential decoding tasks.`,name:"use_mems"},{anchor:"transformers.XLNetForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLNetForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLNetForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L1801",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig"
>XLNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) — Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) — Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Je=new Yt({props:{$$slots:{default:[Cr]},$$scope:{ctx:w}}}),ze=new de({props:{anchor:"transformers.XLNetForTokenClassification.forward.example",$$slots:{default:[Jr]},$$scope:{ctx:w}}}),Ut=new W({props:{title:"XLNetForQuestionAnsweringSimple",local:"transformers.XLNetForQuestionAnsweringSimple",headingTag:"h2"}}),qt=new N({props:{name:"class transformers.XLNetForQuestionAnsweringSimple",anchor:"transformers.XLNetForQuestionAnsweringSimple",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLNetForQuestionAnsweringSimple.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnsweringSimple">XLNetForQuestionAnsweringSimple</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L2046"}}),It=new N({props:{name:"forward",anchor:"transformers.XLNetForQuestionAnsweringSimple.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mems",val:": typing.Optional[torch.Tensor] = None"},{name:"perm_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"target_mapping",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"use_mems",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states (see <code>mems</code> output below) . Can be used to speed up sequential
decoding. The token ids which have their past given to this model should not be passed as <code>input_ids</code> as
they have already been computed.</p>
<p><code>use_mems</code> has to be set to <code>True</code> to make use of <code>mems</code>.`,name:"mems"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.perm_mask",description:`<strong>perm_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the attention pattern for each input token with values selected in <code>[0, 1]</code>:</p>
<ul>
<li>if <code>perm_mask[k, i, j] = 0</code>, i attend to j in batch k;</li>
<li>if <code>perm_mask[k, i, j] = 1</code>, i does not attend to j in batch k.</li>
</ul>
<p>If not set, each token attends to all the others (full bidirectional attention). Only used during
pretraining (to define factorization order) or for sequential decoding (generation).`,name:"perm_mask"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.target_mapping",description:`<strong>target_mapping</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the output tokens to use. If <code>target_mapping[k, i, j] = 1</code>, the i-th predict in batch k is
on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
(generation).`,name:"target_mapping"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.input_mask",description:`<strong>input_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Negative of <code>attention_mask</code>, i.e. with 0 for
real tokens and 1 for padding which is kept for compatibility with the original code base.</p>
<p>Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>masked</strong>,</li>
<li>0 for tokens that are <strong>not masked</strong>.</li>
</ul>
<p>You can only uses one of <code>input_mask</code> and <code>attention_mask</code>.`,name:"input_mask"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.use_mems",description:`<strong>use_mems</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to use memory states to speed up sequential decoding. If set to <code>True</code>, the model will use the hidden
states from previous forward passes to compute attention, which can significantly improve performance for
sequential decoding tasks.`,name:"use_mems"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L2057",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig"
>XLNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.</p>
</li>
<li>
<p><strong>start_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length,)</code>) — Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length,)</code>) — Span-end scores (before SoftMax).</p>
</li>
<li>
<p><strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) — Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ue=new Yt({props:{$$slots:{default:[zr]},$$scope:{ctx:w}}}),qe=new de({props:{anchor:"transformers.XLNetForQuestionAnsweringSimple.forward.example",$$slots:{default:[Ur]},$$scope:{ctx:w}}}),Wt=new W({props:{title:"XLNetForQuestionAnswering",local:"transformers.XLNetForQuestionAnswering",headingTag:"h2"}}),Zt=new N({props:{name:"class transformers.XLNetForQuestionAnswering",anchor:"transformers.XLNetForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLNetForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetForQuestionAnswering">XLNetForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L2169"}}),Bt=new N({props:{name:"forward",anchor:"transformers.XLNetForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"mems",val:": typing.Optional[torch.Tensor] = None"},{name:"perm_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"target_mapping",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"is_impossible",val:": typing.Optional[torch.Tensor] = None"},{name:"cls_index",val:": typing.Optional[torch.Tensor] = None"},{name:"p_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"use_mems",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLNetForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLNetForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLNetForQuestionAnswering.forward.mems",description:`<strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) &#x2014;
Contains pre-computed hidden-states (see <code>mems</code> output below) . Can be used to speed up sequential
decoding. The token ids which have their past given to this model should not be passed as <code>input_ids</code> as
they have already been computed.</p>
<p><code>use_mems</code> has to be set to <code>True</code> to make use of <code>mems</code>.`,name:"mems"},{anchor:"transformers.XLNetForQuestionAnswering.forward.perm_mask",description:`<strong>perm_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the attention pattern for each input token with values selected in <code>[0, 1]</code>:</p>
<ul>
<li>if <code>perm_mask[k, i, j] = 0</code>, i attend to j in batch k;</li>
<li>if <code>perm_mask[k, i, j] = 1</code>, i does not attend to j in batch k.</li>
</ul>
<p>If not set, each token attends to all the others (full bidirectional attention). Only used during
pretraining (to define factorization order) or for sequential decoding (generation).`,name:"perm_mask"},{anchor:"transformers.XLNetForQuestionAnswering.forward.target_mapping",description:`<strong>target_mapping</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_predict, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to indicate the output tokens to use. If <code>target_mapping[k, i, j] = 1</code>, the i-th predict in batch k is
on the j-th token. Only used during pretraining for partial prediction or for sequential decoding
(generation).`,name:"target_mapping"},{anchor:"transformers.XLNetForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLNetForQuestionAnswering.forward.input_mask",description:`<strong>input_mask</strong> (<code>torch.FloatTensor</code> of shape <code>batch_size, sequence_length</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Negative of <code>attention_mask</code>, i.e. with 0 for
real tokens and 1 for padding which is kept for compatibility with the original code base.</p>
<p>Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>masked</strong>,</li>
<li>0 for tokens that are <strong>not masked</strong>.</li>
</ul>
<p>You can only uses one of <code>input_mask</code> and <code>attention_mask</code>.`,name:"input_mask"},{anchor:"transformers.XLNetForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLNetForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLNetForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.XLNetForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.XLNetForQuestionAnswering.forward.is_impossible",description:`<strong>is_impossible</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels whether a question has an answer or no answer (SQuAD 2.0)`,name:"is_impossible"},{anchor:"transformers.XLNetForQuestionAnswering.forward.cls_index",description:`<strong>cls_index</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the classification token to use as input for computing plausibility of the
answer.`,name:"cls_index"},{anchor:"transformers.XLNetForQuestionAnswering.forward.p_mask",description:`<strong>p_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Optional mask of tokens which can&#x2019;t be in answers (e.g. [CLS], [PAD], &#x2026;). 1.0 means token should be
masked. 0.0 mean token is not masked.`,name:"p_mask"},{anchor:"transformers.XLNetForQuestionAnswering.forward.use_mems",description:`<strong>use_mems</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to use memory states to speed up sequential decoding. If set to <code>True</code>, the model will use the hidden
states from previous forward passes to compute attention, which can significantly improve performance for
sequential decoding tasks.`,name:"use_mems"},{anchor:"transformers.XLNetForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLNetForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLNetForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlnet/modeling_xlnet.py#L2183",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetConfig"
>XLNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned if both <code>start_positions</code> and <code>end_positions</code> are provided) — Classification loss as the sum of start token, end token (and is_impossible if provided) classification
losses.</p>
</li>
<li>
<p><strong>start_top_log_probs</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.start_n_top)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) — Log probabilities for the top config.start_n_top start token possibilities (beam-search).</p>
</li>
<li>
<p><strong>start_top_index</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.start_n_top)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) — Indices for the top config.start_n_top start token possibilities (beam-search).</p>
</li>
<li>
<p><strong>end_top_log_probs</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.start_n_top * config.end_n_top)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) — Log probabilities for the top <code>config.start_n_top * config.end_n_top</code> end token possibilities
(beam-search).</p>
</li>
<li>
<p><strong>end_top_index</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.start_n_top * config.end_n_top)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) — Indices for the top <code>config.start_n_top * config.end_n_top</code> end token possibilities (beam-search).</p>
</li>
<li>
<p><strong>cls_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>, returned if <code>start_positions</code> or <code>end_positions</code> is not provided) — Log probabilities for the <code>is_impossible</code> label of the answers.</p>
</li>
<li>
<p><strong>mems</strong> (<code>list[torch.FloatTensor]</code> of length <code>config.n_layers</code>) — Contains pre-computed hidden-states. Can be used (see <code>mems</code> input) to speed up sequential decoding. The
token ids which have their past given to this model should not be passed as <code>input_ids</code> as they have
already been computed.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput"
>transformers.models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ie=new Yt({props:{$$slots:{default:[qr]},$$scope:{ctx:w}}}),We=new de({props:{anchor:"transformers.XLNetForQuestionAnswering.forward.example",$$slots:{default:[Ir]},$$scope:{ctx:w}}}),Gt=new Tr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xlnet.md"}}),{c(){n=l("meta"),y=s(),p=l("p"),c=s(),T=l("p"),T.innerHTML=t,M=s(),u(Ve.$$.fragment),Dn=s(),Te=l("div"),Te.innerHTML=ma,Kn=s(),u(Qe.$$.fragment),eo=s(),Ae=l("p"),Ae.innerHTML=ha,to=s(),Re=l("p"),Re.textContent=ua,no=s(),Oe=l("p"),Oe.innerHTML=ga,oo=s(),Ee=l("p"),Ee.innerHTML=fa,so=s(),u(Ye.$$.fragment),ao=s(),Pe=l("ul"),Pe.innerHTML=_a,ro=s(),u(De.$$.fragment),io=s(),Ke=l("ul"),Ke.innerHTML=ba,lo=s(),u(et.$$.fragment),co=s(),B=l("div"),u(tt.$$.fragment),Bo=s(),Pt=l("p"),Pt.innerHTML=ka,Go=s(),Dt=l("p"),Dt.innerHTML=Ta,Ho=s(),u(ye.$$.fragment),po=s(),u(nt.$$.fragment),mo=s(),L=l("div"),u(ot.$$.fragment),So=s(),Kt=l("p"),Kt.innerHTML=ya,Vo=s(),en=l("p"),en.innerHTML=Ma,Qo=s(),K=l("div"),u(st.$$.fragment),Ao=s(),tn=l("p"),tn.textContent=wa,Ro=s(),nn=l("ul"),nn.innerHTML=va,Oo=s(),Me=l("div"),u(at.$$.fragment),Eo=s(),on=l("p"),on.innerHTML=xa,Yo=s(),S=l("div"),u(rt.$$.fragment),Po=s(),sn=l("p"),sn.textContent=Na,Do=s(),u(we.$$.fragment),Ko=s(),an=l("p"),an.innerHTML=La,es=s(),rn=l("div"),u(it.$$.fragment),ho=s(),u(lt.$$.fragment),uo=s(),$=l("div"),u(dt.$$.fragment),ts=s(),ln=l("p"),ln.innerHTML=Xa,ns=s(),dn=l("p"),dn.innerHTML=ja,os=s(),ee=l("div"),u(ct.$$.fragment),ss=s(),cn=l("p"),cn.textContent=$a,as=s(),pn=l("ul"),pn.innerHTML=Fa,rs=s(),V=l("div"),u(pt.$$.fragment),is=s(),mn=l("p"),mn.textContent=Ca,ls=s(),u(ve.$$.fragment),ds=s(),hn=l("p"),hn.innerHTML=Ja,go=s(),u(mt.$$.fragment),fo=s(),pe=l("div"),u(ht.$$.fragment),cs=s(),un=l("p"),un.innerHTML=za,_o=s(),me=l("div"),u(ut.$$.fragment),ps=s(),gn=l("p"),gn.innerHTML=Ua,bo=s(),he=l("div"),u(gt.$$.fragment),ms=s(),fn=l("p"),fn.innerHTML=qa,ko=s(),ue=l("div"),u(ft.$$.fragment),hs=s(),_n=l("p"),_n.innerHTML=Ia,To=s(),ge=l("div"),u(_t.$$.fragment),us=s(),bn=l("p"),bn.innerHTML=Wa,yo=s(),fe=l("div"),u(bt.$$.fragment),gs=s(),kn=l("p"),kn.innerHTML=Za,Mo=s(),_e=l("div"),u(kt.$$.fragment),fs=s(),Tn=l("p"),Tn.innerHTML=Ba,wo=s(),u(Tt.$$.fragment),vo=s(),F=l("div"),u(yt.$$.fragment),_s=s(),yn=l("p"),yn.textContent=Ga,bs=s(),Mn=l("p"),Mn.innerHTML=Ha,ks=s(),wn=l("p"),wn.innerHTML=Sa,Ts=s(),te=l("div"),u(Mt.$$.fragment),ys=s(),vn=l("p"),vn.innerHTML=Va,Ms=s(),u(xe.$$.fragment),xo=s(),u(wt.$$.fragment),No=s(),C=l("div"),u(vt.$$.fragment),ws=s(),xn=l("p"),xn.textContent=Qa,vs=s(),Nn=l("p"),Nn.innerHTML=Aa,xs=s(),Ln=l("p"),Ln.innerHTML=Ra,Ns=s(),Q=l("div"),u(xt.$$.fragment),Ls=s(),Xn=l("p"),Xn.innerHTML=Oa,Xs=s(),u(Ne.$$.fragment),js=s(),u(Le.$$.fragment),Lo=s(),u(Nt.$$.fragment),Xo=s(),J=l("div"),u(Lt.$$.fragment),$s=s(),jn=l("p"),jn.textContent=Ea,Fs=s(),$n=l("p"),$n.innerHTML=Ya,Cs=s(),Fn=l("p"),Fn.innerHTML=Pa,Js=s(),Z=l("div"),u(Xt.$$.fragment),zs=s(),Cn=l("p"),Cn.innerHTML=Da,Us=s(),u(Xe.$$.fragment),qs=s(),u(je.$$.fragment),Is=s(),u($e.$$.fragment),jo=s(),u(jt.$$.fragment),$o=s(),z=l("div"),u($t.$$.fragment),Ws=s(),Jn=l("p"),Jn.textContent=Ka,Zs=s(),zn=l("p"),zn.innerHTML=er,Bs=s(),Un=l("p"),Un.innerHTML=tr,Gs=s(),A=l("div"),u(Ft.$$.fragment),Hs=s(),qn=l("p"),qn.innerHTML=nr,Ss=s(),u(Fe.$$.fragment),Vs=s(),u(Ce.$$.fragment),Fo=s(),u(Ct.$$.fragment),Co=s(),U=l("div"),u(Jt.$$.fragment),Qs=s(),In=l("p"),In.textContent=or,As=s(),Wn=l("p"),Wn.innerHTML=sr,Rs=s(),Zn=l("p"),Zn.innerHTML=ar,Os=s(),R=l("div"),u(zt.$$.fragment),Es=s(),Bn=l("p"),Bn.innerHTML=rr,Ys=s(),u(Je.$$.fragment),Ps=s(),u(ze.$$.fragment),Jo=s(),u(Ut.$$.fragment),zo=s(),q=l("div"),u(qt.$$.fragment),Ds=s(),Gn=l("p"),Gn.innerHTML=ir,Ks=s(),Hn=l("p"),Hn.innerHTML=lr,ea=s(),Sn=l("p"),Sn.innerHTML=dr,ta=s(),O=l("div"),u(It.$$.fragment),na=s(),Vn=l("p"),Vn.innerHTML=cr,oa=s(),u(Ue.$$.fragment),sa=s(),u(qe.$$.fragment),Uo=s(),u(Wt.$$.fragment),qo=s(),I=l("div"),u(Zt.$$.fragment),aa=s(),Qn=l("p"),Qn.innerHTML=pr,ra=s(),An=l("p"),An.innerHTML=mr,ia=s(),Rn=l("p"),Rn.innerHTML=hr,la=s(),E=l("div"),u(Bt.$$.fragment),da=s(),On=l("p"),On.innerHTML=ur,ca=s(),u(Ie.$$.fragment),pa=s(),u(We.$$.fragment),Io=s(),u(Gt.$$.fragment),Wo=s(),Yn=l("p"),this.h()},l(e){const i=kr("svelte-u9bgzb",document.head);n=d(i,"META",{name:!0,content:!0}),i.forEach(r),y=a(e),p=d(e,"P",{}),x(p).forEach(r),c=a(e),T=d(e,"P",{"data-svelte-h":!0}),h(T)!=="svelte-1ku4t55"&&(T.innerHTML=t),M=a(e),g(Ve.$$.fragment,e),Dn=a(e),Te=d(e,"DIV",{class:!0,"data-svelte-h":!0}),h(Te)!=="svelte-13t8s2t"&&(Te.innerHTML=ma),Kn=a(e),g(Qe.$$.fragment,e),eo=a(e),Ae=d(e,"P",{"data-svelte-h":!0}),h(Ae)!=="svelte-121m0tn"&&(Ae.innerHTML=ha),to=a(e),Re=d(e,"P",{"data-svelte-h":!0}),h(Re)!=="svelte-vfdo9a"&&(Re.textContent=ua),no=a(e),Oe=d(e,"P",{"data-svelte-h":!0}),h(Oe)!=="svelte-ebukil"&&(Oe.innerHTML=ga),oo=a(e),Ee=d(e,"P",{"data-svelte-h":!0}),h(Ee)!=="svelte-1j7svg"&&(Ee.innerHTML=fa),so=a(e),g(Ye.$$.fragment,e),ao=a(e),Pe=d(e,"UL",{"data-svelte-h":!0}),h(Pe)!=="svelte-1gubzyh"&&(Pe.innerHTML=_a),ro=a(e),g(De.$$.fragment,e),io=a(e),Ke=d(e,"UL",{"data-svelte-h":!0}),h(Ke)!=="svelte-qxfc7p"&&(Ke.innerHTML=ba),lo=a(e),g(et.$$.fragment,e),co=a(e),B=d(e,"DIV",{class:!0});var Y=x(B);g(tt.$$.fragment,Y),Bo=a(Y),Pt=d(Y,"P",{"data-svelte-h":!0}),h(Pt)!=="svelte-q2trno"&&(Pt.innerHTML=ka),Go=a(Y),Dt=d(Y,"P",{"data-svelte-h":!0}),h(Dt)!=="svelte-1ek1ss9"&&(Dt.innerHTML=Ta),Ho=a(Y),g(ye.$$.fragment,Y),Y.forEach(r),po=a(e),g(nt.$$.fragment,e),mo=a(e),L=d(e,"DIV",{class:!0});var X=x(L);g(ot.$$.fragment,X),So=a(X),Kt=d(X,"P",{"data-svelte-h":!0}),h(Kt)!=="svelte-1dqc3xp"&&(Kt.innerHTML=ya),Vo=a(X),en=d(X,"P",{"data-svelte-h":!0}),h(en)!=="svelte-ntrhio"&&(en.innerHTML=Ma),Qo=a(X),K=d(X,"DIV",{class:!0});var be=x(K);g(st.$$.fragment,be),Ao=a(be),tn=d(be,"P",{"data-svelte-h":!0}),h(tn)!=="svelte-1dgk30w"&&(tn.textContent=wa),Ro=a(be),nn=d(be,"UL",{"data-svelte-h":!0}),h(nn)!=="svelte-zi1mnq"&&(nn.innerHTML=va),be.forEach(r),Oo=a(X),Me=d(X,"DIV",{class:!0});var Ht=x(Me);g(at.$$.fragment,Ht),Eo=a(Ht),on=d(Ht,"P",{"data-svelte-h":!0}),h(on)!=="svelte-1f4f5kp"&&(on.innerHTML=xa),Ht.forEach(r),Yo=a(X),S=d(X,"DIV",{class:!0});var P=x(S);g(rt.$$.fragment,P),Po=a(P),sn=d(P,"P",{"data-svelte-h":!0}),h(sn)!=="svelte-1nwvqaq"&&(sn.textContent=Na),Do=a(P),g(we.$$.fragment,P),Ko=a(P),an=d(P,"P",{"data-svelte-h":!0}),h(an)!=="svelte-owoxgn"&&(an.innerHTML=La),P.forEach(r),es=a(X),rn=d(X,"DIV",{class:!0});var Pn=x(rn);g(it.$$.fragment,Pn),Pn.forEach(r),X.forEach(r),ho=a(e),g(lt.$$.fragment,e),uo=a(e),$=d(e,"DIV",{class:!0});var G=x($);g(dt.$$.fragment,G),ts=a(G),ln=d(G,"P",{"data-svelte-h":!0}),h(ln)!=="svelte-1jtbcee"&&(ln.innerHTML=Xa),ns=a(G),dn=d(G,"P",{"data-svelte-h":!0}),h(dn)!=="svelte-gxzj9w"&&(dn.innerHTML=ja),os=a(G),ee=d(G,"DIV",{class:!0});var ke=x(ee);g(ct.$$.fragment,ke),ss=a(ke),cn=d(ke,"P",{"data-svelte-h":!0}),h(cn)!=="svelte-1dgk30w"&&(cn.textContent=$a),as=a(ke),pn=d(ke,"UL",{"data-svelte-h":!0}),h(pn)!=="svelte-zi1mnq"&&(pn.innerHTML=Fa),ke.forEach(r),rs=a(G),V=d(G,"DIV",{class:!0});var D=x(V);g(pt.$$.fragment,D),is=a(D),mn=d(D,"P",{"data-svelte-h":!0}),h(mn)!=="svelte-1nwvqaq"&&(mn.textContent=Ca),ls=a(D),g(ve.$$.fragment,D),ds=a(D),hn=d(D,"P",{"data-svelte-h":!0}),h(hn)!=="svelte-owoxgn"&&(hn.innerHTML=Ja),D.forEach(r),G.forEach(r),go=a(e),g(mt.$$.fragment,e),fo=a(e),pe=d(e,"DIV",{class:!0});var St=x(pe);g(ht.$$.fragment,St),cs=a(St),un=d(St,"P",{"data-svelte-h":!0}),h(un)!=="svelte-19wdepc"&&(un.innerHTML=za),St.forEach(r),_o=a(e),me=d(e,"DIV",{class:!0});var Vt=x(me);g(ut.$$.fragment,Vt),ps=a(Vt),gn=d(Vt,"P",{"data-svelte-h":!0}),h(gn)!=="svelte-dwuybi"&&(gn.innerHTML=Ua),Vt.forEach(r),bo=a(e),he=d(e,"DIV",{class:!0});var Qt=x(he);g(gt.$$.fragment,Qt),ms=a(Qt),fn=d(Qt,"P",{"data-svelte-h":!0}),h(fn)!=="svelte-1f4a5te"&&(fn.innerHTML=qa),Qt.forEach(r),ko=a(e),ue=d(e,"DIV",{class:!0});var At=x(ue);g(ft.$$.fragment,At),hs=a(At),_n=d(At,"P",{"data-svelte-h":!0}),h(_n)!=="svelte-167wgbe"&&(_n.innerHTML=Ia),At.forEach(r),To=a(e),ge=d(e,"DIV",{class:!0});var Rt=x(ge);g(_t.$$.fragment,Rt),us=a(Rt),bn=d(Rt,"P",{"data-svelte-h":!0}),h(bn)!=="svelte-3njxff"&&(bn.innerHTML=Wa),Rt.forEach(r),yo=a(e),fe=d(e,"DIV",{class:!0});var Ot=x(fe);g(bt.$$.fragment,Ot),gs=a(Ot),kn=d(Ot,"P",{"data-svelte-h":!0}),h(kn)!=="svelte-rpz99w"&&(kn.innerHTML=Za),Ot.forEach(r),Mo=a(e),_e=d(e,"DIV",{class:!0});var Et=x(_e);g(kt.$$.fragment,Et),fs=a(Et),Tn=d(Et,"P",{"data-svelte-h":!0}),h(Tn)!=="svelte-9c0708"&&(Tn.innerHTML=Ba),Et.forEach(r),wo=a(e),g(Tt.$$.fragment,e),vo=a(e),F=d(e,"DIV",{class:!0});var H=x(F);g(yt.$$.fragment,H),_s=a(H),yn=d(H,"P",{"data-svelte-h":!0}),h(yn)!=="svelte-9rukcz"&&(yn.textContent=Ga),bs=a(H),Mn=d(H,"P",{"data-svelte-h":!0}),h(Mn)!=="svelte-q52n56"&&(Mn.innerHTML=Ha),ks=a(H),wn=d(H,"P",{"data-svelte-h":!0}),h(wn)!=="svelte-hswkmf"&&(wn.innerHTML=Sa),Ts=a(H),te=d(H,"DIV",{class:!0});var En=x(te);g(Mt.$$.fragment,En),ys=a(En),vn=d(En,"P",{"data-svelte-h":!0}),h(vn)!=="svelte-txdsuy"&&(vn.innerHTML=Va),Ms=a(En),g(xe.$$.fragment,En),En.forEach(r),H.forEach(r),xo=a(e),g(wt.$$.fragment,e),No=a(e),C=d(e,"DIV",{class:!0});var ne=x(C);g(vt.$$.fragment,ne),ws=a(ne),xn=d(ne,"P",{"data-svelte-h":!0}),h(xn)!=="svelte-s260t6"&&(xn.textContent=Qa),vs=a(ne),Nn=d(ne,"P",{"data-svelte-h":!0}),h(Nn)!=="svelte-q52n56"&&(Nn.innerHTML=Aa),xs=a(ne),Ln=d(ne,"P",{"data-svelte-h":!0}),h(Ln)!=="svelte-hswkmf"&&(Ln.innerHTML=Ra),Ns=a(ne),Q=d(ne,"DIV",{class:!0});var Ze=x(Q);g(xt.$$.fragment,Ze),Ls=a(Ze),Xn=d(Ze,"P",{"data-svelte-h":!0}),h(Xn)!=="svelte-1qwm8j0"&&(Xn.innerHTML=Oa),Xs=a(Ze),g(Ne.$$.fragment,Ze),js=a(Ze),g(Le.$$.fragment,Ze),Ze.forEach(r),ne.forEach(r),Lo=a(e),g(Nt.$$.fragment,e),Xo=a(e),J=d(e,"DIV",{class:!0});var oe=x(J);g(Lt.$$.fragment,oe),$s=a(oe),jn=d(oe,"P",{"data-svelte-h":!0}),h(jn)!=="svelte-1ccr0de"&&(jn.textContent=Ea),Fs=a(oe),$n=d(oe,"P",{"data-svelte-h":!0}),h($n)!=="svelte-q52n56"&&($n.innerHTML=Ya),Cs=a(oe),Fn=d(oe,"P",{"data-svelte-h":!0}),h(Fn)!=="svelte-hswkmf"&&(Fn.innerHTML=Pa),Js=a(oe),Z=d(oe,"DIV",{class:!0});var se=x(Z);g(Xt.$$.fragment,se),zs=a(se),Cn=d(se,"P",{"data-svelte-h":!0}),h(Cn)!=="svelte-p7ryew"&&(Cn.innerHTML=Da),Us=a(se),g(Xe.$$.fragment,se),qs=a(se),g(je.$$.fragment,se),Is=a(se),g($e.$$.fragment,se),se.forEach(r),oe.forEach(r),jo=a(e),g(jt.$$.fragment,e),$o=a(e),z=d(e,"DIV",{class:!0});var ae=x(z);g($t.$$.fragment,ae),Ws=a(ae),Jn=d(ae,"P",{"data-svelte-h":!0}),h(Jn)!=="svelte-i3uliy"&&(Jn.textContent=Ka),Zs=a(ae),zn=d(ae,"P",{"data-svelte-h":!0}),h(zn)!=="svelte-q52n56"&&(zn.innerHTML=er),Bs=a(ae),Un=d(ae,"P",{"data-svelte-h":!0}),h(Un)!=="svelte-hswkmf"&&(Un.innerHTML=tr),Gs=a(ae),A=d(ae,"DIV",{class:!0});var Be=x(A);g(Ft.$$.fragment,Be),Hs=a(Be),qn=d(Be,"P",{"data-svelte-h":!0}),h(qn)!=="svelte-1omckf0"&&(qn.innerHTML=nr),Ss=a(Be),g(Fe.$$.fragment,Be),Vs=a(Be),g(Ce.$$.fragment,Be),Be.forEach(r),ae.forEach(r),Fo=a(e),g(Ct.$$.fragment,e),Co=a(e),U=d(e,"DIV",{class:!0});var re=x(U);g(Jt.$$.fragment,re),Qs=a(re),In=d(re,"P",{"data-svelte-h":!0}),h(In)!=="svelte-1eppb4t"&&(In.textContent=or),As=a(re),Wn=d(re,"P",{"data-svelte-h":!0}),h(Wn)!=="svelte-q52n56"&&(Wn.innerHTML=sr),Rs=a(re),Zn=d(re,"P",{"data-svelte-h":!0}),h(Zn)!=="svelte-hswkmf"&&(Zn.innerHTML=ar),Os=a(re),R=d(re,"DIV",{class:!0});var Ge=x(R);g(zt.$$.fragment,Ge),Es=a(Ge),Bn=d(Ge,"P",{"data-svelte-h":!0}),h(Bn)!=="svelte-8c8uyy"&&(Bn.innerHTML=rr),Ys=a(Ge),g(Je.$$.fragment,Ge),Ps=a(Ge),g(ze.$$.fragment,Ge),Ge.forEach(r),re.forEach(r),Jo=a(e),g(Ut.$$.fragment,e),zo=a(e),q=d(e,"DIV",{class:!0});var ie=x(q);g(qt.$$.fragment,ie),Ds=a(ie),Gn=d(ie,"P",{"data-svelte-h":!0}),h(Gn)!=="svelte-1gmn8ay"&&(Gn.innerHTML=ir),Ks=a(ie),Hn=d(ie,"P",{"data-svelte-h":!0}),h(Hn)!=="svelte-q52n56"&&(Hn.innerHTML=lr),ea=a(ie),Sn=d(ie,"P",{"data-svelte-h":!0}),h(Sn)!=="svelte-hswkmf"&&(Sn.innerHTML=dr),ta=a(ie),O=d(ie,"DIV",{class:!0});var He=x(O);g(It.$$.fragment,He),na=a(He),Vn=d(He,"P",{"data-svelte-h":!0}),h(Vn)!=="svelte-qzpgeq"&&(Vn.innerHTML=cr),oa=a(He),g(Ue.$$.fragment,He),sa=a(He),g(qe.$$.fragment,He),He.forEach(r),ie.forEach(r),Uo=a(e),g(Wt.$$.fragment,e),qo=a(e),I=d(e,"DIV",{class:!0});var le=x(I);g(Zt.$$.fragment,le),aa=a(le),Qn=d(le,"P",{"data-svelte-h":!0}),h(Qn)!=="svelte-19bgfi6"&&(Qn.innerHTML=pr),ra=a(le),An=d(le,"P",{"data-svelte-h":!0}),h(An)!=="svelte-q52n56"&&(An.innerHTML=mr),ia=a(le),Rn=d(le,"P",{"data-svelte-h":!0}),h(Rn)!=="svelte-hswkmf"&&(Rn.innerHTML=hr),la=a(le),E=d(le,"DIV",{class:!0});var Se=x(E);g(Bt.$$.fragment,Se),da=a(Se),On=d(Se,"P",{"data-svelte-h":!0}),h(On)!=="svelte-n4c4su"&&(On.innerHTML=ur),ca=a(Se),g(Ie.$$.fragment,Se),pa=a(Se),g(We.$$.fragment,Se),Se.forEach(r),le.forEach(r),Io=a(e),g(Gt.$$.fragment,e),Wo=a(e),Yn=d(e,"P",{}),x(Yn).forEach(r),this.h()},h(){v(n,"name","hf:doc:metadata"),v(n,"content",Zr),v(Te,"class","flex flex-wrap space-x-1"),v(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(rn,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(_e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,i){o(document.head,n),m(e,y,i),m(e,p,i),m(e,c,i),m(e,T,i),m(e,M,i),f(Ve,e,i),m(e,Dn,i),m(e,Te,i),m(e,Kn,i),f(Qe,e,i),m(e,eo,i),m(e,Ae,i),m(e,to,i),m(e,Re,i),m(e,no,i),m(e,Oe,i),m(e,oo,i),m(e,Ee,i),m(e,so,i),f(Ye,e,i),m(e,ao,i),m(e,Pe,i),m(e,ro,i),f(De,e,i),m(e,io,i),m(e,Ke,i),m(e,lo,i),f(et,e,i),m(e,co,i),m(e,B,i),f(tt,B,null),o(B,Bo),o(B,Pt),o(B,Go),o(B,Dt),o(B,Ho),f(ye,B,null),m(e,po,i),f(nt,e,i),m(e,mo,i),m(e,L,i),f(ot,L,null),o(L,So),o(L,Kt),o(L,Vo),o(L,en),o(L,Qo),o(L,K),f(st,K,null),o(K,Ao),o(K,tn),o(K,Ro),o(K,nn),o(L,Oo),o(L,Me),f(at,Me,null),o(Me,Eo),o(Me,on),o(L,Yo),o(L,S),f(rt,S,null),o(S,Po),o(S,sn),o(S,Do),f(we,S,null),o(S,Ko),o(S,an),o(L,es),o(L,rn),f(it,rn,null),m(e,ho,i),f(lt,e,i),m(e,uo,i),m(e,$,i),f(dt,$,null),o($,ts),o($,ln),o($,ns),o($,dn),o($,os),o($,ee),f(ct,ee,null),o(ee,ss),o(ee,cn),o(ee,as),o(ee,pn),o($,rs),o($,V),f(pt,V,null),o(V,is),o(V,mn),o(V,ls),f(ve,V,null),o(V,ds),o(V,hn),m(e,go,i),f(mt,e,i),m(e,fo,i),m(e,pe,i),f(ht,pe,null),o(pe,cs),o(pe,un),m(e,_o,i),m(e,me,i),f(ut,me,null),o(me,ps),o(me,gn),m(e,bo,i),m(e,he,i),f(gt,he,null),o(he,ms),o(he,fn),m(e,ko,i),m(e,ue,i),f(ft,ue,null),o(ue,hs),o(ue,_n),m(e,To,i),m(e,ge,i),f(_t,ge,null),o(ge,us),o(ge,bn),m(e,yo,i),m(e,fe,i),f(bt,fe,null),o(fe,gs),o(fe,kn),m(e,Mo,i),m(e,_e,i),f(kt,_e,null),o(_e,fs),o(_e,Tn),m(e,wo,i),f(Tt,e,i),m(e,vo,i),m(e,F,i),f(yt,F,null),o(F,_s),o(F,yn),o(F,bs),o(F,Mn),o(F,ks),o(F,wn),o(F,Ts),o(F,te),f(Mt,te,null),o(te,ys),o(te,vn),o(te,Ms),f(xe,te,null),m(e,xo,i),f(wt,e,i),m(e,No,i),m(e,C,i),f(vt,C,null),o(C,ws),o(C,xn),o(C,vs),o(C,Nn),o(C,xs),o(C,Ln),o(C,Ns),o(C,Q),f(xt,Q,null),o(Q,Ls),o(Q,Xn),o(Q,Xs),f(Ne,Q,null),o(Q,js),f(Le,Q,null),m(e,Lo,i),f(Nt,e,i),m(e,Xo,i),m(e,J,i),f(Lt,J,null),o(J,$s),o(J,jn),o(J,Fs),o(J,$n),o(J,Cs),o(J,Fn),o(J,Js),o(J,Z),f(Xt,Z,null),o(Z,zs),o(Z,Cn),o(Z,Us),f(Xe,Z,null),o(Z,qs),f(je,Z,null),o(Z,Is),f($e,Z,null),m(e,jo,i),f(jt,e,i),m(e,$o,i),m(e,z,i),f($t,z,null),o(z,Ws),o(z,Jn),o(z,Zs),o(z,zn),o(z,Bs),o(z,Un),o(z,Gs),o(z,A),f(Ft,A,null),o(A,Hs),o(A,qn),o(A,Ss),f(Fe,A,null),o(A,Vs),f(Ce,A,null),m(e,Fo,i),f(Ct,e,i),m(e,Co,i),m(e,U,i),f(Jt,U,null),o(U,Qs),o(U,In),o(U,As),o(U,Wn),o(U,Rs),o(U,Zn),o(U,Os),o(U,R),f(zt,R,null),o(R,Es),o(R,Bn),o(R,Ys),f(Je,R,null),o(R,Ps),f(ze,R,null),m(e,Jo,i),f(Ut,e,i),m(e,zo,i),m(e,q,i),f(qt,q,null),o(q,Ds),o(q,Gn),o(q,Ks),o(q,Hn),o(q,ea),o(q,Sn),o(q,ta),o(q,O),f(It,O,null),o(O,na),o(O,Vn),o(O,oa),f(Ue,O,null),o(O,sa),f(qe,O,null),m(e,Uo,i),f(Wt,e,i),m(e,qo,i),m(e,I,i),f(Zt,I,null),o(I,aa),o(I,Qn),o(I,ra),o(I,An),o(I,ia),o(I,Rn),o(I,la),o(I,E),f(Bt,E,null),o(E,da),o(E,On),o(E,ca),f(Ie,E,null),o(E,pa),f(We,E,null),m(e,Io,i),f(Gt,e,i),m(e,Wo,i),m(e,Yn,i),Zo=!0},p(e,[i]){const Y={};i&2&&(Y.$$scope={dirty:i,ctx:e}),ye.$set(Y);const X={};i&2&&(X.$$scope={dirty:i,ctx:e}),we.$set(X);const be={};i&2&&(be.$$scope={dirty:i,ctx:e}),ve.$set(be);const Ht={};i&2&&(Ht.$$scope={dirty:i,ctx:e}),xe.$set(Ht);const P={};i&2&&(P.$$scope={dirty:i,ctx:e}),Ne.$set(P);const Pn={};i&2&&(Pn.$$scope={dirty:i,ctx:e}),Le.$set(Pn);const G={};i&2&&(G.$$scope={dirty:i,ctx:e}),Xe.$set(G);const ke={};i&2&&(ke.$$scope={dirty:i,ctx:e}),je.$set(ke);const D={};i&2&&(D.$$scope={dirty:i,ctx:e}),$e.$set(D);const St={};i&2&&(St.$$scope={dirty:i,ctx:e}),Fe.$set(St);const Vt={};i&2&&(Vt.$$scope={dirty:i,ctx:e}),Ce.$set(Vt);const Qt={};i&2&&(Qt.$$scope={dirty:i,ctx:e}),Je.$set(Qt);const At={};i&2&&(At.$$scope={dirty:i,ctx:e}),ze.$set(At);const Rt={};i&2&&(Rt.$$scope={dirty:i,ctx:e}),Ue.$set(Rt);const Ot={};i&2&&(Ot.$$scope={dirty:i,ctx:e}),qe.$set(Ot);const Et={};i&2&&(Et.$$scope={dirty:i,ctx:e}),Ie.$set(Et);const H={};i&2&&(H.$$scope={dirty:i,ctx:e}),We.$set(H)},i(e){Zo||(_(Ve.$$.fragment,e),_(Qe.$$.fragment,e),_(Ye.$$.fragment,e),_(De.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(ye.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(we.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(ve.$$.fragment,e),_(mt.$$.fragment,e),_(ht.$$.fragment,e),_(ut.$$.fragment,e),_(gt.$$.fragment,e),_(ft.$$.fragment,e),_(_t.$$.fragment,e),_(bt.$$.fragment,e),_(kt.$$.fragment,e),_(Tt.$$.fragment,e),_(yt.$$.fragment,e),_(Mt.$$.fragment,e),_(xe.$$.fragment,e),_(wt.$$.fragment,e),_(vt.$$.fragment,e),_(xt.$$.fragment,e),_(Ne.$$.fragment,e),_(Le.$$.fragment,e),_(Nt.$$.fragment,e),_(Lt.$$.fragment,e),_(Xt.$$.fragment,e),_(Xe.$$.fragment,e),_(je.$$.fragment,e),_($e.$$.fragment,e),_(jt.$$.fragment,e),_($t.$$.fragment,e),_(Ft.$$.fragment,e),_(Fe.$$.fragment,e),_(Ce.$$.fragment,e),_(Ct.$$.fragment,e),_(Jt.$$.fragment,e),_(zt.$$.fragment,e),_(Je.$$.fragment,e),_(ze.$$.fragment,e),_(Ut.$$.fragment,e),_(qt.$$.fragment,e),_(It.$$.fragment,e),_(Ue.$$.fragment,e),_(qe.$$.fragment,e),_(Wt.$$.fragment,e),_(Zt.$$.fragment,e),_(Bt.$$.fragment,e),_(Ie.$$.fragment,e),_(We.$$.fragment,e),_(Gt.$$.fragment,e),Zo=!0)},o(e){b(Ve.$$.fragment,e),b(Qe.$$.fragment,e),b(Ye.$$.fragment,e),b(De.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(ye.$$.fragment,e),b(nt.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b(rt.$$.fragment,e),b(we.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(ve.$$.fragment,e),b(mt.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(gt.$$.fragment,e),b(ft.$$.fragment,e),b(_t.$$.fragment,e),b(bt.$$.fragment,e),b(kt.$$.fragment,e),b(Tt.$$.fragment,e),b(yt.$$.fragment,e),b(Mt.$$.fragment,e),b(xe.$$.fragment,e),b(wt.$$.fragment,e),b(vt.$$.fragment,e),b(xt.$$.fragment,e),b(Ne.$$.fragment,e),b(Le.$$.fragment,e),b(Nt.$$.fragment,e),b(Lt.$$.fragment,e),b(Xt.$$.fragment,e),b(Xe.$$.fragment,e),b(je.$$.fragment,e),b($e.$$.fragment,e),b(jt.$$.fragment,e),b($t.$$.fragment,e),b(Ft.$$.fragment,e),b(Fe.$$.fragment,e),b(Ce.$$.fragment,e),b(Ct.$$.fragment,e),b(Jt.$$.fragment,e),b(zt.$$.fragment,e),b(Je.$$.fragment,e),b(ze.$$.fragment,e),b(Ut.$$.fragment,e),b(qt.$$.fragment,e),b(It.$$.fragment,e),b(Ue.$$.fragment,e),b(qe.$$.fragment,e),b(Wt.$$.fragment,e),b(Zt.$$.fragment,e),b(Bt.$$.fragment,e),b(Ie.$$.fragment,e),b(We.$$.fragment,e),b(Gt.$$.fragment,e),Zo=!1},d(e){e&&(r(y),r(p),r(c),r(T),r(M),r(Dn),r(Te),r(Kn),r(eo),r(Ae),r(to),r(Re),r(no),r(Oe),r(oo),r(Ee),r(so),r(ao),r(Pe),r(ro),r(io),r(Ke),r(lo),r(co),r(B),r(po),r(mo),r(L),r(ho),r(uo),r($),r(go),r(fo),r(pe),r(_o),r(me),r(bo),r(he),r(ko),r(ue),r(To),r(ge),r(yo),r(fe),r(Mo),r(_e),r(wo),r(vo),r(F),r(xo),r(No),r(C),r(Lo),r(Xo),r(J),r(jo),r($o),r(z),r(Fo),r(Co),r(U),r(Jo),r(zo),r(q),r(Uo),r(qo),r(I),r(Io),r(Wo),r(Yn)),r(n),k(Ve,e),k(Qe,e),k(Ye,e),k(De,e),k(et,e),k(tt),k(ye),k(nt,e),k(ot),k(st),k(at),k(rt),k(we),k(it),k(lt,e),k(dt),k(ct),k(pt),k(ve),k(mt,e),k(ht),k(ut),k(gt),k(ft),k(_t),k(bt),k(kt),k(Tt,e),k(yt),k(Mt),k(xe),k(wt,e),k(vt),k(xt),k(Ne),k(Le),k(Nt,e),k(Lt),k(Xt),k(Xe),k(je),k($e),k(jt,e),k($t),k(Ft),k(Fe),k(Ce),k(Ct,e),k(Jt),k(zt),k(Je),k(ze),k(Ut,e),k(qt),k(It),k(Ue),k(qe),k(Wt,e),k(Zt),k(Bt),k(Ie),k(We),k(Gt,e)}}}const Zr='{"title":"XLNet","local":"xlnet","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"XLNetConfig","local":"transformers.XLNetConfig","sections":[],"depth":2},{"title":"XLNetTokenizer","local":"transformers.XLNetTokenizer","sections":[],"depth":2},{"title":"XLNetTokenizerFast","local":"transformers.XLNetTokenizerFast","sections":[],"depth":2},{"title":"XLNet specific outputs","local":"transformers.models.xlnet.modeling_xlnet.XLNetModelOutput","sections":[],"depth":2},{"title":"XLNetModel","local":"transformers.XLNetModel","sections":[],"depth":2},{"title":"XLNetLMHeadModel","local":"transformers.XLNetLMHeadModel","sections":[],"depth":2},{"title":"XLNetForSequenceClassification","local":"transformers.XLNetForSequenceClassification","sections":[],"depth":2},{"title":"XLNetForMultipleChoice","local":"transformers.XLNetForMultipleChoice","sections":[],"depth":2},{"title":"XLNetForTokenClassification","local":"transformers.XLNetForTokenClassification","sections":[],"depth":2},{"title":"XLNetForQuestionAnsweringSimple","local":"transformers.XLNetForQuestionAnsweringSimple","sections":[],"depth":2},{"title":"XLNetForQuestionAnswering","local":"transformers.XLNetForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Br(w){return fr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Or extends _r{constructor(n){super(),br(this,n,Br,Wr,gr,{})}}export{Or as component};
