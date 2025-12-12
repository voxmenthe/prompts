import{s as Ca,o as $a,n as J}from"../chunks/scheduler.18a86fab.js";import{S as Fa,i as Wa,g as c,s,r as h,A as Ia,h as p,f as l,c as a,j,x as u,u as f,k as v,y as o,a as m,v as g,d as _,t as y,w as k}from"../chunks/index.98837b22.js";import{T as _e}from"../chunks/Tip.77304350.js";import{D as L}from"../chunks/Docstring.a1ef7999.js";import{C as D}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ie}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as R,E as Za}from"../chunks/getInferenceSnippets.06c2775f.js";function Ba(w){let t,b="Examples:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEx1a2VDb25maWclMkMlMjBMdWtlTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTFVLRSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwTHVrZUNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEx1a2VNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LukeConfig, LukeModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a LUKE configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = LukeConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-kvfsh7"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function qa(w){let t,b="be encoded differently whether it is at the beginning of the sentence (without space) or not:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEx1a2VUb2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBMdWtlVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzdHVkaW8tb3VzaWElMkZsdWtlLWJhc2UlMjIpJTBBdG9rZW5pemVyKCUyMkhlbGxvJTIwd29ybGQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTBBdG9rZW5pemVyKCUyMiUyMEhlbGxvJTIwd29ybGQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LukeTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LukeTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">31414</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">20920</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-12atnao"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function Ra(w){let t,b="When used with <code>is_split_into_words=True</code>, this tokenizer will add a space before each word (even the first one).";return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-jhmxzm"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function Ea(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function Na(w){let t,b="Examples:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMdWtlTW9kZWwlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzdHVkaW8tb3VzaWElMkZsdWtlLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBMdWtlTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiklMEElMEF0ZXh0JTIwJTNEJTIwJTIyQmV5b25jJUMzJUE5JTIwbGl2ZXMlMjBpbiUyMExvcyUyMEFuZ2VsZXMuJTIyJTBBZW50aXR5X3NwYW5zJTIwJTNEJTIwJTVCKDAlMkMlMjA3KSU1RCUyMCUyMCUyMyUyMGNoYXJhY3Rlci1iYXNlZCUyMGVudGl0eSUyMHNwYW4lMjBjb3JyZXNwb25kaW5nJTIwdG8lMjAlMjJCZXlvbmMlQzMlQTklMjIlMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplcih0ZXh0JTJDJTIwZW50aXR5X3NwYW5zJTNEZW50aXR5X3NwYW5zJTJDJTIwYWRkX3ByZWZpeF9zcGFjZSUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmVuY29kaW5nKSUwQXdvcmRfbGFzdF9oaWRkZW5fc3RhdGUlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBZW50aXR5X2xhc3RfaGlkZGVuX3N0YXRlJTIwJTNEJTIwb3V0cHV0cy5lbnRpdHlfbGFzdF9oaWRkZW5fc3RhdGUlMEElMEF0ZXh0JTIwJTNEJTIwJTIyQmV5b25jJUMzJUE5JTIwbGl2ZXMlMjBpbiUyMExvcyUyMEFuZ2VsZXMuJTIyJTBBZW50aXRpZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlMjJCZXlvbmMlQzMlQTklMjIlMkMlMEElMjAlMjAlMjAlMjAlMjJMb3MlMjBBbmdlbGVzJTIyJTJDJTBBJTVEJTIwJTIwJTIzJTIwV2lraXBlZGlhJTIwZW50aXR5JTIwdGl0bGVzJTIwY29ycmVzcG9uZGluZyUyMHRvJTIwdGhlJTIwZW50aXR5JTIwbWVudGlvbnMlMjAlMjJCZXlvbmMlQzMlQTklMjIlMjBhbmQlMjAlMjJMb3MlMjBBbmdlbGVzJTIyJTBBZW50aXR5X3NwYW5zJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwKDAlMkMlMjA3KSUyQyUwQSUyMCUyMCUyMCUyMCgxNyUyQyUyMDI4KSUyQyUwQSU1RCUyMCUyMCUyMyUyMGNoYXJhY3Rlci1iYXNlZCUyMGVudGl0eSUyMHNwYW5zJTIwY29ycmVzcG9uZGluZyUyMHRvJTIwJTIyQmV5b25jJUMzJUE5JTIyJTIwYW5kJTIwJTIyTG9zJTIwQW5nZWxlcyUyMiUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMHRleHQlMkMlMjBlbnRpdGllcyUzRGVudGl0aWVzJTJDJTIwZW50aXR5X3NwYW5zJTNEZW50aXR5X3NwYW5zJTJDJTIwYWRkX3ByZWZpeF9zcGFjZSUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmVuY29kaW5nKSUwQXdvcmRfbGFzdF9oaWRkZW5fc3RhdGUlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBZW50aXR5X2xhc3RfaGlkZGVuX3N0YXRlJTIwJTNEJTIwb3V0cHV0cy5lbnRpdHlfbGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeModel.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-comment"># Compute the contextualized entity representation corresponding to the entity mention &quot;Beyoncé&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;Beyoncé lives in Los Angeles.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_spans = [(<span class="hljs-number">0</span>, <span class="hljs-number">7</span>)]  <span class="hljs-comment"># character-based entity span corresponding to &quot;Beyoncé&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>word_last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_last_hidden_state = outputs.entity_last_hidden_state
<span class="hljs-comment"># Input Wikipedia entities to obtain enriched contextualized representations of word tokens</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;Beyoncé lives in Los Angeles.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>entities = [
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Beyoncé&quot;</span>,
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Los Angeles&quot;</span>,
<span class="hljs-meta">... </span>]  <span class="hljs-comment"># Wikipedia entity titles corresponding to the entity mentions &quot;Beyoncé&quot; and &quot;Los Angeles&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_spans = [
<span class="hljs-meta">... </span>    (<span class="hljs-number">0</span>, <span class="hljs-number">7</span>),
<span class="hljs-meta">... </span>    (<span class="hljs-number">17</span>, <span class="hljs-number">28</span>),
<span class="hljs-meta">... </span>]  <span class="hljs-comment"># character-based entity spans corresponding to &quot;Beyoncé&quot; and &quot;Los Angeles&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(
<span class="hljs-meta">... </span>    text, entities=entities, entity_spans=entity_spans, add_prefix_space=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>word_last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_last_hidden_state = outputs.entity_last_hidden_state`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-kvfsh7"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function Va(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function Ga(w){let t,b="Example:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMdWtlRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMEx1a2VGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCUzQ21hc2slM0UuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCUzQ21hc2slM0UlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTIzJTIwbWFzayUyMGxhYmVscyUyMG9mJTIwbm9uLSUzQ21hc2slM0UlMjB0b2tlbnMlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQlMkMlMjBsYWJlbHMlMkMlMjAtMTAwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFyb3VuZChvdXRwdXRzLmxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForMaskedLM.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function Xa(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function Sa(w){let t,b="Examples:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMdWtlRm9yRW50aXR5Q2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzdHVkaW8tb3VzaWElMkZsdWtlLWxhcmdlLWZpbmV0dW5lZC1vcGVuLWVudGl0eSUyMiklMEFtb2RlbCUyMCUzRCUyMEx1a2VGb3JFbnRpdHlDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1sYXJnZS1maW5ldHVuZWQtb3Blbi1lbnRpdHklMjIpJTBBJTBBdGV4dCUyMCUzRCUyMCUyMkJleW9uYyVDMyVBOSUyMGxpdmVzJTIwaW4lMjBMb3MlMjBBbmdlbGVzLiUyMiUwQWVudGl0eV9zcGFucyUyMCUzRCUyMCU1QigwJTJDJTIwNyklNUQlMjAlMjAlMjMlMjBjaGFyYWN0ZXItYmFzZWQlMjBlbnRpdHklMjBzcGFuJTIwY29ycmVzcG9uZGluZyUyMHRvJTIwJTIyQmV5b25jJUMzJUE5JTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHRleHQlMkMlMjBlbnRpdHlfc3BhbnMlM0RlbnRpdHlfc3BhbnMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQXByZWRpY3RlZF9jbGFzc19pZHglMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKS5pdGVtKCklMEFwcmludCglMjJQcmVkaWN0ZWQlMjBjbGFzcyUzQSUyMiUyQyUyMG1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZHglNUQp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForEntityClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-large-finetuned-open-entity&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForEntityClassification.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-large-finetuned-open-entity&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;Beyoncé lives in Los Angeles.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_spans = [(<span class="hljs-number">0</span>, <span class="hljs-number">7</span>)]  <span class="hljs-comment"># character-based entity span corresponding to &quot;Beyoncé&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, entity_spans=entity_spans, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_idx = logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Predicted class:&quot;</span>, model.config.id2label[predicted_class_idx])
Predicted <span class="hljs-keyword">class</span>: person`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-kvfsh7"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function Ha(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function Qa(w){let t,b="Examples:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMdWtlRm9yRW50aXR5UGFpckNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1sYXJnZS1maW5ldHVuZWQtdGFjcmVkJTIyKSUwQW1vZGVsJTIwJTNEJTIwTHVrZUZvckVudGl0eVBhaXJDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1sYXJnZS1maW5ldHVuZWQtdGFjcmVkJTIyKSUwQSUwQXRleHQlMjAlM0QlMjAlMjJCZXlvbmMlQzMlQTklMjBsaXZlcyUyMGluJTIwTG9zJTIwQW5nZWxlcy4lMjIlMEFlbnRpdHlfc3BhbnMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAoMCUyQyUyMDcpJTJDJTBBJTIwJTIwJTIwJTIwKDE3JTJDJTIwMjgpJTJDJTBBJTVEJTIwJTIwJTIzJTIwY2hhcmFjdGVyLWJhc2VkJTIwZW50aXR5JTIwc3BhbnMlMjBjb3JyZXNwb25kaW5nJTIwdG8lMjAlMjJCZXlvbmMlQzMlQTklMjIlMjBhbmQlMjAlMjJMb3MlMjBBbmdlbGVzJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHRleHQlMkMlMjBlbnRpdHlfc3BhbnMlM0RlbnRpdHlfc3BhbnMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQXByZWRpY3RlZF9jbGFzc19pZHglMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKS5pdGVtKCklMEFwcmludCglMjJQcmVkaWN0ZWQlMjBjbGFzcyUzQSUyMiUyQyUyMG1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZHglNUQp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForEntityPairClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-large-finetuned-tacred&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForEntityPairClassification.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-large-finetuned-tacred&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;Beyoncé lives in Los Angeles.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_spans = [
<span class="hljs-meta">... </span>    (<span class="hljs-number">0</span>, <span class="hljs-number">7</span>),
<span class="hljs-meta">... </span>    (<span class="hljs-number">17</span>, <span class="hljs-number">28</span>),
<span class="hljs-meta">... </span>]  <span class="hljs-comment"># character-based entity spans corresponding to &quot;Beyoncé&quot; and &quot;Los Angeles&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, entity_spans=entity_spans, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_idx = logits.argmax(-<span class="hljs-number">1</span>).item()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Predicted class:&quot;</span>, model.config.id2label[predicted_class_idx])
Predicted <span class="hljs-keyword">class</span>: per:cities_of_residence`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-kvfsh7"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function Aa(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function Pa(w){let t,b="Examples:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMdWtlRm9yRW50aXR5U3BhbkNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1sYXJnZS1maW5ldHVuZWQtY29ubGwtMjAwMyUyMiklMEFtb2RlbCUyMCUzRCUyMEx1a2VGb3JFbnRpdHlTcGFuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtbGFyZ2UtZmluZXR1bmVkLWNvbmxsLTIwMDMlMjIpJTBBJTBBdGV4dCUyMCUzRCUyMCUyMkJleW9uYyVDMyVBOSUyMGxpdmVzJTIwaW4lMjBMb3MlMjBBbmdlbGVzJTIyJTBBJTBBd29yZF9zdGFydF9wb3NpdGlvbnMlMjAlM0QlMjAlNUIwJTJDJTIwOCUyQyUyMDE0JTJDJTIwMTclMkMlMjAyMSU1RCUyMCUyMCUyMyUyMGNoYXJhY3Rlci1iYXNlZCUyMHN0YXJ0JTIwcG9zaXRpb25zJTIwb2YlMjB3b3JkJTIwdG9rZW5zJTBBd29yZF9lbmRfcG9zaXRpb25zJTIwJTNEJTIwJTVCNyUyQyUyMDEzJTJDJTIwMTYlMkMlMjAyMCUyQyUyMDI4JTVEJTIwJTIwJTIzJTIwY2hhcmFjdGVyLWJhc2VkJTIwZW5kJTIwcG9zaXRpb25zJTIwb2YlMjB3b3JkJTIwdG9rZW5zJTBBZW50aXR5X3NwYW5zJTIwJTNEJTIwJTVCJTVEJTBBZm9yJTIwaSUyQyUyMHN0YXJ0X3BvcyUyMGluJTIwZW51bWVyYXRlKHdvcmRfc3RhcnRfcG9zaXRpb25zKSUzQSUwQSUyMCUyMCUyMCUyMGZvciUyMGVuZF9wb3MlMjBpbiUyMHdvcmRfZW5kX3Bvc2l0aW9ucyU1QmklM0ElNUQlM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBlbnRpdHlfc3BhbnMuYXBwZW5kKChzdGFydF9wb3MlMkMlMjBlbmRfcG9zKSklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIodGV4dCUyQyUyMGVudGl0eV9zcGFucyUzRGVudGl0eV9zcGFucyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBcHJlZGljdGVkX2NsYXNzX2luZGljZXMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKS5zcXVlZXplKCkudG9saXN0KCklMEFmb3IlMjBzcGFuJTJDJTIwcHJlZGljdGVkX2NsYXNzX2lkeCUyMGluJTIwemlwKGVudGl0eV9zcGFucyUyQyUyMHByZWRpY3RlZF9jbGFzc19pbmRpY2VzKSUzQSUwQSUyMCUyMCUyMCUyMGlmJTIwcHJlZGljdGVkX2NsYXNzX2lkeCUyMCElM0QlMjAwJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcHJpbnQodGV4dCU1QnNwYW4lNUIwJTVEJTIwJTNBJTIwc3BhbiU1QjElNUQlNUQlMkMlMjBtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWR4JTVEKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForEntitySpanClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-large-finetuned-conll-2003&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForEntitySpanClassification.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-large-finetuned-conll-2003&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;Beyoncé lives in Los Angeles&quot;</span>
<span class="hljs-comment"># List all possible entity spans in the text</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>word_start_positions = [<span class="hljs-number">0</span>, <span class="hljs-number">8</span>, <span class="hljs-number">14</span>, <span class="hljs-number">17</span>, <span class="hljs-number">21</span>]  <span class="hljs-comment"># character-based start positions of word tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>word_end_positions = [<span class="hljs-number">7</span>, <span class="hljs-number">13</span>, <span class="hljs-number">16</span>, <span class="hljs-number">20</span>, <span class="hljs-number">28</span>]  <span class="hljs-comment"># character-based end positions of word tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_spans = []
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> i, start_pos <span class="hljs-keyword">in</span> <span class="hljs-built_in">enumerate</span>(word_start_positions):
<span class="hljs-meta">... </span>    <span class="hljs-keyword">for</span> end_pos <span class="hljs-keyword">in</span> word_end_positions[i:]:
<span class="hljs-meta">... </span>        entity_spans.append((start_pos, end_pos))

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, entity_spans=entity_spans, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_indices = logits.argmax(-<span class="hljs-number">1</span>).squeeze().tolist()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> span, predicted_class_idx <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(entity_spans, predicted_class_indices):
<span class="hljs-meta">... </span>    <span class="hljs-keyword">if</span> predicted_class_idx != <span class="hljs-number">0</span>:
<span class="hljs-meta">... </span>        <span class="hljs-built_in">print</span>(text[span[<span class="hljs-number">0</span>] : span[<span class="hljs-number">1</span>]], model.config.id2label[predicted_class_idx])
Beyoncé PER
Los Angeles LOC`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-kvfsh7"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function Ya(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function Oa(w){let t,b="Example of single-label classification:",i,d,T;return d=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEx1a2VGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTHVrZUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwTHVrZUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-ykxpe4"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function Da(w){let t,b="Example of multi-label classification:",i,d,T;return d=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEx1a2VGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTHVrZUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwTHVrZUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-1l8e32d"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function Ka(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function er(w){let t,b="Example:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMdWtlRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMEx1a2VGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1iYXNlJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFjaG9pY2UwJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2l0aCUyMGElMjBmb3JrJTIwYW5kJTIwYSUyMGtuaWZlLiUyMiUwQWNob2ljZTElMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aGlsZSUyMGhlbGQlMjBpbiUyMHRoZSUyMGhhbmQuJTIyJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDApLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMGNob2ljZTAlMjBpcyUyMGNvcnJlY3QlMjAoYWNjb3JkaW5nJTIwdG8lMjBXaWtpcGVkaWElMjAlM0IpKSUyQyUyMGJhdGNoJTIwc2l6ZSUyMDElMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlMkMlMjBwcm9tcHQlNUQlMkMlMjAlNUJjaG9pY2UwJTJDJTIwY2hvaWNlMSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiolN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUyQyUyMGxhYmVscyUzRGxhYmVscyklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBpcyUyMDElMEElMEElMjMlMjB0aGUlMjBsaW5lYXIlMjBjbGFzc2lmaWVyJTIwc3RpbGwlMjBuZWVkcyUyMHRvJTIwYmUlMjB0cmFpbmVkJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function tr(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function nr(w){let t,b="Example:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMdWtlRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTHVrZUZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForTokenClassification.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function or(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=p(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,d){m(i,t,d)},p:J,d(i){i&&l(t)}}}function sr(w){let t,b="Example:",i,d,T;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMdWtlRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMEx1a2VGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1iYXNlJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LukeForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=s(),h(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=b),i=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,i,M),g(d,n,M),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){y(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(i)),k(d,n)}}}function ar(w){let t,b,i,d,T,n="<em>This model was released on 2020-10-02 and added to Hugging Face Transformers on 2021-05-03.</em>",M,Ve,Jn,ye,Ws='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Un,Ge,xn,Xe,Is=`The LUKE model was proposed in <a href="https://huggingface.co/papers/2010.01057" rel="nofollow">LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention</a> by Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda and Yuji Matsumoto.
It is based on RoBERTa and adds entity embeddings as well as an entity-aware self-attention mechanism, which helps
improve performance on various downstream tasks involving reasoning about entities such as named entity recognition,
extractive and cloze-style question answering, entity typing, and relation classification.`,Cn,Se,Zs="The abstract from the paper is the following:",$n,He,Bs=`<em>Entity representations are useful in natural language tasks involving entities. In this paper, we propose new
pretrained contextualized representations of words and entities based on the bidirectional transformer. The proposed
model treats words and entities in a given text as independent tokens, and outputs contextualized representations of
them. Our model is trained using a new pretraining task based on the masked language model of BERT. The task involves
predicting randomly masked words and entities in a large entity-annotated corpus retrieved from Wikipedia. We also
propose an entity-aware self-attention mechanism that is an extension of the self-attention mechanism of the
transformer, and considers the types of tokens (words or entities) when computing attention scores. The proposed model
achieves impressive empirical performance on a wide range of entity-related tasks. In particular, it obtains
state-of-the-art results on five well-known datasets: Open Entity (entity typing), TACRED (relation classification),
CoNLL-2003 (named entity recognition), ReCoRD (cloze-style question answering), and SQuAD 1.1 (extractive question
answering).</em>`,Fn,Qe,qs='This model was contributed by <a href="https://huggingface.co/ikuyamada" rel="nofollow">ikuyamada</a> and <a href="https://huggingface.co/nielsr" rel="nofollow">nielsr</a>. The original code can be found <a href="https://github.com/studio-ousia/luke" rel="nofollow">here</a>.',Wn,Ae,In,Pe,Rs=`<li><p>This implementation is the same as <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel">RobertaModel</a> with the addition of entity embeddings as well
as an entity-aware self-attention mechanism, which improves performance on tasks involving reasoning about entities.</p></li> <li><p>LUKE treats entities as input tokens; therefore, it takes <code>entity_ids</code>, <code>entity_attention_mask</code>,
<code>entity_token_type_ids</code> and <code>entity_position_ids</code> as extra input. You can obtain those using
<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeTokenizer">LukeTokenizer</a>.</p></li> <li><p><a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeTokenizer">LukeTokenizer</a> takes <code>entities</code> and <code>entity_spans</code> (character-based start and end
positions of the entities in the input text) as extra input. <code>entities</code> typically consist of [MASK] entities or
Wikipedia entities. The brief description when inputting these entities are as follows:</p> <ul><li><em>Inputting [MASK] entities to compute entity representations</em>: The [MASK] entity is used to mask entities to be
predicted during pretraining. When LUKE receives the [MASK] entity, it tries to predict the original entity by
gathering the information about the entity from the input text. Therefore, the [MASK] entity can be used to address
downstream tasks requiring the information of entities in text such as entity typing, relation classification, and
named entity recognition.</li> <li><em>Inputting Wikipedia entities to compute knowledge-enhanced token representations</em>: LUKE learns rich information
(or knowledge) about Wikipedia entities during pretraining and stores the information in its entity embedding. By
using Wikipedia entities as input tokens, LUKE outputs token representations enriched by the information stored in
the embeddings of these entities. This is particularly effective for tasks requiring real-world knowledge, such as
question answering.</li></ul></li> <li><p>There are three head models for the former use case:</p> <ul><li><a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntityClassification">LukeForEntityClassification</a>, for tasks to classify a single entity in an input text such as
entity typing, e.g. the <a href="https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html" rel="nofollow">Open Entity dataset</a>.
This model places a linear head on top of the output entity representation.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntityPairClassification">LukeForEntityPairClassification</a>, for tasks to classify the relationship between two entities
such as relation classification, e.g. the <a href="https://nlp.stanford.edu/projects/tacred/" rel="nofollow">TACRED dataset</a>. This
model places a linear head on top of the concatenated output representation of the pair of given entities.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntitySpanClassification">LukeForEntitySpanClassification</a>, for tasks to classify the sequence of entity spans, such as
named entity recognition (NER). This model places a linear head on top of the output entity representations. You
can address NER using this model by inputting all possible entity spans in the text to the model.</li></ul> <p><a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeTokenizer">LukeTokenizer</a> has a <code>task</code> argument, which enables you to easily create an input to these
head models by specifying <code>task=&quot;entity_classification&quot;</code>, <code>task=&quot;entity_pair_classification&quot;</code>, or
<code>task=&quot;entity_span_classification&quot;</code>. Please refer to the example code of each head models.</p></li>`,Zn,Ye,Es="Usage example:",Bn,Oe,qn,De,Rn,Ke,Ns='<li><a href="https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LUKE" rel="nofollow">A demo notebook on how to fine-tune [LukeForEntityPairClassification](/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntityPairClassification) for relation classification</a></li> <li><a href="https://github.com/studio-ousia/luke/tree/master/notebooks" rel="nofollow">Notebooks showcasing how you to reproduce the results as reported in the paper with the HuggingFace implementation of LUKE</a></li> <li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',En,et,Nn,N,tt,mo,It,Vs=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeModel">LukeModel</a>. It is used to instantiate a LUKE
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the LUKE
<a href="https://huggingface.co/studio-ousia/luke-base" rel="nofollow">studio-ousia/luke-base</a> architecture.`,uo,Zt,Gs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,ho,ke,Vn,nt,Gn,z,ot,fo,Bt,Xs="Constructs a LUKE tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.",go,qt,Ss="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",_o,be,yo,Rt,Hs=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.`,ko,Te,bo,Et,Qs=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods. It also creates entity sequences, namely
<code>entity_ids</code>, <code>entity_attention_mask</code>, <code>entity_token_type_ids</code>, and <code>entity_position_ids</code> to be used by the LUKE
model.`,To,Me,st,Mo,Nt,As=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences, depending on the task you want to prepare them for.`,wo,Vt,at,Xn,rt,Sn,x,it,vo,Gt,Ps="The bare LUKE model transformer outputting raw hidden-states for both word tokens and entities without any",jo,Xt,Ys=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zo,St,Os=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Lo,K,lt,Jo,Ht,Ds='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeModel">LukeModel</a> forward method, overrides the <code>__call__</code> special method.',Uo,we,xo,ve,Hn,dt,Qn,C,ct,Co,Qt,Ks=`The LUKE model with a language modeling head and entity prediction head on top for masked language modeling and
masked entity prediction.`,$o,At,ea=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Fo,Pt,ta=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wo,ee,pt,Io,Yt,na='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMaskedLM">LukeForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Zo,je,Bo,ze,An,mt,Pn,$,ut,qo,Ot,oa=`The LUKE model with a classification head on top (a linear layer on top of the hidden state of the first entity
token) for entity classification tasks, such as Open Entity.`,Ro,Dt,sa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Eo,Kt,aa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,No,te,ht,Vo,en,ra='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntityClassification">LukeForEntityClassification</a> forward method, overrides the <code>__call__</code> special method.',Go,Le,Xo,Je,Yn,ft,On,F,gt,So,tn,ia=`The LUKE model with a classification head on top (a linear layer on top of the hidden states of the two entity
tokens) for entity pair classification tasks, such as TACRED.`,Ho,nn,la=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qo,on,da=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ao,ne,_t,Po,sn,ca='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntityPairClassification">LukeForEntityPairClassification</a> forward method, overrides the <code>__call__</code> special method.',Yo,Ue,Oo,xe,Dn,yt,Kn,W,kt,Do,an,pa=`The LUKE model with a span classification head on top (a linear layer on top of the hidden states output) for tasks
such as named entity recognition.`,Ko,rn,ma=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,es,ln,ua=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ts,oe,bt,ns,dn,ha='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntitySpanClassification">LukeForEntitySpanClassification</a> forward method, overrides the <code>__call__</code> special method.',os,Ce,ss,$e,eo,Tt,to,I,Mt,as,cn,fa=`The LUKE Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,rs,pn,ga=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,is,mn,_a=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ls,E,wt,ds,un,ya='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForSequenceClassification">LukeForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',cs,Fe,ps,We,ms,Ie,no,vt,oo,Z,jt,us,hn,ka=`The Luke Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,hs,fn,ba=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,fs,gn,Ta=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,gs,se,zt,_s,_n,Ma='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMultipleChoice">LukeForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',ys,Ze,ks,Be,so,Lt,ao,B,Jt,bs,yn,wa=`The LUKE Model with a token classification head on top (a linear layer on top of the hidden-states output). To
solve Named-Entity Recognition (NER) task using LUKE, <code>LukeForEntitySpanClassification</code> is more suitable than this
class.`,Ts,kn,va=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ms,bn,ja=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ws,ae,Ut,vs,Tn,za='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForTokenClassification">LukeForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',js,qe,zs,Re,ro,xt,io,q,Ct,Ls,Mn,La=`The Luke transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Js,wn,Ja=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Us,vn,Ua=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xs,re,$t,Cs,jn,xa='The <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForQuestionAnswering">LukeForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',$s,Ee,Fs,Ne,lo,Ft,co,zn,po;return Ve=new R({props:{title:"LUKE",local:"luke",headingTag:"h1"}}),Ge=new R({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ae=new R({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Oe=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEx1a2VUb2tlbml6ZXIlMkMlMjBMdWtlTW9kZWwlMkMlMjBMdWtlRm9yRW50aXR5UGFpckNsYXNzaWZpY2F0aW9uJTBBJTBBbW9kZWwlMjAlM0QlMjBMdWtlTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMnN0dWRpby1vdXNpYSUyRmx1a2UtYmFzZSUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBMdWtlVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzdHVkaW8tb3VzaWElMkZsdWtlLWJhc2UlMjIpJTBBJTBBdGV4dCUyMCUzRCUyMCUyMkJleW9uYyVDMyVBOSUyMGxpdmVzJTIwaW4lMjBMb3MlMjBBbmdlbGVzLiUyMiUwQWVudGl0eV9zcGFucyUyMCUzRCUyMCU1QigwJTJDJTIwNyklNUQlMjAlMjAlMjMlMjBjaGFyYWN0ZXItYmFzZWQlMjBlbnRpdHklMjBzcGFuJTIwY29ycmVzcG9uZGluZyUyMHRvJTIwJTIyQmV5b25jJUMzJUE5JTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHRleHQlMkMlMjBlbnRpdHlfc3BhbnMlM0RlbnRpdHlfc3BhbnMlMkMlMjBhZGRfcHJlZml4X3NwYWNlJTNEVHJ1ZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQXdvcmRfbGFzdF9oaWRkZW5fc3RhdGUlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBZW50aXR5X2xhc3RfaGlkZGVuX3N0YXRlJTIwJTNEJTIwb3V0cHV0cy5lbnRpdHlfbGFzdF9oaWRkZW5fc3RhdGUlMEElMEFlbnRpdGllcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMkJleW9uYyVDMyVBOSUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMkxvcyUyMEFuZ2VsZXMlMjIlMkMlMEElNUQlMjAlMjAlMjMlMjBXaWtpcGVkaWElMjBlbnRpdHklMjB0aXRsZXMlMjBjb3JyZXNwb25kaW5nJTIwdG8lMjB0aGUlMjBlbnRpdHklMjBtZW50aW9ucyUyMCUyMkJleW9uYyVDMyVBOSUyMiUyMGFuZCUyMCUyMkxvcyUyMEFuZ2VsZXMlMjIlMEFlbnRpdHlfc3BhbnMlMjAlM0QlMjAlNUIoMCUyQyUyMDcpJTJDJTIwKDE3JTJDJTIwMjgpJTVEJTIwJTIwJTIzJTIwY2hhcmFjdGVyLWJhc2VkJTIwZW50aXR5JTIwc3BhbnMlMjBjb3JyZXNwb25kaW5nJTIwdG8lMjAlMjJCZXlvbmMlQzMlQTklMjIlMjBhbmQlMjAlMjJMb3MlMjBBbmdlbGVzJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHRleHQlMkMlMjBlbnRpdGllcyUzRGVudGl0aWVzJTJDJTIwZW50aXR5X3NwYW5zJTNEZW50aXR5X3NwYW5zJTJDJTIwYWRkX3ByZWZpeF9zcGFjZSUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEF3b3JkX2xhc3RfaGlkZGVuX3N0YXRlJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZSUwQWVudGl0eV9sYXN0X2hpZGRlbl9zdGF0ZSUyMCUzRCUyMG91dHB1dHMuZW50aXR5X2xhc3RfaGlkZGVuX3N0YXRlJTBBJTBBbW9kZWwlMjAlM0QlMjBMdWtlRm9yRW50aXR5UGFpckNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJzdHVkaW8tb3VzaWElMkZsdWtlLWxhcmdlLWZpbmV0dW5lZC10YWNyZWQlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwTHVrZVRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyc3R1ZGlvLW91c2lhJTJGbHVrZS1sYXJnZS1maW5ldHVuZWQtdGFjcmVkJTIyKSUwQWVudGl0eV9zcGFucyUyMCUzRCUyMCU1QigwJTJDJTIwNyklMkMlMjAoMTclMkMlMjAyOCklNUQlMjAlMjAlMjMlMjBjaGFyYWN0ZXItYmFzZWQlMjBlbnRpdHklMjBzcGFucyUyMGNvcnJlc3BvbmRpbmclMjB0byUyMCUyMkJleW9uYyVDMyVBOSUyMiUyMGFuZCUyMCUyMkxvcyUyMEFuZ2VsZXMlMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIodGV4dCUyQyUyMGVudGl0eV9zcGFucyUzRGVudGl0eV9zcGFucyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBcHJlZGljdGVkX2NsYXNzX2lkeCUyMCUzRCUyMGludChsb2dpdHMlNUIwJTVELmFyZ21heCgpKSUwQXByaW50KCUyMlByZWRpY3RlZCUyMGNsYXNzJTNBJTIyJTJDJTIwbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkeCU1RCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LukeTokenizer, LukeModel, LukeForEntityPairClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeModel.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LukeTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-base&quot;</span>)
<span class="hljs-comment"># Example 1: Computing the contextualized entity representation corresponding to the entity mention &quot;Beyoncé&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;Beyoncé lives in Los Angeles.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_spans = [(<span class="hljs-number">0</span>, <span class="hljs-number">7</span>)]  <span class="hljs-comment"># character-based entity span corresponding to &quot;Beyoncé&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>word_last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_last_hidden_state = outputs.entity_last_hidden_state
<span class="hljs-comment"># Example 2: Inputting Wikipedia entities to obtain enriched contextualized representations</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>entities = [
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Beyoncé&quot;</span>,
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Los Angeles&quot;</span>,
<span class="hljs-meta">... </span>]  <span class="hljs-comment"># Wikipedia entity titles corresponding to the entity mentions &quot;Beyoncé&quot; and &quot;Los Angeles&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_spans = [(<span class="hljs-number">0</span>, <span class="hljs-number">7</span>), (<span class="hljs-number">17</span>, <span class="hljs-number">28</span>)]  <span class="hljs-comment"># character-based entity spans corresponding to &quot;Beyoncé&quot; and &quot;Los Angeles&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>word_last_hidden_state = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_last_hidden_state = outputs.entity_last_hidden_state
<span class="hljs-comment"># Example 3: Classifying the relationship between two entities using LukeForEntityPairClassification head model</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>model = LukeForEntityPairClassification.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-large-finetuned-tacred&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LukeTokenizer.from_pretrained(<span class="hljs-string">&quot;studio-ousia/luke-large-finetuned-tacred&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>entity_spans = [(<span class="hljs-number">0</span>, <span class="hljs-number">7</span>), (<span class="hljs-number">17</span>, <span class="hljs-number">28</span>)]  <span class="hljs-comment"># character-based entity spans corresponding to &quot;Beyoncé&quot; and &quot;Los Angeles&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(text, entity_spans=entity_spans, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_idx = <span class="hljs-built_in">int</span>(logits[<span class="hljs-number">0</span>].argmax())
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Predicted class:&quot;</span>, model.config.id2label[predicted_class_idx])`,wrap:!1}}),De=new R({props:{title:"Resources",local:"resources",headingTag:"h2"}}),et=new R({props:{title:"LukeConfig",local:"transformers.LukeConfig",headingTag:"h2"}}),tt=new L({props:{name:"class transformers.LukeConfig",anchor:"transformers.LukeConfig",parameters:[{name:"vocab_size",val:" = 50267"},{name:"entity_vocab_size",val:" = 500000"},{name:"hidden_size",val:" = 768"},{name:"entity_emb_size",val:" = 256"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"use_entity_aware_attention",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LukeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50267) &#x2014;
Vocabulary size of the LUKE model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeModel">LukeModel</a>.`,name:"vocab_size"},{anchor:"transformers.LukeConfig.entity_vocab_size",description:`<strong>entity_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 500000) &#x2014;
Entity vocabulary size of the LUKE model. Defines the number of different entities that can be represented
by the <code>entity_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeModel">LukeModel</a>.`,name:"entity_vocab_size"},{anchor:"transformers.LukeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.LukeConfig.entity_emb_size",description:`<strong>entity_emb_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The number of dimensions of the entity embedding.`,name:"entity_emb_size"},{anchor:"transformers.LukeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.LukeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.LukeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.LukeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.LukeConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.LukeConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.LukeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.LukeConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeModel">LukeModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.LukeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.LukeConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.LukeConfig.use_entity_aware_attention",description:`<strong>use_entity_aware_attention</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should use the entity-aware self-attention mechanism proposed in <a href="https://huggingface.co/papers/2010.01057" rel="nofollow">LUKE: Deep
Contextualized Entity Representations with Entity-aware Self-attention (Yamada et
al.)</a>.`,name:"use_entity_aware_attention"},{anchor:"transformers.LukeConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"},{anchor:"transformers.LukeConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.LukeConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.LukeConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/configuration_luke.py#L24"}}),ke=new ie({props:{anchor:"transformers.LukeConfig.example",$$slots:{default:[Ba]},$$scope:{ctx:w}}}),nt=new R({props:{title:"LukeTokenizer",local:"transformers.LukeTokenizer",headingTag:"h2"}}),ot=new L({props:{name:"class transformers.LukeTokenizer",anchor:"transformers.LukeTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"entity_vocab_file",val:""},{name:"task",val:" = None"},{name:"max_entity_length",val:" = 32"},{name:"max_mention_length",val:" = 30"},{name:"entity_token_1",val:" = '<ent>'"},{name:"entity_token_2",val:" = '<ent2>'"},{name:"entity_unk_token",val:" = '[UNK]'"},{name:"entity_pad_token",val:" = '[PAD]'"},{name:"entity_mask_token",val:" = '[MASK]'"},{name:"entity_mask2_token",val:" = '[MASK2]'"},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LukeTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.LukeTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.LukeTokenizer.entity_vocab_file",description:`<strong>entity_vocab_file</strong> (<code>str</code>) &#x2014;
Path to the entity vocabulary file.`,name:"entity_vocab_file"},{anchor:"transformers.LukeTokenizer.task",description:`<strong>task</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Task for which you want to prepare sequences. One of <code>&quot;entity_classification&quot;</code>,
<code>&quot;entity_pair_classification&quot;</code>, or <code>&quot;entity_span_classification&quot;</code>. If you specify this argument, the entity
sequence is automatically created based on the given entity span(s).`,name:"task"},{anchor:"transformers.LukeTokenizer.max_entity_length",description:`<strong>max_entity_length</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The maximum length of <code>entity_ids</code>.`,name:"max_entity_length"},{anchor:"transformers.LukeTokenizer.max_mention_length",description:`<strong>max_mention_length</strong> (<code>int</code>, <em>optional</em>, defaults to 30) &#x2014;
The maximum number of tokens inside an entity span.`,name:"max_mention_length"},{anchor:"transformers.LukeTokenizer.entity_token_1",description:`<strong>entity_token_1</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;ent&gt;</code>) &#x2014;
The special token used to represent an entity span in a word token sequence. This token is only used when
<code>task</code> is set to <code>&quot;entity_classification&quot;</code> or <code>&quot;entity_pair_classification&quot;</code>.`,name:"entity_token_1"},{anchor:"transformers.LukeTokenizer.entity_token_2",description:`<strong>entity_token_2</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;ent2&gt;</code>) &#x2014;
The special token used to represent an entity span in a word token sequence. This token is only used when
<code>task</code> is set to <code>&quot;entity_pair_classification&quot;</code>.`,name:"entity_token_2"},{anchor:"transformers.LukeTokenizer.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.LukeTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.LukeTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.LukeTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.LukeTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.LukeTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LukeTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.LukeTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.LukeTokenizer.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (LUKE tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/tokenization_luke.py#L174"}}),be=new ie({props:{anchor:"transformers.LukeTokenizer.example",$$slots:{default:[qa]},$$scope:{ctx:w}}}),Te=new _e({props:{$$slots:{default:[Ra]},$$scope:{ctx:w}}}),st=new L({props:{name:"__call__",anchor:"transformers.LukeTokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str]]"},{name:"text_pair",val:": typing.Union[str, list[str], NoneType] = None"},{name:"entity_spans",val:": typing.Union[list[tuple[int, int]], list[list[tuple[int, int]]], NoneType] = None"},{name:"entity_spans_pair",val:": typing.Union[list[tuple[int, int]], list[list[tuple[int, int]]], NoneType] = None"},{name:"entities",val:": typing.Union[list[str], list[list[str]], NoneType] = None"},{name:"entities_pair",val:": typing.Union[list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"max_entity_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": typing.Optional[bool] = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LukeTokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
tokenizer does not support tokenization based on pretokenized strings.`,name:"text"},{anchor:"transformers.LukeTokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
tokenizer does not support tokenization based on pretokenized strings.`,name:"text_pair"},{anchor:"transformers.LukeTokenizer.__call__.entity_spans",description:`<strong>entity_spans</strong> (<code>list[tuple[int, int]]</code>, <code>list[list[tuple[int, int]]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
with two integers denoting character-based start and end positions of entities. If you specify
<code>&quot;entity_classification&quot;</code> or <code>&quot;entity_pair_classification&quot;</code> as the <code>task</code> argument in the constructor,
the length of each sequence must be 1 or 2, respectively. If you specify <code>entities</code>, the length of each
sequence must be equal to the length of each sequence of <code>entities</code>.`,name:"entity_spans"},{anchor:"transformers.LukeTokenizer.__call__.entity_spans_pair",description:`<strong>entity_spans_pair</strong> (<code>list[tuple[int, int]]</code>, <code>list[list[tuple[int, int]]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
with two integers denoting character-based start and end positions of entities. If you specify the
<code>task</code> argument in the constructor, this argument is ignored. If you specify <code>entities_pair</code>, the
length of each sequence must be equal to the length of each sequence of <code>entities_pair</code>.`,name:"entity_spans_pair"},{anchor:"transformers.LukeTokenizer.__call__.entities",description:`<strong>entities</strong> (<code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
Angeles). This argument is ignored if you specify the <code>task</code> argument in the constructor. The length of
each sequence must be equal to the length of each sequence of <code>entity_spans</code>. If you specify
<code>entity_spans</code> without specifying this argument, the entity sequence or the batch of entity sequences
is automatically constructed by filling it with the [MASK] entity.`,name:"entities"},{anchor:"transformers.LukeTokenizer.__call__.entities_pair",description:`<strong>entities_pair</strong> (<code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
Angeles). This argument is ignored if you specify the <code>task</code> argument in the constructor. The length of
each sequence must be equal to the length of each sequence of <code>entity_spans_pair</code>. If you specify
<code>entity_spans_pair</code> without specifying this argument, the entity sequence or the batch of entity
sequences is automatically constructed by filling it with the [MASK] entity.`,name:"entities_pair"},{anchor:"transformers.LukeTokenizer.__call__.max_entity_length",description:`<strong>max_entity_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The maximum length of <code>entity_ids</code>.`,name:"max_entity_length"},{anchor:"transformers.LukeTokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.LukeTokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.LukeTokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
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
</ul>`,name:"truncation"},{anchor:"transformers.LukeTokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.LukeTokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.LukeTokenizer.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.LukeTokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.LukeTokenizer.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.LukeTokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.LukeTokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.LukeTokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.LukeTokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.LukeTokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.LukeTokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.LukeTokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.LukeTokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.LukeTokenizer.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/tokenization_luke.py#L556",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
<p><strong>entity_ids</strong> — List of entity ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>entity_position_ids</strong> — List of entity positions in the input sequence to be fed to a model.</p>
</li>
<li>
<p><strong>entity_token_type_ids</strong> — List of entity token type ids to be fed to a model (when
<code>return_token_type_ids=True</code> or if <em>“entity_token_type_ids”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a></p>
</li>
<li>
<p><strong>entity_attention_mask</strong> — List of indices specifying which entities should be attended to by the model
(when <code>return_attention_mask=True</code> or if <em>“entity_attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>entity_start_positions</strong> — List of the start positions of entities in the word token sequence (when
<code>task="entity_span_classification"</code>).</p>
</li>
<li>
<p><strong>entity_end_positions</strong> — List of the end positions of entities in the word token sequence (when
<code>task="entity_span_classification"</code>).</p>
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
`}}),at=new L({props:{name:"save_vocabulary",anchor:"transformers.LukeTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/tokenization_luke.py#L1694"}}),rt=new R({props:{title:"LukeModel",local:"transformers.LukeModel",headingTag:"h2"}}),it=new L({props:{name:"class transformers.LukeModel",anchor:"transformers.LukeModel",parameters:[{name:"config",val:": LukeConfig"},{name:"add_pooling_layer",val:": bool = True"}],parametersDescription:[{anchor:"transformers.LukeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig">LukeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.LukeModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L811"}}),lt=new L({props:{name:"forward",anchor:"transformers.LukeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeModel.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeModel.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeModel.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeModel.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L844",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.BaseLukeModelOutputWithPooling</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, hidden_size)</code>) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a
Linear layer and a Tanh activation function.</p>
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
<li>
<p><strong>entity_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length, hidden_size)</code>) — Sequence of entity hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.BaseLukeModelOutputWithPooling</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),we=new _e({props:{$$slots:{default:[Ea]},$$scope:{ctx:w}}}),ve=new ie({props:{anchor:"transformers.LukeModel.forward.example",$$slots:{default:[Na]},$$scope:{ctx:w}}}),dt=new R({props:{title:"LukeForMaskedLM",local:"transformers.LukeForMaskedLM",headingTag:"h2"}}),ct=new L({props:{name:"class transformers.LukeForMaskedLM",anchor:"transformers.LukeForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LukeForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMaskedLM">LukeForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1085"}}),pt=new L({props:{name:"forward",anchor:"transformers.LukeForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeForMaskedLM.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeForMaskedLM.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeForMaskedLM.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeForMaskedLM.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.LukeForMaskedLM.forward.entity_labels",description:`<strong>entity_labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"entity_labels"},{anchor:"transformers.LukeForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1111",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.LukeMaskedLMOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — The sum of masked language modeling (MLM) loss and entity prediction loss.</p>
</li>
<li>
<p><strong>mlm_loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Masked language modeling (MLM) loss.</p>
</li>
<li>
<p><strong>mep_loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Masked entity prediction (MEP) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>entity_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the entity prediction head (scores for each entity vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.LukeMaskedLMOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),je=new _e({props:{$$slots:{default:[Va]},$$scope:{ctx:w}}}),ze=new ie({props:{anchor:"transformers.LukeForMaskedLM.forward.example",$$slots:{default:[Ga]},$$scope:{ctx:w}}}),mt=new R({props:{title:"LukeForEntityClassification",local:"transformers.LukeForEntityClassification",headingTag:"h2"}}),ut=new L({props:{name:"class transformers.LukeForEntityClassification",anchor:"transformers.LukeForEntityClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LukeForEntityClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntityClassification">LukeForEntityClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1234"}}),ht=new L({props:{name:"forward",anchor:"transformers.LukeForEntityClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeForEntityClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeForEntityClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeForEntityClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeForEntityClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeForEntityClassification.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeForEntityClassification.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeForEntityClassification.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeForEntityClassification.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeForEntityClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeForEntityClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeForEntityClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code> or <code>(batch_size, num_labels)</code>, <em>optional</em>) &#x2014;
Labels for computing the classification loss. If the shape is <code>(batch_size,)</code>, the cross entropy loss is
used for the single-label classification. In this case, labels should contain the indices that should be in
<code>[0, ..., config.num_labels - 1]</code>. If the shape is <code>(batch_size, num_labels)</code>, the binary cross entropy
loss is used for the multi-label classification. In this case, labels should only contain <code>[0, 1]</code>, where 0
and 1 indicate false and true, respectively.`,name:"labels"},{anchor:"transformers.LukeForEntityClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeForEntityClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeForEntityClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1247",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.EntityClassificationOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.EntityClassificationOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Le=new _e({props:{$$slots:{default:[Xa]},$$scope:{ctx:w}}}),Je=new ie({props:{anchor:"transformers.LukeForEntityClassification.forward.example",$$slots:{default:[Sa]},$$scope:{ctx:w}}}),ft=new R({props:{title:"LukeForEntityPairClassification",local:"transformers.LukeForEntityPairClassification",headingTag:"h2"}}),gt=new L({props:{name:"class transformers.LukeForEntityPairClassification",anchor:"transformers.LukeForEntityPairClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LukeForEntityPairClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntityPairClassification">LukeForEntityPairClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1364"}}),_t=new L({props:{name:"forward",anchor:"transformers.LukeForEntityPairClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeForEntityPairClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeForEntityPairClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeForEntityPairClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeForEntityPairClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeForEntityPairClassification.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeForEntityPairClassification.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeForEntityPairClassification.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeForEntityPairClassification.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeForEntityPairClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeForEntityPairClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeForEntityPairClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code> or <code>(batch_size, num_labels)</code>, <em>optional</em>) &#x2014;
Labels for computing the classification loss. If the shape is <code>(batch_size,)</code>, the cross entropy loss is
used for the single-label classification. In this case, labels should contain the indices that should be in
<code>[0, ..., config.num_labels - 1]</code>. If the shape is <code>(batch_size, num_labels)</code>, the binary cross entropy
loss is used for the multi-label classification. In this case, labels should only contain <code>[0, 1]</code>, where 0
and 1 indicate false and true, respectively.`,name:"labels"},{anchor:"transformers.LukeForEntityPairClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeForEntityPairClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeForEntityPairClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1377",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.EntityPairClassificationOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.EntityPairClassificationOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ue=new _e({props:{$$slots:{default:[Ha]},$$scope:{ctx:w}}}),xe=new ie({props:{anchor:"transformers.LukeForEntityPairClassification.forward.example",$$slots:{default:[Qa]},$$scope:{ctx:w}}}),yt=new R({props:{title:"LukeForEntitySpanClassification",local:"transformers.LukeForEntitySpanClassification",headingTag:"h2"}}),kt=new L({props:{name:"class transformers.LukeForEntitySpanClassification",anchor:"transformers.LukeForEntitySpanClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LukeForEntitySpanClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForEntitySpanClassification">LukeForEntitySpanClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1499"}}),bt=new L({props:{name:"forward",anchor:"transformers.LukeForEntitySpanClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeForEntitySpanClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeForEntitySpanClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeForEntitySpanClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeForEntitySpanClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeForEntitySpanClassification.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeForEntitySpanClassification.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeForEntitySpanClassification.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeForEntitySpanClassification.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeForEntitySpanClassification.forward.entity_start_positions",description:`<strong>entity_start_positions</strong> (<code>torch.LongTensor</code>, <em>optional</em>) &#x2014;
The start positions of entities in the word token sequence.`,name:"entity_start_positions"},{anchor:"transformers.LukeForEntitySpanClassification.forward.entity_end_positions",description:`<strong>entity_end_positions</strong> (<code>torch.LongTensor</code>, <em>optional</em>) &#x2014;
The end positions of entities in the word token sequence.`,name:"entity_end_positions"},{anchor:"transformers.LukeForEntitySpanClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeForEntitySpanClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeForEntitySpanClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code> or <code>(batch_size, entity_length, num_labels)</code>, <em>optional</em>) &#x2014;
Labels for computing the classification loss. If the shape is <code>(batch_size, entity_length)</code>, the cross
entropy loss is used for the single-label classification. In this case, labels should contain the indices
that should be in <code>[0, ..., config.num_labels - 1]</code>. If the shape is <code>(batch_size, entity_length, num_labels)</code>, the binary cross entropy loss is used for the multi-label classification. In this case,
labels should only contain <code>[0, 1]</code>, where 0 and 1 indicate false and true, respectively.`,name:"labels"},{anchor:"transformers.LukeForEntitySpanClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeForEntitySpanClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeForEntitySpanClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1512",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.EntitySpanClassificationOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length, config.num_labels)</code>) — Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.EntitySpanClassificationOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ce=new _e({props:{$$slots:{default:[Aa]},$$scope:{ctx:w}}}),$e=new ie({props:{anchor:"transformers.LukeForEntitySpanClassification.forward.example",$$slots:{default:[Pa]},$$scope:{ctx:w}}}),Tt=new R({props:{title:"LukeForSequenceClassification",local:"transformers.LukeForSequenceClassification",headingTag:"h2"}}),Mt=new L({props:{name:"class transformers.LukeForSequenceClassification",anchor:"transformers.LukeForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LukeForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForSequenceClassification">LukeForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1658"}}),wt=new L({props:{name:"forward",anchor:"transformers.LukeForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeForSequenceClassification.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeForSequenceClassification.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeForSequenceClassification.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeForSequenceClassification.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.LukeForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1671",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.LukeSequenceClassifierOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.LukeSequenceClassifierOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Fe=new _e({props:{$$slots:{default:[Ya]},$$scope:{ctx:w}}}),We=new ie({props:{anchor:"transformers.LukeForSequenceClassification.forward.example",$$slots:{default:[Oa]},$$scope:{ctx:w}}}),Ie=new ie({props:{anchor:"transformers.LukeForSequenceClassification.forward.example-2",$$slots:{default:[Da]},$$scope:{ctx:w}}}),vt=new R({props:{title:"LukeForMultipleChoice",local:"transformers.LukeForMultipleChoice",headingTag:"h2"}}),jt=new L({props:{name:"class transformers.LukeForMultipleChoice",anchor:"transformers.LukeForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LukeForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForMultipleChoice">LukeForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L2008"}}),zt=new L({props:{name:"forward",anchor:"transformers.LukeForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeForMultipleChoice.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeForMultipleChoice.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeForMultipleChoice.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeForMultipleChoice.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.LukeForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L2021",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.LukeMultipleChoiceModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) — <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.LukeMultipleChoiceModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ze=new _e({props:{$$slots:{default:[Ka]},$$scope:{ctx:w}}}),Be=new ie({props:{anchor:"transformers.LukeForMultipleChoice.forward.example",$$slots:{default:[er]},$$scope:{ctx:w}}}),Lt=new R({props:{title:"LukeForTokenClassification",local:"transformers.LukeForTokenClassification",headingTag:"h2"}}),Jt=new L({props:{name:"class transformers.LukeForTokenClassification",anchor:"transformers.LukeForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LukeForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForTokenClassification">LukeForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1785"}}),Ut=new L({props:{name:"forward",anchor:"transformers.LukeForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeForTokenClassification.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeForTokenClassification.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeForTokenClassification.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeForTokenClassification.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.LukeForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1799",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.LukeTokenClassifierOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) — Classification scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.LukeTokenClassifierOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),qe=new _e({props:{$$slots:{default:[tr]},$$scope:{ctx:w}}}),Re=new ie({props:{anchor:"transformers.LukeForTokenClassification.forward.example",$$slots:{default:[nr]},$$scope:{ctx:w}}}),xt=new R({props:{title:"LukeForQuestionAnswering",local:"transformers.LukeForQuestionAnswering",headingTag:"h2"}}),Ct=new L({props:{name:"class transformers.LukeForQuestionAnswering",anchor:"transformers.LukeForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LukeForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeForQuestionAnswering">LukeForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1889"}}),$t=new L({props:{name:"forward",anchor:"transformers.LukeForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.FloatTensor] = None"},{name:"entity_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"entity_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"entity_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LukeForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LukeForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LukeForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.LukeForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LukeForQuestionAnswering.forward.entity_ids",description:`<strong>entity_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>) &#x2014;
Indices of entity tokens in the entity vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.`,name:"entity_ids"},{anchor:"transformers.LukeForQuestionAnswering.forward.entity_attention_mask",description:`<strong>entity_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding entity token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for entity tokens that are <strong>not masked</strong>,</li>
<li>0 for entity tokens that are <strong>masked</strong>.</li>
</ul>`,name:"entity_attention_mask"},{anchor:"transformers.LukeForQuestionAnswering.forward.entity_token_type_ids",description:`<strong>entity_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the entity token inputs. Indices are
selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>portion A</em> entity token,</li>
<li>1 corresponds to a <em>portion B</em> entity token.</li>
</ul>`,name:"entity_token_type_ids"},{anchor:"transformers.LukeForQuestionAnswering.forward.entity_position_ids",description:`<strong>entity_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, entity_length, max_mention_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input entity in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.`,name:"entity_position_ids"},{anchor:"transformers.LukeForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LukeForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LukeForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.LukeForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.LukeForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LukeForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LukeForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/luke/modeling_luke.py#L1901",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.luke.modeling_luke.LukeQuestionAnsweringModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/luke#transformers.LukeConfig"
>LukeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.</p>
</li>
<li>
<p><strong>start_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>, defaults to <code>None</code>) — Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>, defaults to <code>None</code>) — Span-end scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>entity_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, entity_length, hidden_size)</code>. Entity hidden-states of the model at the output of each
layer plus the initial entity embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.luke.modeling_luke.LukeQuestionAnsweringModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ee=new _e({props:{$$slots:{default:[or]},$$scope:{ctx:w}}}),Ne=new ie({props:{anchor:"transformers.LukeForQuestionAnswering.forward.example",$$slots:{default:[sr]},$$scope:{ctx:w}}}),Ft=new Za({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/luke.md"}}),{c(){t=c("meta"),b=s(),i=c("p"),d=s(),T=c("p"),T.innerHTML=n,M=s(),h(Ve.$$.fragment),Jn=s(),ye=c("div"),ye.innerHTML=Ws,Un=s(),h(Ge.$$.fragment),xn=s(),Xe=c("p"),Xe.innerHTML=Is,Cn=s(),Se=c("p"),Se.textContent=Zs,$n=s(),He=c("p"),He.innerHTML=Bs,Fn=s(),Qe=c("p"),Qe.innerHTML=qs,Wn=s(),h(Ae.$$.fragment),In=s(),Pe=c("ul"),Pe.innerHTML=Rs,Zn=s(),Ye=c("p"),Ye.textContent=Es,Bn=s(),h(Oe.$$.fragment),qn=s(),h(De.$$.fragment),Rn=s(),Ke=c("ul"),Ke.innerHTML=Ns,En=s(),h(et.$$.fragment),Nn=s(),N=c("div"),h(tt.$$.fragment),mo=s(),It=c("p"),It.innerHTML=Vs,uo=s(),Zt=c("p"),Zt.innerHTML=Gs,ho=s(),h(ke.$$.fragment),Vn=s(),h(nt.$$.fragment),Gn=s(),z=c("div"),h(ot.$$.fragment),fo=s(),Bt=c("p"),Bt.textContent=Xs,go=s(),qt=c("p"),qt.textContent=Ss,_o=s(),h(be.$$.fragment),yo=s(),Rt=c("p"),Rt.innerHTML=Hs,ko=s(),h(Te.$$.fragment),bo=s(),Et=c("p"),Et.innerHTML=Qs,To=s(),Me=c("div"),h(st.$$.fragment),Mo=s(),Nt=c("p"),Nt.textContent=As,wo=s(),Vt=c("div"),h(at.$$.fragment),Xn=s(),h(rt.$$.fragment),Sn=s(),x=c("div"),h(it.$$.fragment),vo=s(),Gt=c("p"),Gt.textContent=Ps,jo=s(),Xt=c("p"),Xt.innerHTML=Ys,zo=s(),St=c("p"),St.innerHTML=Os,Lo=s(),K=c("div"),h(lt.$$.fragment),Jo=s(),Ht=c("p"),Ht.innerHTML=Ds,Uo=s(),h(we.$$.fragment),xo=s(),h(ve.$$.fragment),Hn=s(),h(dt.$$.fragment),Qn=s(),C=c("div"),h(ct.$$.fragment),Co=s(),Qt=c("p"),Qt.textContent=Ks,$o=s(),At=c("p"),At.innerHTML=ea,Fo=s(),Pt=c("p"),Pt.innerHTML=ta,Wo=s(),ee=c("div"),h(pt.$$.fragment),Io=s(),Yt=c("p"),Yt.innerHTML=na,Zo=s(),h(je.$$.fragment),Bo=s(),h(ze.$$.fragment),An=s(),h(mt.$$.fragment),Pn=s(),$=c("div"),h(ut.$$.fragment),qo=s(),Ot=c("p"),Ot.textContent=oa,Ro=s(),Dt=c("p"),Dt.innerHTML=sa,Eo=s(),Kt=c("p"),Kt.innerHTML=aa,No=s(),te=c("div"),h(ht.$$.fragment),Vo=s(),en=c("p"),en.innerHTML=ra,Go=s(),h(Le.$$.fragment),Xo=s(),h(Je.$$.fragment),Yn=s(),h(ft.$$.fragment),On=s(),F=c("div"),h(gt.$$.fragment),So=s(),tn=c("p"),tn.textContent=ia,Ho=s(),nn=c("p"),nn.innerHTML=la,Qo=s(),on=c("p"),on.innerHTML=da,Ao=s(),ne=c("div"),h(_t.$$.fragment),Po=s(),sn=c("p"),sn.innerHTML=ca,Yo=s(),h(Ue.$$.fragment),Oo=s(),h(xe.$$.fragment),Dn=s(),h(yt.$$.fragment),Kn=s(),W=c("div"),h(kt.$$.fragment),Do=s(),an=c("p"),an.textContent=pa,Ko=s(),rn=c("p"),rn.innerHTML=ma,es=s(),ln=c("p"),ln.innerHTML=ua,ts=s(),oe=c("div"),h(bt.$$.fragment),ns=s(),dn=c("p"),dn.innerHTML=ha,os=s(),h(Ce.$$.fragment),ss=s(),h($e.$$.fragment),eo=s(),h(Tt.$$.fragment),to=s(),I=c("div"),h(Mt.$$.fragment),as=s(),cn=c("p"),cn.textContent=fa,rs=s(),pn=c("p"),pn.innerHTML=ga,is=s(),mn=c("p"),mn.innerHTML=_a,ls=s(),E=c("div"),h(wt.$$.fragment),ds=s(),un=c("p"),un.innerHTML=ya,cs=s(),h(Fe.$$.fragment),ps=s(),h(We.$$.fragment),ms=s(),h(Ie.$$.fragment),no=s(),h(vt.$$.fragment),oo=s(),Z=c("div"),h(jt.$$.fragment),us=s(),hn=c("p"),hn.textContent=ka,hs=s(),fn=c("p"),fn.innerHTML=ba,fs=s(),gn=c("p"),gn.innerHTML=Ta,gs=s(),se=c("div"),h(zt.$$.fragment),_s=s(),_n=c("p"),_n.innerHTML=Ma,ys=s(),h(Ze.$$.fragment),ks=s(),h(Be.$$.fragment),so=s(),h(Lt.$$.fragment),ao=s(),B=c("div"),h(Jt.$$.fragment),bs=s(),yn=c("p"),yn.innerHTML=wa,Ts=s(),kn=c("p"),kn.innerHTML=va,Ms=s(),bn=c("p"),bn.innerHTML=ja,ws=s(),ae=c("div"),h(Ut.$$.fragment),vs=s(),Tn=c("p"),Tn.innerHTML=za,js=s(),h(qe.$$.fragment),zs=s(),h(Re.$$.fragment),ro=s(),h(xt.$$.fragment),io=s(),q=c("div"),h(Ct.$$.fragment),Ls=s(),Mn=c("p"),Mn.innerHTML=La,Js=s(),wn=c("p"),wn.innerHTML=Ja,Us=s(),vn=c("p"),vn.innerHTML=Ua,xs=s(),re=c("div"),h($t.$$.fragment),Cs=s(),jn=c("p"),jn.innerHTML=xa,$s=s(),h(Ee.$$.fragment),Fs=s(),h(Ne.$$.fragment),lo=s(),h(Ft.$$.fragment),co=s(),zn=c("p"),this.h()},l(e){const r=Ia("svelte-u9bgzb",document.head);t=p(r,"META",{name:!0,content:!0}),r.forEach(l),b=a(e),i=p(e,"P",{}),j(i).forEach(l),d=a(e),T=p(e,"P",{"data-svelte-h":!0}),u(T)!=="svelte-1m9kcmq"&&(T.innerHTML=n),M=a(e),f(Ve.$$.fragment,e),Jn=a(e),ye=p(e,"DIV",{class:!0,"data-svelte-h":!0}),u(ye)!=="svelte-13t8s2t"&&(ye.innerHTML=Ws),Un=a(e),f(Ge.$$.fragment,e),xn=a(e),Xe=p(e,"P",{"data-svelte-h":!0}),u(Xe)!=="svelte-11fct62"&&(Xe.innerHTML=Is),Cn=a(e),Se=p(e,"P",{"data-svelte-h":!0}),u(Se)!=="svelte-vfdo9a"&&(Se.textContent=Zs),$n=a(e),He=p(e,"P",{"data-svelte-h":!0}),u(He)!=="svelte-cwy98g"&&(He.innerHTML=Bs),Fn=a(e),Qe=p(e,"P",{"data-svelte-h":!0}),u(Qe)!=="svelte-165dmn4"&&(Qe.innerHTML=qs),Wn=a(e),f(Ae.$$.fragment,e),In=a(e),Pe=p(e,"UL",{"data-svelte-h":!0}),u(Pe)!=="svelte-qgmptf"&&(Pe.innerHTML=Rs),Zn=a(e),Ye=p(e,"P",{"data-svelte-h":!0}),u(Ye)!=="svelte-1vn55jb"&&(Ye.textContent=Es),Bn=a(e),f(Oe.$$.fragment,e),qn=a(e),f(De.$$.fragment,e),Rn=a(e),Ke=p(e,"UL",{"data-svelte-h":!0}),u(Ke)!=="svelte-wteugd"&&(Ke.innerHTML=Ns),En=a(e),f(et.$$.fragment,e),Nn=a(e),N=p(e,"DIV",{class:!0});var le=j(N);f(tt.$$.fragment,le),mo=a(le),It=p(le,"P",{"data-svelte-h":!0}),u(It)!=="svelte-rm5gy4"&&(It.innerHTML=Vs),uo=a(le),Zt=p(le,"P",{"data-svelte-h":!0}),u(Zt)!=="svelte-1ek1ss9"&&(Zt.innerHTML=Gs),ho=a(le),f(ke.$$.fragment,le),le.forEach(l),Vn=a(e),f(nt.$$.fragment,e),Gn=a(e),z=p(e,"DIV",{class:!0});var U=j(z);f(ot.$$.fragment,U),fo=a(U),Bt=p(U,"P",{"data-svelte-h":!0}),u(Bt)!=="svelte-1dy4zle"&&(Bt.textContent=Xs),go=a(U),qt=p(U,"P",{"data-svelte-h":!0}),u(qt)!=="svelte-1s077p3"&&(qt.textContent=Ss),_o=a(U),f(be.$$.fragment,U),yo=a(U),Rt=p(U,"P",{"data-svelte-h":!0}),u(Rt)!=="svelte-1jfcabo"&&(Rt.innerHTML=Hs),ko=a(U),f(Te.$$.fragment,U),bo=a(U),Et=p(U,"P",{"data-svelte-h":!0}),u(Et)!=="svelte-iqqkx8"&&(Et.innerHTML=Qs),To=a(U),Me=p(U,"DIV",{class:!0});var Wt=j(Me);f(st.$$.fragment,Wt),Mo=a(Wt),Nt=p(Wt,"P",{"data-svelte-h":!0}),u(Nt)!=="svelte-16lcbtv"&&(Nt.textContent=As),Wt.forEach(l),wo=a(U),Vt=p(U,"DIV",{class:!0});var Ln=j(Vt);f(at.$$.fragment,Ln),Ln.forEach(l),U.forEach(l),Xn=a(e),f(rt.$$.fragment,e),Sn=a(e),x=p(e,"DIV",{class:!0});var V=j(x);f(it.$$.fragment,V),vo=a(V),Gt=p(V,"P",{"data-svelte-h":!0}),u(Gt)!=="svelte-11g8pwm"&&(Gt.textContent=Ps),jo=a(V),Xt=p(V,"P",{"data-svelte-h":!0}),u(Xt)!=="svelte-q52n56"&&(Xt.innerHTML=Ys),zo=a(V),St=p(V,"P",{"data-svelte-h":!0}),u(St)!=="svelte-hswkmf"&&(St.innerHTML=Os),Lo=a(V),K=p(V,"DIV",{class:!0});var de=j(K);f(lt.$$.fragment,de),Jo=a(de),Ht=p(de,"P",{"data-svelte-h":!0}),u(Ht)!=="svelte-7t7rmg"&&(Ht.innerHTML=Ds),Uo=a(de),f(we.$$.fragment,de),xo=a(de),f(ve.$$.fragment,de),de.forEach(l),V.forEach(l),Hn=a(e),f(dt.$$.fragment,e),Qn=a(e),C=p(e,"DIV",{class:!0});var G=j(C);f(ct.$$.fragment,G),Co=a(G),Qt=p(G,"P",{"data-svelte-h":!0}),u(Qt)!=="svelte-1ebahsp"&&(Qt.textContent=Ks),$o=a(G),At=p(G,"P",{"data-svelte-h":!0}),u(At)!=="svelte-q52n56"&&(At.innerHTML=ea),Fo=a(G),Pt=p(G,"P",{"data-svelte-h":!0}),u(Pt)!=="svelte-hswkmf"&&(Pt.innerHTML=ta),Wo=a(G),ee=p(G,"DIV",{class:!0});var ce=j(ee);f(pt.$$.fragment,ce),Io=a(ce),Yt=p(ce,"P",{"data-svelte-h":!0}),u(Yt)!=="svelte-15plojc"&&(Yt.innerHTML=na),Zo=a(ce),f(je.$$.fragment,ce),Bo=a(ce),f(ze.$$.fragment,ce),ce.forEach(l),G.forEach(l),An=a(e),f(mt.$$.fragment,e),Pn=a(e),$=p(e,"DIV",{class:!0});var X=j($);f(ut.$$.fragment,X),qo=a(X),Ot=p(X,"P",{"data-svelte-h":!0}),u(Ot)!=="svelte-fm8z69"&&(Ot.textContent=oa),Ro=a(X),Dt=p(X,"P",{"data-svelte-h":!0}),u(Dt)!=="svelte-q52n56"&&(Dt.innerHTML=sa),Eo=a(X),Kt=p(X,"P",{"data-svelte-h":!0}),u(Kt)!=="svelte-hswkmf"&&(Kt.innerHTML=aa),No=a(X),te=p(X,"DIV",{class:!0});var pe=j(te);f(ht.$$.fragment,pe),Vo=a(pe),en=p(pe,"P",{"data-svelte-h":!0}),u(en)!=="svelte-vgii5w"&&(en.innerHTML=ra),Go=a(pe),f(Le.$$.fragment,pe),Xo=a(pe),f(Je.$$.fragment,pe),pe.forEach(l),X.forEach(l),Yn=a(e),f(ft.$$.fragment,e),On=a(e),F=p(e,"DIV",{class:!0});var S=j(F);f(gt.$$.fragment,S),So=a(S),tn=p(S,"P",{"data-svelte-h":!0}),u(tn)!=="svelte-1qackjh"&&(tn.textContent=ia),Ho=a(S),nn=p(S,"P",{"data-svelte-h":!0}),u(nn)!=="svelte-q52n56"&&(nn.innerHTML=la),Qo=a(S),on=p(S,"P",{"data-svelte-h":!0}),u(on)!=="svelte-hswkmf"&&(on.innerHTML=da),Ao=a(S),ne=p(S,"DIV",{class:!0});var me=j(ne);f(_t.$$.fragment,me),Po=a(me),sn=p(me,"P",{"data-svelte-h":!0}),u(sn)!=="svelte-15n7lnc"&&(sn.innerHTML=ca),Yo=a(me),f(Ue.$$.fragment,me),Oo=a(me),f(xe.$$.fragment,me),me.forEach(l),S.forEach(l),Dn=a(e),f(yt.$$.fragment,e),Kn=a(e),W=p(e,"DIV",{class:!0});var H=j(W);f(kt.$$.fragment,H),Do=a(H),an=p(H,"P",{"data-svelte-h":!0}),u(an)!=="svelte-3wjma0"&&(an.textContent=pa),Ko=a(H),rn=p(H,"P",{"data-svelte-h":!0}),u(rn)!=="svelte-q52n56"&&(rn.innerHTML=ma),es=a(H),ln=p(H,"P",{"data-svelte-h":!0}),u(ln)!=="svelte-hswkmf"&&(ln.innerHTML=ua),ts=a(H),oe=p(H,"DIV",{class:!0});var ue=j(oe);f(bt.$$.fragment,ue),ns=a(ue),dn=p(ue,"P",{"data-svelte-h":!0}),u(dn)!=="svelte-10hv1ns"&&(dn.innerHTML=ha),os=a(ue),f(Ce.$$.fragment,ue),ss=a(ue),f($e.$$.fragment,ue),ue.forEach(l),H.forEach(l),eo=a(e),f(Tt.$$.fragment,e),to=a(e),I=p(e,"DIV",{class:!0});var Q=j(I);f(Mt.$$.fragment,Q),as=a(Q),cn=p(Q,"P",{"data-svelte-h":!0}),u(cn)!=="svelte-1k7t2k"&&(cn.textContent=fa),rs=a(Q),pn=p(Q,"P",{"data-svelte-h":!0}),u(pn)!=="svelte-q52n56"&&(pn.innerHTML=ga),is=a(Q),mn=p(Q,"P",{"data-svelte-h":!0}),u(mn)!=="svelte-hswkmf"&&(mn.innerHTML=_a),ls=a(Q),E=p(Q,"DIV",{class:!0});var A=j(E);f(wt.$$.fragment,A),ds=a(A),un=p(A,"P",{"data-svelte-h":!0}),u(un)!=="svelte-naj9c"&&(un.innerHTML=ya),cs=a(A),f(Fe.$$.fragment,A),ps=a(A),f(We.$$.fragment,A),ms=a(A),f(Ie.$$.fragment,A),A.forEach(l),Q.forEach(l),no=a(e),f(vt.$$.fragment,e),oo=a(e),Z=p(e,"DIV",{class:!0});var P=j(Z);f(jt.$$.fragment,P),us=a(P),hn=p(P,"P",{"data-svelte-h":!0}),u(hn)!=="svelte-r0l024"&&(hn.textContent=ka),hs=a(P),fn=p(P,"P",{"data-svelte-h":!0}),u(fn)!=="svelte-q52n56"&&(fn.innerHTML=ba),fs=a(P),gn=p(P,"P",{"data-svelte-h":!0}),u(gn)!=="svelte-hswkmf"&&(gn.innerHTML=Ta),gs=a(P),se=p(P,"DIV",{class:!0});var he=j(se);f(zt.$$.fragment,he),_s=a(he),_n=p(he,"P",{"data-svelte-h":!0}),u(_n)!=="svelte-1xx1v88"&&(_n.innerHTML=Ma),ys=a(he),f(Ze.$$.fragment,he),ks=a(he),f(Be.$$.fragment,he),he.forEach(l),P.forEach(l),so=a(e),f(Lt.$$.fragment,e),ao=a(e),B=p(e,"DIV",{class:!0});var Y=j(B);f(Jt.$$.fragment,Y),bs=a(Y),yn=p(Y,"P",{"data-svelte-h":!0}),u(yn)!=="svelte-19t8p80"&&(yn.innerHTML=wa),Ts=a(Y),kn=p(Y,"P",{"data-svelte-h":!0}),u(kn)!=="svelte-q52n56"&&(kn.innerHTML=va),Ms=a(Y),bn=p(Y,"P",{"data-svelte-h":!0}),u(bn)!=="svelte-hswkmf"&&(bn.innerHTML=ja),ws=a(Y),ae=p(Y,"DIV",{class:!0});var fe=j(ae);f(Ut.$$.fragment,fe),vs=a(fe),Tn=p(fe,"P",{"data-svelte-h":!0}),u(Tn)!=="svelte-1dmy5ia"&&(Tn.innerHTML=za),js=a(fe),f(qe.$$.fragment,fe),zs=a(fe),f(Re.$$.fragment,fe),fe.forEach(l),Y.forEach(l),ro=a(e),f(xt.$$.fragment,e),io=a(e),q=p(e,"DIV",{class:!0});var O=j(q);f(Ct.$$.fragment,O),Ls=a(O),Mn=p(O,"P",{"data-svelte-h":!0}),u(Mn)!=="svelte-5zrlu8"&&(Mn.innerHTML=La),Js=a(O),wn=p(O,"P",{"data-svelte-h":!0}),u(wn)!=="svelte-q52n56"&&(wn.innerHTML=Ja),Us=a(O),vn=p(O,"P",{"data-svelte-h":!0}),u(vn)!=="svelte-hswkmf"&&(vn.innerHTML=Ua),xs=a(O),re=p(O,"DIV",{class:!0});var ge=j(re);f($t.$$.fragment,ge),Cs=a(ge),jn=p(ge,"P",{"data-svelte-h":!0}),u(jn)!=="svelte-1yi4t28"&&(jn.innerHTML=xa),$s=a(ge),f(Ee.$$.fragment,ge),Fs=a(ge),f(Ne.$$.fragment,ge),ge.forEach(l),O.forEach(l),lo=a(e),f(Ft.$$.fragment,e),co=a(e),zn=p(e,"P",{}),j(zn).forEach(l),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",rr),v(ye,"class","flex flex-wrap space-x-1"),v(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,r){o(document.head,t),m(e,b,r),m(e,i,r),m(e,d,r),m(e,T,r),m(e,M,r),g(Ve,e,r),m(e,Jn,r),m(e,ye,r),m(e,Un,r),g(Ge,e,r),m(e,xn,r),m(e,Xe,r),m(e,Cn,r),m(e,Se,r),m(e,$n,r),m(e,He,r),m(e,Fn,r),m(e,Qe,r),m(e,Wn,r),g(Ae,e,r),m(e,In,r),m(e,Pe,r),m(e,Zn,r),m(e,Ye,r),m(e,Bn,r),g(Oe,e,r),m(e,qn,r),g(De,e,r),m(e,Rn,r),m(e,Ke,r),m(e,En,r),g(et,e,r),m(e,Nn,r),m(e,N,r),g(tt,N,null),o(N,mo),o(N,It),o(N,uo),o(N,Zt),o(N,ho),g(ke,N,null),m(e,Vn,r),g(nt,e,r),m(e,Gn,r),m(e,z,r),g(ot,z,null),o(z,fo),o(z,Bt),o(z,go),o(z,qt),o(z,_o),g(be,z,null),o(z,yo),o(z,Rt),o(z,ko),g(Te,z,null),o(z,bo),o(z,Et),o(z,To),o(z,Me),g(st,Me,null),o(Me,Mo),o(Me,Nt),o(z,wo),o(z,Vt),g(at,Vt,null),m(e,Xn,r),g(rt,e,r),m(e,Sn,r),m(e,x,r),g(it,x,null),o(x,vo),o(x,Gt),o(x,jo),o(x,Xt),o(x,zo),o(x,St),o(x,Lo),o(x,K),g(lt,K,null),o(K,Jo),o(K,Ht),o(K,Uo),g(we,K,null),o(K,xo),g(ve,K,null),m(e,Hn,r),g(dt,e,r),m(e,Qn,r),m(e,C,r),g(ct,C,null),o(C,Co),o(C,Qt),o(C,$o),o(C,At),o(C,Fo),o(C,Pt),o(C,Wo),o(C,ee),g(pt,ee,null),o(ee,Io),o(ee,Yt),o(ee,Zo),g(je,ee,null),o(ee,Bo),g(ze,ee,null),m(e,An,r),g(mt,e,r),m(e,Pn,r),m(e,$,r),g(ut,$,null),o($,qo),o($,Ot),o($,Ro),o($,Dt),o($,Eo),o($,Kt),o($,No),o($,te),g(ht,te,null),o(te,Vo),o(te,en),o(te,Go),g(Le,te,null),o(te,Xo),g(Je,te,null),m(e,Yn,r),g(ft,e,r),m(e,On,r),m(e,F,r),g(gt,F,null),o(F,So),o(F,tn),o(F,Ho),o(F,nn),o(F,Qo),o(F,on),o(F,Ao),o(F,ne),g(_t,ne,null),o(ne,Po),o(ne,sn),o(ne,Yo),g(Ue,ne,null),o(ne,Oo),g(xe,ne,null),m(e,Dn,r),g(yt,e,r),m(e,Kn,r),m(e,W,r),g(kt,W,null),o(W,Do),o(W,an),o(W,Ko),o(W,rn),o(W,es),o(W,ln),o(W,ts),o(W,oe),g(bt,oe,null),o(oe,ns),o(oe,dn),o(oe,os),g(Ce,oe,null),o(oe,ss),g($e,oe,null),m(e,eo,r),g(Tt,e,r),m(e,to,r),m(e,I,r),g(Mt,I,null),o(I,as),o(I,cn),o(I,rs),o(I,pn),o(I,is),o(I,mn),o(I,ls),o(I,E),g(wt,E,null),o(E,ds),o(E,un),o(E,cs),g(Fe,E,null),o(E,ps),g(We,E,null),o(E,ms),g(Ie,E,null),m(e,no,r),g(vt,e,r),m(e,oo,r),m(e,Z,r),g(jt,Z,null),o(Z,us),o(Z,hn),o(Z,hs),o(Z,fn),o(Z,fs),o(Z,gn),o(Z,gs),o(Z,se),g(zt,se,null),o(se,_s),o(se,_n),o(se,ys),g(Ze,se,null),o(se,ks),g(Be,se,null),m(e,so,r),g(Lt,e,r),m(e,ao,r),m(e,B,r),g(Jt,B,null),o(B,bs),o(B,yn),o(B,Ts),o(B,kn),o(B,Ms),o(B,bn),o(B,ws),o(B,ae),g(Ut,ae,null),o(ae,vs),o(ae,Tn),o(ae,js),g(qe,ae,null),o(ae,zs),g(Re,ae,null),m(e,ro,r),g(xt,e,r),m(e,io,r),m(e,q,r),g(Ct,q,null),o(q,Ls),o(q,Mn),o(q,Js),o(q,wn),o(q,Us),o(q,vn),o(q,xs),o(q,re),g($t,re,null),o(re,Cs),o(re,jn),o(re,$s),g(Ee,re,null),o(re,Fs),g(Ne,re,null),m(e,lo,r),g(Ft,e,r),m(e,co,r),m(e,zn,r),po=!0},p(e,[r]){const le={};r&2&&(le.$$scope={dirty:r,ctx:e}),ke.$set(le);const U={};r&2&&(U.$$scope={dirty:r,ctx:e}),be.$set(U);const Wt={};r&2&&(Wt.$$scope={dirty:r,ctx:e}),Te.$set(Wt);const Ln={};r&2&&(Ln.$$scope={dirty:r,ctx:e}),we.$set(Ln);const V={};r&2&&(V.$$scope={dirty:r,ctx:e}),ve.$set(V);const de={};r&2&&(de.$$scope={dirty:r,ctx:e}),je.$set(de);const G={};r&2&&(G.$$scope={dirty:r,ctx:e}),ze.$set(G);const ce={};r&2&&(ce.$$scope={dirty:r,ctx:e}),Le.$set(ce);const X={};r&2&&(X.$$scope={dirty:r,ctx:e}),Je.$set(X);const pe={};r&2&&(pe.$$scope={dirty:r,ctx:e}),Ue.$set(pe);const S={};r&2&&(S.$$scope={dirty:r,ctx:e}),xe.$set(S);const me={};r&2&&(me.$$scope={dirty:r,ctx:e}),Ce.$set(me);const H={};r&2&&(H.$$scope={dirty:r,ctx:e}),$e.$set(H);const ue={};r&2&&(ue.$$scope={dirty:r,ctx:e}),Fe.$set(ue);const Q={};r&2&&(Q.$$scope={dirty:r,ctx:e}),We.$set(Q);const A={};r&2&&(A.$$scope={dirty:r,ctx:e}),Ie.$set(A);const P={};r&2&&(P.$$scope={dirty:r,ctx:e}),Ze.$set(P);const he={};r&2&&(he.$$scope={dirty:r,ctx:e}),Be.$set(he);const Y={};r&2&&(Y.$$scope={dirty:r,ctx:e}),qe.$set(Y);const fe={};r&2&&(fe.$$scope={dirty:r,ctx:e}),Re.$set(fe);const O={};r&2&&(O.$$scope={dirty:r,ctx:e}),Ee.$set(O);const ge={};r&2&&(ge.$$scope={dirty:r,ctx:e}),Ne.$set(ge)},i(e){po||(_(Ve.$$.fragment,e),_(Ge.$$.fragment,e),_(Ae.$$.fragment,e),_(Oe.$$.fragment,e),_(De.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(ke.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(be.$$.fragment,e),_(Te.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(we.$$.fragment,e),_(ve.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(je.$$.fragment,e),_(ze.$$.fragment,e),_(mt.$$.fragment,e),_(ut.$$.fragment,e),_(ht.$$.fragment,e),_(Le.$$.fragment,e),_(Je.$$.fragment,e),_(ft.$$.fragment,e),_(gt.$$.fragment,e),_(_t.$$.fragment,e),_(Ue.$$.fragment,e),_(xe.$$.fragment,e),_(yt.$$.fragment,e),_(kt.$$.fragment,e),_(bt.$$.fragment,e),_(Ce.$$.fragment,e),_($e.$$.fragment,e),_(Tt.$$.fragment,e),_(Mt.$$.fragment,e),_(wt.$$.fragment,e),_(Fe.$$.fragment,e),_(We.$$.fragment,e),_(Ie.$$.fragment,e),_(vt.$$.fragment,e),_(jt.$$.fragment,e),_(zt.$$.fragment,e),_(Ze.$$.fragment,e),_(Be.$$.fragment,e),_(Lt.$$.fragment,e),_(Jt.$$.fragment,e),_(Ut.$$.fragment,e),_(qe.$$.fragment,e),_(Re.$$.fragment,e),_(xt.$$.fragment,e),_(Ct.$$.fragment,e),_($t.$$.fragment,e),_(Ee.$$.fragment,e),_(Ne.$$.fragment,e),_(Ft.$$.fragment,e),po=!0)},o(e){y(Ve.$$.fragment,e),y(Ge.$$.fragment,e),y(Ae.$$.fragment,e),y(Oe.$$.fragment,e),y(De.$$.fragment,e),y(et.$$.fragment,e),y(tt.$$.fragment,e),y(ke.$$.fragment,e),y(nt.$$.fragment,e),y(ot.$$.fragment,e),y(be.$$.fragment,e),y(Te.$$.fragment,e),y(st.$$.fragment,e),y(at.$$.fragment,e),y(rt.$$.fragment,e),y(it.$$.fragment,e),y(lt.$$.fragment,e),y(we.$$.fragment,e),y(ve.$$.fragment,e),y(dt.$$.fragment,e),y(ct.$$.fragment,e),y(pt.$$.fragment,e),y(je.$$.fragment,e),y(ze.$$.fragment,e),y(mt.$$.fragment,e),y(ut.$$.fragment,e),y(ht.$$.fragment,e),y(Le.$$.fragment,e),y(Je.$$.fragment,e),y(ft.$$.fragment,e),y(gt.$$.fragment,e),y(_t.$$.fragment,e),y(Ue.$$.fragment,e),y(xe.$$.fragment,e),y(yt.$$.fragment,e),y(kt.$$.fragment,e),y(bt.$$.fragment,e),y(Ce.$$.fragment,e),y($e.$$.fragment,e),y(Tt.$$.fragment,e),y(Mt.$$.fragment,e),y(wt.$$.fragment,e),y(Fe.$$.fragment,e),y(We.$$.fragment,e),y(Ie.$$.fragment,e),y(vt.$$.fragment,e),y(jt.$$.fragment,e),y(zt.$$.fragment,e),y(Ze.$$.fragment,e),y(Be.$$.fragment,e),y(Lt.$$.fragment,e),y(Jt.$$.fragment,e),y(Ut.$$.fragment,e),y(qe.$$.fragment,e),y(Re.$$.fragment,e),y(xt.$$.fragment,e),y(Ct.$$.fragment,e),y($t.$$.fragment,e),y(Ee.$$.fragment,e),y(Ne.$$.fragment,e),y(Ft.$$.fragment,e),po=!1},d(e){e&&(l(b),l(i),l(d),l(T),l(M),l(Jn),l(ye),l(Un),l(xn),l(Xe),l(Cn),l(Se),l($n),l(He),l(Fn),l(Qe),l(Wn),l(In),l(Pe),l(Zn),l(Ye),l(Bn),l(qn),l(Rn),l(Ke),l(En),l(Nn),l(N),l(Vn),l(Gn),l(z),l(Xn),l(Sn),l(x),l(Hn),l(Qn),l(C),l(An),l(Pn),l($),l(Yn),l(On),l(F),l(Dn),l(Kn),l(W),l(eo),l(to),l(I),l(no),l(oo),l(Z),l(so),l(ao),l(B),l(ro),l(io),l(q),l(lo),l(co),l(zn)),l(t),k(Ve,e),k(Ge,e),k(Ae,e),k(Oe,e),k(De,e),k(et,e),k(tt),k(ke),k(nt,e),k(ot),k(be),k(Te),k(st),k(at),k(rt,e),k(it),k(lt),k(we),k(ve),k(dt,e),k(ct),k(pt),k(je),k(ze),k(mt,e),k(ut),k(ht),k(Le),k(Je),k(ft,e),k(gt),k(_t),k(Ue),k(xe),k(yt,e),k(kt),k(bt),k(Ce),k($e),k(Tt,e),k(Mt),k(wt),k(Fe),k(We),k(Ie),k(vt,e),k(jt),k(zt),k(Ze),k(Be),k(Lt,e),k(Jt),k(Ut),k(qe),k(Re),k(xt,e),k(Ct),k($t),k(Ee),k(Ne),k(Ft,e)}}}const rr='{"title":"LUKE","local":"luke","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"LukeConfig","local":"transformers.LukeConfig","sections":[],"depth":2},{"title":"LukeTokenizer","local":"transformers.LukeTokenizer","sections":[],"depth":2},{"title":"LukeModel","local":"transformers.LukeModel","sections":[],"depth":2},{"title":"LukeForMaskedLM","local":"transformers.LukeForMaskedLM","sections":[],"depth":2},{"title":"LukeForEntityClassification","local":"transformers.LukeForEntityClassification","sections":[],"depth":2},{"title":"LukeForEntityPairClassification","local":"transformers.LukeForEntityPairClassification","sections":[],"depth":2},{"title":"LukeForEntitySpanClassification","local":"transformers.LukeForEntitySpanClassification","sections":[],"depth":2},{"title":"LukeForSequenceClassification","local":"transformers.LukeForSequenceClassification","sections":[],"depth":2},{"title":"LukeForMultipleChoice","local":"transformers.LukeForMultipleChoice","sections":[],"depth":2},{"title":"LukeForTokenClassification","local":"transformers.LukeForTokenClassification","sections":[],"depth":2},{"title":"LukeForQuestionAnswering","local":"transformers.LukeForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function ir(w){return $a(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class fr extends Fa{constructor(t){super(),Wa(this,t,ir,ar,Ca,{})}}export{fr as component};
