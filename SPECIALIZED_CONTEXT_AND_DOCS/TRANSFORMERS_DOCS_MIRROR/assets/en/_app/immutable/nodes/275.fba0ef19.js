import{s as kn,o as wn,n as V}from"../chunks/scheduler.18a86fab.js";import{S as vn,i as $n,g as m,s as l,r as f,A as Jn,h as u,f as a,c as d,j as H,x as T,u as g,k as W,l as jn,y as h,a as i,v as _,d as M,t as y,w as b}from"../chunks/index.98837b22.js";import{T as wt}from"../chunks/Tip.77304350.js";import{D as L}from"../chunks/Docstring.a1ef7999.js";import{C as X}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Pe}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Ie,E as Un}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as zn,a as Tn}from"../chunks/HfOption.6641485e.js";function Cn(w){let t,p='This model was contributed by <a href="https://huggingface.co/sshleifer" rel="nofollow">sshleifer</a>.',s,r,c="Click on the MarianMT models in the right sidebar for more examples of how to apply MarianMT to translation tasks.";return{c(){t=m("p"),t.innerHTML=p,s=l(),r=m("p"),r.textContent=c},l(n){t=u(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-15213jf"&&(t.innerHTML=p),s=d(n),r=u(n,"P",{"data-svelte-h":!0}),T(r)!=="svelte-9ilby1"&&(r.textContent=c)},m(n,k){i(n,t,k),i(n,s,k),i(n,r,k)},p:V,d(n){n&&(a(t),a(s),a(r))}}}function xn(w){let t,p;return t=new X({props:{code:"JTBBaW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUyMnRyYW5zbGF0aW9uX2VuX3RvX2RlJTIyJTJDJTIwbW9kZWwlM0QlMjJIZWxzaW5raS1OTFAlMkZvcHVzLW10LWVuLWRlJTIyJTJDJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTIwZGV2aWNlJTNEMCklMEFwaXBlbGluZSglMjJIZWxsbyUyQyUyMGhvdyUyMGFyZSUyMHlvdSUzRiUyMiklMEE=",highlighted:`
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(<span class="hljs-string">&quot;translation_en_to_de&quot;</span>, model=<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-de&quot;</span>, dtype=torch.float16, device=<span class="hljs-number">0</span>)
pipeline(<span class="hljs-string">&quot;Hello, how are you?&quot;</span>)
`,wrap:!1}}),{c(){f(t.$$.fragment)},l(s){g(t.$$.fragment,s)},m(s,r){_(t,s,r),p=!0},p:V,i(s){p||(M(t.$$.fragment,s),p=!0)},o(s){y(t.$$.fragment,s),p=!1},d(s){b(t,s)}}}function In(w){let t,p;return t=new X({props:{code:"JTBBaW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkhlbHNpbmtpLU5MUCUyRm9wdXMtbXQtZW4tZGUlMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXEyU2VxTE0uZnJvbV9wcmV0cmFpbmVkKCUyMkhlbHNpbmtpLU5MUCUyRm9wdXMtbXQtZW4tZGUlMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUyQyUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMGhvdyUyMGFyZSUyMHlvdSUzRiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKSUwQQ==",highlighted:`
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-de&quot;</span>)
model = AutoModelForSeq2SeqLM.from_pretrained(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-de&quot;</span>, dtype=torch.float16, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>)

inputs = tokenizer(<span class="hljs-string">&quot;Hello, how are you?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model.generate(**inputs, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))
`,wrap:!1}}),{c(){f(t.$$.fragment)},l(s){g(t.$$.fragment,s)},m(s,r){_(t,s,r),p=!0},p:V,i(s){p||(M(t.$$.fragment,s),p=!0)},o(s){y(t.$$.fragment,s),p=!1},d(s){b(t,s)}}}function Zn(w){let t,p,s,r;return t=new Tn({props:{id:"usage",option:"Pipeline",$$slots:{default:[xn]},$$scope:{ctx:w}}}),s=new Tn({props:{id:"usage",option:"AutoModel",$$slots:{default:[In]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),p=l(),f(s.$$.fragment)},l(c){g(t.$$.fragment,c),p=d(c),g(s.$$.fragment,c)},m(c,n){_(t,c,n),i(c,p,n),_(s,c,n),r=!0},p(c,n){const k={};n&2&&(k.$$scope={dirty:n,ctx:c}),t.$set(k);const F={};n&2&&(F.$$scope={dirty:n,ctx:c}),s.$set(F)},i(c){r||(M(t.$$.fragment,c),M(s.$$.fragment,c),r=!0)},o(c){y(t.$$.fragment,c),y(s.$$.fragment,c),r=!1},d(c){c&&a(p),b(t,c),b(s,c)}}}function Wn(w){let t,p="Examples:",s,r,c;return r=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmlhbk1vZGVsJTJDJTIwTWFyaWFuQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyME1hcmlhbiUyMEhlbHNpbmtpLU5MUCUyRm9wdXMtbXQtZW4tZGUlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwTWFyaWFuQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwSGVsc2lua2ktTkxQJTJGb3B1cy1tdC1lbi1kZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwTWFyaWFuTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarianModel, MarianConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Marian Helsinki-NLP/opus-mt-en-de style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MarianConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Helsinki-NLP/opus-mt-en-de style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MarianModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,s=l(),f(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-kvfsh7"&&(t.textContent=p),s=d(n),g(r.$$.fragment,n)},m(n,k){i(n,t,k),i(n,s,k),_(r,n,k),c=!0},p:V,i(n){c||(M(r.$$.fragment,n),c=!0)},o(n){y(r.$$.fragment,n),c=!1},d(n){n&&(a(t),a(s)),b(r,n)}}}function Fn(w){let t,p="Examples:",s,r,c;return r=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmlhbkZvckNhdXNhbExNJTJDJTIwTWFyaWFuVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBNYXJpYW5Gb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIySGVsc2lua2ktTkxQJTJGb3B1cy1tdC1lbi1kZSUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBNYXJpYW5Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkhlbHNpbmtpLU5MUCUyRm9wdXMtbXQtZW4tZGUlMjIpJTBBc3JjX3RleHRzJTIwJTNEJTIwJTVCJTIySSUyMGFtJTIwYSUyMHNtYWxsJTIwZnJvZy4lMjIlMkMlMjAlMjJUb20lMjBhc2tlZCUyMGhpcyUyMHRlYWNoZXIlMjBmb3IlMjBhZHZpY2UuJTIyJTVEJTBBdGd0X3RleHRzJTIwJTNEJTIwJTVCJTIySWNoJTIwYmluJTIwZWluJTIwa2xlaW5lciUyMEZyb3NjaC4lMjIlMkMlMjAlMjJUb20lMjBiYXQlMjBzZWluZW4lMjBMZWhyZXIlMjB1bSUyMFJhdC4lMjIlNUQlMjAlMjAlMjMlMjBvcHRpb25hbCUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihzcmNfdGV4dHMlMkMlMjB0ZXh0X3RhcmdldCUzRHRndF90ZXh0cyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTIwJTIwJTIzJTIwc2hvdWxkJTIwd29yaw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarianForCausalLM, MarianTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MarianForCausalLM.from_pretrained(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-de&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MarianTokenizer.from_pretrained(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-de&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>src_texts = [<span class="hljs-string">&quot;I am a small frog.&quot;</span>, <span class="hljs-string">&quot;Tom asked his teacher for advice.&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>tgt_texts = [<span class="hljs-string">&quot;Ich bin ein kleiner Frosch.&quot;</span>, <span class="hljs-string">&quot;Tom bat seinen Lehrer um Rat.&quot;</span>]  <span class="hljs-comment"># optional</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(src_texts, text_target=tgt_texts, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)  <span class="hljs-comment"># should work</span>`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,s=l(),f(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-kvfsh7"&&(t.textContent=p),s=d(n),g(r.$$.fragment,n)},m(n,k){i(n,t,k),i(n,s,k),_(r,n,k),c=!0},p:V,i(n){c||(M(r.$$.fragment,n),c=!0)},o(n){y(r.$$.fragment,n),c=!1},d(n){n&&(a(t),a(s)),b(r,n)}}}function Nn(w){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(s){t=u(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(s,r){i(s,t,r)},p:V,d(s){s&&a(t)}}}function qn(w){let t,p="Example:",s,r,c;return r=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNYXJpYW5Nb2RlbCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkhlbHNpbmtpLU5MUCUyRm9wdXMtbXQtZW4tZGUlMjIpJTBBbW9kZWwlMjAlM0QlMjBNYXJpYW5Nb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIySGVsc2lua2ktTkxQJTJGb3B1cy1tdC1lbi1kZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyU3R1ZGllcyUyMGhhdmUlMjBiZWVuJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBZGVjb2Rlcl9pbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIyJTNDcGFkJTNFJTIwU3R1ZGllbiUyMGhhYmVuJTIwZ2V6ZWlndCUyMGRhc3MlMjBlcyUyMGhpbGZyZWljaCUyMGlzdCUyMGVpbmVuJTIwSHVuZCUyMHp1JTIwYmVzaXR6ZW4lMjIlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBJTIwJTIwJTIwJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMEEpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUzRGlucHV0cy5pbnB1dF9pZHMlMkMlMjBkZWNvZGVyX2lucHV0X2lkcyUzRGRlY29kZXJfaW5wdXRzLmlucHV0X2lkcyklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBbGlzdChsYXN0X2hpZGRlbl9zdGF0ZXMuc2hhcGUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MarianModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-de&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MarianModel.from_pretrained(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-de&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;&lt;pad&gt; Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen&quot;</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>    add_special_tokens=<span class="hljs-literal">False</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">26</span>, <span class="hljs-number">512</span>]`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,s=l(),f(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=p),s=d(n),g(r.$$.fragment,n)},m(n,k){i(n,t,k),i(n,s,k),_(r,n,k),c=!0},p:V,i(n){c||(M(r.$$.fragment,n),c=!0)},o(n){y(r.$$.fragment,n),c=!1},d(n){n&&(a(t),a(s)),b(r,n)}}}function Bn(w){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(s){t=u(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(s,r){i(s,t,r)},p:V,d(s){s&&a(t)}}}function Gn(w){let t,p="Example:",s,r,c;return r=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNYXJpYW5NVE1vZGVsJTBBJTBBc3JjJTIwJTNEJTIwJTIyZnIlMjIlMjAlMjAlMjMlMjBzb3VyY2UlMjBsYW5ndWFnZSUwQXRyZyUyMCUzRCUyMCUyMmVuJTIyJTIwJTIwJTIzJTIwdGFyZ2V0JTIwbGFuZ3VhZ2UlMEElMEFtb2RlbF9uYW1lJTIwJTNEJTIwZiUyMkhlbHNpbmtpLU5MUCUyRm9wdXMtbXQtJTdCc3JjJTdELSU3QnRyZyU3RCUyMiUwQW1vZGVsJTIwJTNEJTIwTWFyaWFuTVRNb2RlbC5mcm9tX3ByZXRyYWluZWQobW9kZWxfbmFtZSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQSUwQXNhbXBsZV90ZXh0JTIwJTNEJTIwJTIybyVDMyVCOSUyMGVzdCUyMGwnYXJyJUMzJUFBdCUyMGRlJTIwYnVzJTIwJTNGJTIyJTBBYmF0Y2glMjAlM0QlMjB0b2tlbml6ZXIoJTVCc2FtcGxlX3RleHQlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmJhdGNoKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MarianMTModel

<span class="hljs-meta">&gt;&gt;&gt; </span>src = <span class="hljs-string">&quot;fr&quot;</span>  <span class="hljs-comment"># source language</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>trg = <span class="hljs-string">&quot;en&quot;</span>  <span class="hljs-comment"># target language</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>model_name = <span class="hljs-string">f&quot;Helsinki-NLP/opus-mt-<span class="hljs-subst">{src}</span>-<span class="hljs-subst">{trg}</span>&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MarianMTModel.from_pretrained(model_name)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(model_name)

<span class="hljs-meta">&gt;&gt;&gt; </span>sample_text = <span class="hljs-string">&quot;où est l&#x27;arrêt de bus ?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>batch = tokenizer([sample_text], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(**batch)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Where&#x27;s the bus stop?&quot;</span>`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,s=l(),f(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=p),s=d(n),g(r.$$.fragment,n)},m(n,k){i(n,t,k),i(n,s,k),_(r,n,k),c=!0},p:V,i(n){c||(M(r.$$.fragment,n),c=!0)},o(n){y(r.$$.fragment,n),c=!1},d(n){n&&(a(t),a(s)),b(r,n)}}}function Hn(w){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(s){t=u(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(s,r){i(s,t,r)},p:V,d(s){s&&a(t)}}}function Vn(w){let t,p="Example:",s,r,c;return r=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNYXJpYW5Gb3JDYXVzYWxMTSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkhlbHNpbmtpLU5MUCUyRm9wdXMtbXQtZnItZW4lMjIpJTBBbW9kZWwlMjAlM0QlMjBNYXJpYW5Gb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIySGVsc2lua2ktTkxQJTJGb3B1cy1tdC1mci1lbiUyMiUyQyUyMGFkZF9jcm9zc19hdHRlbnRpb24lM0RGYWxzZSklMEFhc3NlcnQlMjBtb2RlbC5jb25maWcuaXNfZGVjb2RlciUyQyUyMGYlMjIlN0Jtb2RlbC5fX2NsYXNzX18lN0QlMjBoYXMlMjB0byUyMGJlJTIwY29uZmlndXJlZCUyMGFzJTIwYSUyMGRlY29kZXIuJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQWV4cGVjdGVkX3NoYXBlJTIwJTNEJTIwJTVCMSUyQyUyMGlucHV0cy5pbnB1dF9pZHMuc2hhcGUlNUItMSU1RCUyQyUyMG1vZGVsLmNvbmZpZy52b2NhYl9zaXplJTVEJTBBbGlzdChsb2dpdHMuc2hhcGUpJTIwJTNEJTNEJTIwZXhwZWN0ZWRfc2hhcGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MarianForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-fr-en&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MarianForCausalLM.from_pretrained(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-fr-en&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_shape = [<span class="hljs-number">1</span>, inputs.input_ids.shape[-<span class="hljs-number">1</span>], model.config.vocab_size]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape) == expected_shape
<span class="hljs-literal">True</span>`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,s=l(),f(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=p),s=d(n),g(r.$$.fragment,n)},m(n,k){i(n,t,k),i(n,s,k),_(r,n,k),c=!0},p:V,i(n){c||(M(r.$$.fragment,n),c=!0)},o(n){y(r.$$.fragment,n),c=!1},d(n){n&&(a(t),a(s)),b(r,n)}}}function Xn(w){let t,p,s,r,c,n="<em>This model was released on 2018-04-01 and added to Hugging Face Transformers on 2020-11-16.</em>",k,F,At='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Qe,oe,Ae,se,Ot='<a href="https://huggingface.co/papers/1804.00344" rel="nofollow">MarianMT</a> is a machine translation model trained with the Marian framework which is written in pure C++. The framework includes its own custom auto-differentiation engine and efficient meta-algorithms to train encoder-decoder models like BART.',Oe,ae,Dt="All MarianMT models are transformer encoder-decoders with 6 layers in each component, use static sinusoidal positional embeddings, don’t have a layernorm embedding, and the model starts generating with the prefix <code>pad_token_id</code> instead of <code>&lt;s/&gt;</code>.",De,re,Kt='You can find all the original MarianMT checkpoints under the <a href="https://huggingface.co/Helsinki-NLP/models?search=opus-mt" rel="nofollow">Language Technology Research Group at the University of Helsinki</a> organization.',Ke,E,et,ie,en='The example below demonstrates how to translate text using <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',tt,S,nt,le,tn='Use the <a href="https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139" rel="nofollow">AttentionMaskVisualizer</a> to better understand what tokens the model can and cannot attend to.',ot,de,st,Y,nn='<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/marianmt-attn-mask.png"/>',at,ce,rt,pe,on='<li>MarianMT models are ~298MB on disk and there are more than 1000 models. Check this <a href="https://huggingface.co/Helsinki-NLP" rel="nofollow">list</a> for supported language pairs. The language codes may be inconsistent. Two digit codes can be found <a href="https://developers.google.com/admin-sdk/directory/v1/languages" rel="nofollow">here</a> while three digit codes may require further searching.</li> <li>Models that require BPE preprocessing are not supported.</li> <li>All model names use the following format: <code>Helsinki-NLP/opus-mt-{src}-{tgt}</code>. Language codes formatted like <code>es_AR</code> usually refer to the <code>code_{region}</code>. For example, <code>es_AR</code> refers to Spanish from Argentina.</li> <li>If a model can output multiple languages, prepend the desired output language to <code>src_txt</code> as shown below. New multilingual models from the <a href="https://github.com/Helsinki-NLP/Tatoeba-Challenge" rel="nofollow">Tatoeba-Challenge</a> require 3 character language codes.</li>',it,me,lt,ue,sn="<li>Older multilingual models use 2 character language codes.</li>",dt,he,ct,fe,pt,j,ge,vt,Ze,an=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianModel">MarianModel</a>. It is used to instantiate an
Marian model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Marian
<a href="https://huggingface.co/Helsinki-NLP/opus-mt-en-de" rel="nofollow">Helsinki-NLP/opus-mt-en-de</a> architecture.`,$t,We,rn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Jt,P,mt,_e,ut,v,Me,jt,Fe,ln='Construct a Marian tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',Ut,Ne,dn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,zt,Q,Ct,A,ye,xt,qe,cn="Build model inputs from a sequence by appending eos_token_id.",ht,be,ft,$,Te,It,Be,pn="The bare Marian Model outputting raw hidden-states without any specific head on top.",Zt,Ge,mn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Wt,He,un=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ft,x,ke,Nt,Ve,hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianModel">MarianModel</a> forward method, overrides the <code>__call__</code> special method.',qt,O,Bt,D,gt,we,_t,J,ve,Gt,Xe,fn="The Marian Model with a language modeling head. Can be used for summarization.",Ht,Re,gn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Vt,Le,_n=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xt,I,$e,Rt,Ee,Mn='The <a href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianMTModel">MarianMTModel</a> forward method, overrides the <code>__call__</code> special method.',Lt,K,Et,ee,Mt,Je,yt,R,je,St,Z,Ue,Yt,Se,yn='The <a href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianForCausalLM">MarianForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Pt,te,Qt,ne,bt,ze,Tt,Ye,kt;return oe=new Ie({props:{title:"MarianMT",local:"marianmt",headingTag:"h1"}}),E=new wt({props:{warning:!1,$$slots:{default:[Cn]},$$scope:{ctx:w}}}),S=new zn({props:{id:"usage",options:["Pipeline","AutoModel"],$$slots:{default:[Zn]},$$scope:{ctx:w}}}),de=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycy51dGlscy5hdHRlbnRpb25fdmlzdWFsaXplciUyMGltcG9ydCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyJTBBJTBBdmlzdWFsaXplciUyMCUzRCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyKCUyMkhlbHNpbmtpLU5MUCUyRm9wdXMtbXQtZW4tZGUlMjIpJTBBdmlzdWFsaXplciglMjJIZWxsbyUyQyUyMGhvdyUyMGFyZSUyMHlvdSUzRiUyMik=",highlighted:`<span class="hljs-keyword">from</span> transformers.utils.attention_visualizer <span class="hljs-keyword">import</span> AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer(<span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-de&quot;</span>)
visualizer(<span class="hljs-string">&quot;Hello, how are you?&quot;</span>)`,wrap:!1}}),ce=new Ie({props:{title:"Notes",local:"notes",headingTag:"h2"}}),me=new X({props:{code:"JTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmlhbk1UTW9kZWwlMkMlMjBNYXJpYW5Ub2tlbml6ZXIlMEElMEElMjMlMjBNb2RlbCUyMHRyYWluZWQlMjBvbiUyMG11bHRpcGxlJTIwc291cmNlJTIwbGFuZ3VhZ2VzJTIwJUUyJTg2JTkyJTIwbXVsdGlwbGUlMjB0YXJnZXQlMjBsYW5ndWFnZXMlMEElMjMlMjBFeGFtcGxlJTNBJTIwbXVsdGlsaW5ndWFsJTIwdG8lMjBBcmFiaWMlMjAoYXJiKSUwQW1vZGVsX25hbWUlMjAlM0QlMjAlMjJIZWxzaW5raS1OTFAlMkZvcHVzLW10LW11bC1tdWwlMjIlMjAlMjAlMjMlMjBUYXRvZWJhJTIwQ2hhbGxlbmdlJTIwbW9kZWwlMEF0b2tlbml6ZXIlMjAlM0QlMjBNYXJpYW5Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX25hbWUpJTBBbW9kZWwlMjAlM0QlMjBNYXJpYW5NVE1vZGVsLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQSUwQSUyMyUyMFByZXBlbmQlMjB0aGUlMjBkZXNpcmVkJTIwb3V0cHV0JTIwbGFuZ3VhZ2UlMjBjb2RlJTIwKDMtbGV0dGVyJTIwSVNPJTIwNjM5LTMpJTBBc3JjX3RleHRzJTIwJTNEJTIwJTVCJTIyYXJiJTNFJTNFJTIwSGVsbG8lMkMlMjBob3clMjBhcmUlMjB5b3UlMjB0b2RheSUzRiUyMiU1RCUwQSUwQSUyMyUyMFRva2VuaXplJTIwYW5kJTIwdHJhbnNsYXRlJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHNyY190ZXh0cyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSUyQyUyMHRydW5jYXRpb24lM0RUcnVlKSUwQXRyYW5zbGF0ZWQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyklMEElMEElMjMlMjBEZWNvZGUlMjBhbmQlMjBwcmludCUyMHJlc3VsdCUwQXRyYW5zbGF0ZWRfdGV4dHMlMjAlM0QlMjB0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKHRyYW5zbGF0ZWQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEFwcmludCh0cmFuc2xhdGVkX3RleHRzJTVCMCU1RCklMEE=",highlighted:`
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarianMTModel, MarianTokenizer

<span class="hljs-comment"># Model trained on multiple source languages → multiple target languages</span>
<span class="hljs-comment"># Example: multilingual to Arabic (arb)</span>
model_name = <span class="hljs-string">&quot;Helsinki-NLP/opus-mt-mul-mul&quot;</span>  <span class="hljs-comment"># Tatoeba Challenge model</span>
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

<span class="hljs-comment"># Prepend the desired output language code (3-letter ISO 639-3)</span>
src_texts = [<span class="hljs-string">&quot;arb&gt;&gt; Hello, how are you today?&quot;</span>]

<span class="hljs-comment"># Tokenize and translate</span>
inputs = tokenizer(src_texts, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>, truncation=<span class="hljs-literal">True</span>)
translated = model.generate(**inputs)

<span class="hljs-comment"># Decode and print result</span>
translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(translated_texts[<span class="hljs-number">0</span>])
`,wrap:!1}}),he=new X({props:{code:"JTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmlhbk1UTW9kZWwlMkMlMjBNYXJpYW5Ub2tlbml6ZXIlMEElMEElMjMlMjBFeGFtcGxlJTNBJTIwb2xkZXIlMjBtdWx0aWxpbmd1YWwlMjBtb2RlbCUyMChsaWtlJTIwZW4lMjAlRTIlODYlOTIlMjBtYW55KSUwQW1vZGVsX25hbWUlMjAlM0QlMjAlMjJIZWxzaW5raS1OTFAlMkZvcHVzLW10LWVuLVJPTUFOQ0UlMjIlMjAlMjAlMjMlMjBFbmdsaXNoJTIwJUUyJTg2JTkyJTIwRnJlbmNoJTJDJTIwU3BhbmlzaCUyQyUyMEl0YWxpYW4lMkMlMjBldGMuJTBBdG9rZW5pemVyJTIwJTNEJTIwTWFyaWFuVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQW1vZGVsJTIwJTNEJTIwTWFyaWFuTVRNb2RlbC5mcm9tX3ByZXRyYWluZWQobW9kZWxfbmFtZSklMEElMEElMjMlMjBQcmVwZW5kJTIwdGhlJTIwMi1sZXR0ZXIlMjBJU08lMjA2MzktMSUyMHRhcmdldCUyMGxhbmd1YWdlJTIwY29kZSUyMChvbGRlciUyMGZvcm1hdCklMEFzcmNfdGV4dHMlMjAlM0QlMjAlNUIlMjIlM0UlM0VmciUzQyUzQyUyMEhlbGxvJTJDJTIwaG93JTIwYXJlJTIweW91JTIwdG9kYXklM0YlMjIlNUQlMEElMEElMjMlMjBUb2tlbml6ZSUyMGFuZCUyMHRyYW5zbGF0ZSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihzcmNfdGV4dHMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUlMkMlMjB0cnVuY2F0aW9uJTNEVHJ1ZSklMEF0cmFuc2xhdGVkJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMpJTBBJTBBJTIzJTIwRGVjb2RlJTIwYW5kJTIwcHJpbnQlMjByZXN1bHQlMEF0cmFuc2xhdGVkX3RleHRzJTIwJTNEJTIwdG9rZW5pemVyLmJhdGNoX2RlY29kZSh0cmFuc2xhdGVkJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBcHJpbnQodHJhbnNsYXRlZF90ZXh0cyU1QjAlNUQpJTBB",highlighted:`
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarianMTModel, MarianTokenizer

<span class="hljs-comment"># Example: older multilingual model (like en → many)</span>
model_name = <span class="hljs-string">&quot;Helsinki-NLP/opus-mt-en-ROMANCE&quot;</span>  <span class="hljs-comment"># English → French, Spanish, Italian, etc.</span>
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

<span class="hljs-comment"># Prepend the 2-letter ISO 639-1 target language code (older format)</span>
src_texts = [<span class="hljs-string">&quot;&gt;&gt;fr&lt;&lt; Hello, how are you today?&quot;</span>]

<span class="hljs-comment"># Tokenize and translate</span>
inputs = tokenizer(src_texts, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>, truncation=<span class="hljs-literal">True</span>)
translated = model.generate(**inputs)

<span class="hljs-comment"># Decode and print result</span>
translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(translated_texts[<span class="hljs-number">0</span>])
`,wrap:!1}}),fe=new Ie({props:{title:"MarianConfig",local:"transformers.MarianConfig",headingTag:"h2"}}),ge=new L({props:{name:"class transformers.MarianConfig",anchor:"transformers.MarianConfig",parameters:[{name:"vocab_size",val:" = 58101"},{name:"decoder_vocab_size",val:" = None"},{name:"max_position_embeddings",val:" = 1024"},{name:"encoder_layers",val:" = 12"},{name:"encoder_ffn_dim",val:" = 4096"},{name:"encoder_attention_heads",val:" = 16"},{name:"decoder_layers",val:" = 12"},{name:"decoder_ffn_dim",val:" = 4096"},{name:"decoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 1024"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 58100"},{name:"scale_embedding",val:" = False"},{name:"pad_token_id",val:" = 58100"},{name:"eos_token_id",val:" = 0"},{name:"forced_eos_token_id",val:" = 0"},{name:"share_encoder_decoder_embeddings",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MarianConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 58101) &#x2014;
Vocabulary size of the Marian model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianModel">MarianModel</a> or <code>TFMarianModel</code>.`,name:"vocab_size"},{anchor:"transformers.MarianConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.MarianConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.MarianConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.MarianConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.MarianConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.MarianConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.MarianConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.MarianConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.MarianConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.MarianConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.MarianConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.MarianConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.MarianConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.MarianConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.MarianConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.MarianConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.MarianConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.MarianConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/configuration_marian.py#L31"}}),P=new Pe({props:{anchor:"transformers.MarianConfig.example",$$slots:{default:[Wn]},$$scope:{ctx:w}}}),_e=new Ie({props:{title:"MarianTokenizer",local:"transformers.MarianTokenizer",headingTag:"h2"}}),Me=new L({props:{name:"class transformers.MarianTokenizer",anchor:"transformers.MarianTokenizer",parameters:[{name:"source_spm",val:""},{name:"target_spm",val:""},{name:"vocab",val:""},{name:"target_vocab_file",val:" = None"},{name:"source_lang",val:" = None"},{name:"target_lang",val:" = None"},{name:"unk_token",val:" = '<unk>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"model_max_length",val:" = 512"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"separate_vocabs",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MarianTokenizer.source_spm",description:`<strong>source_spm</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a .spm extension) that
contains the vocabulary for the source language.`,name:"source_spm"},{anchor:"transformers.MarianTokenizer.target_spm",description:`<strong>target_spm</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a .spm extension) that
contains the vocabulary for the target language.`,name:"target_spm"},{anchor:"transformers.MarianTokenizer.source_lang",description:`<strong>source_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the source language.`,name:"source_lang"},{anchor:"transformers.MarianTokenizer.target_lang",description:`<strong>target_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the target language.`,name:"target_lang"},{anchor:"transformers.MarianTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MarianTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.MarianTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MarianTokenizer.model_max_length",description:`<strong>model_max_length</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sentence length the model accepts.`,name:"model_max_length"},{anchor:"transformers.MarianTokenizer.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&quot;&lt;eop&gt;&quot;, &quot;&lt;eod&gt;&quot;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"},{anchor:"transformers.MarianTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/tokenization_marian.py#L45"}}),Q=new Pe({props:{anchor:"transformers.MarianTokenizer.example",$$slots:{default:[Fn]},$$scope:{ctx:w}}}),ye=new L({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MarianTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/tokenization_marian.py#L267"}}),be=new Ie({props:{title:"MarianModel",local:"transformers.MarianModel",headingTag:"h2"}}),Te=new L({props:{name:"class transformers.MarianModel",anchor:"transformers.MarianModel",parameters:[{name:"config",val:": MarianConfig"}],parametersDescription:[{anchor:"transformers.MarianModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig">MarianConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/modeling_marian.py#L1111"}}),ke=new L({props:{name:"forward",anchor:"transformers.MarianModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[tuple[torch.Tensor], transformers.modeling_outputs.BaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MarianModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MarianModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MarianModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Marian uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.MarianModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MarianModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MarianModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MarianModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MarianModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>Union[tuple[torch.Tensor], ~modeling_outputs.BaseModelOutput, NoneType]</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MarianModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MarianModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MarianModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MarianModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MarianModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MarianModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MarianModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MarianModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/modeling_marian.py#L1192",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig"
>MarianConfig</a>) and inputs.</p>
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
`}}),O=new wt({props:{$$slots:{default:[Nn]},$$scope:{ctx:w}}}),D=new Pe({props:{anchor:"transformers.MarianModel.forward.example",$$slots:{default:[qn]},$$scope:{ctx:w}}}),we=new Ie({props:{title:"MarianMTModel",local:"transformers.MarianMTModel",headingTag:"h2"}}),ve=new L({props:{name:"class transformers.MarianMTModel",anchor:"transformers.MarianMTModel",parameters:[{name:"config",val:": MarianConfig"}],parametersDescription:[{anchor:"transformers.MarianMTModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig">MarianConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/modeling_marian.py#L1316"}}),$e=new L({props:{name:"forward",anchor:"transformers.MarianMTModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Union[tuple[torch.Tensor], transformers.modeling_outputs.BaseModelOutput, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MarianMTModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MarianMTModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MarianMTModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Marian uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.MarianMTModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MarianMTModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MarianMTModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MarianMTModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MarianMTModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>Union[tuple[torch.Tensor], ~modeling_outputs.BaseModelOutput, NoneType]</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MarianMTModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MarianMTModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MarianMTModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MarianMTModel.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MarianMTModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MarianMTModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MarianMTModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MarianMTModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MarianMTModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/modeling_marian.py#L1446",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig"
>MarianConfig</a>) and inputs.</p>
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
`}}),K=new wt({props:{$$slots:{default:[Bn]},$$scope:{ctx:w}}}),ee=new Pe({props:{anchor:"transformers.MarianMTModel.forward.example",$$slots:{default:[Gn]},$$scope:{ctx:w}}}),Je=new Ie({props:{title:"MarianForCausalLM",local:"transformers.MarianForCausalLM",headingTag:"h2"}}),je=new L({props:{name:"class transformers.MarianForCausalLM",anchor:"transformers.MarianForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/modeling_marian.py#L1585"}}),Ue=new L({props:{name:"forward",anchor:"transformers.MarianForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.MarianForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MarianForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MarianForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MarianForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.MarianForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MarianForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MarianForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MarianForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MarianForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MarianForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MarianForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MarianForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MarianForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MarianForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/marian/modeling_marian.py#L1611",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/marian#transformers.MarianConfig"
>MarianConfig</a>) and inputs.</p>
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
`}}),te=new wt({props:{$$slots:{default:[Hn]},$$scope:{ctx:w}}}),ne=new Pe({props:{anchor:"transformers.MarianForCausalLM.forward.example",$$slots:{default:[Vn]},$$scope:{ctx:w}}}),ze=new Un({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/marian.md"}}),{c(){t=m("meta"),p=l(),s=m("p"),r=l(),c=m("p"),c.innerHTML=n,k=l(),F=m("div"),F.innerHTML=At,Qe=l(),f(oe.$$.fragment),Ae=l(),se=m("p"),se.innerHTML=Ot,Oe=l(),ae=m("p"),ae.innerHTML=Dt,De=l(),re=m("p"),re.innerHTML=Kt,Ke=l(),f(E.$$.fragment),et=l(),ie=m("p"),ie.innerHTML=en,tt=l(),f(S.$$.fragment),nt=l(),le=m("p"),le.innerHTML=tn,ot=l(),f(de.$$.fragment),st=l(),Y=m("div"),Y.innerHTML=nn,at=l(),f(ce.$$.fragment),rt=l(),pe=m("ul"),pe.innerHTML=on,it=l(),f(me.$$.fragment),lt=l(),ue=m("ul"),ue.innerHTML=sn,dt=l(),f(he.$$.fragment),ct=l(),f(fe.$$.fragment),pt=l(),j=m("div"),f(ge.$$.fragment),vt=l(),Ze=m("p"),Ze.innerHTML=an,$t=l(),We=m("p"),We.innerHTML=rn,Jt=l(),f(P.$$.fragment),mt=l(),f(_e.$$.fragment),ut=l(),v=m("div"),f(Me.$$.fragment),jt=l(),Fe=m("p"),Fe.innerHTML=ln,Ut=l(),Ne=m("p"),Ne.innerHTML=dn,zt=l(),f(Q.$$.fragment),Ct=l(),A=m("div"),f(ye.$$.fragment),xt=l(),qe=m("p"),qe.textContent=cn,ht=l(),f(be.$$.fragment),ft=l(),$=m("div"),f(Te.$$.fragment),It=l(),Be=m("p"),Be.textContent=pn,Zt=l(),Ge=m("p"),Ge.innerHTML=mn,Wt=l(),He=m("p"),He.innerHTML=un,Ft=l(),x=m("div"),f(ke.$$.fragment),Nt=l(),Ve=m("p"),Ve.innerHTML=hn,qt=l(),f(O.$$.fragment),Bt=l(),f(D.$$.fragment),gt=l(),f(we.$$.fragment),_t=l(),J=m("div"),f(ve.$$.fragment),Gt=l(),Xe=m("p"),Xe.textContent=fn,Ht=l(),Re=m("p"),Re.innerHTML=gn,Vt=l(),Le=m("p"),Le.innerHTML=_n,Xt=l(),I=m("div"),f($e.$$.fragment),Rt=l(),Ee=m("p"),Ee.innerHTML=Mn,Lt=l(),f(K.$$.fragment),Et=l(),f(ee.$$.fragment),Mt=l(),f(Je.$$.fragment),yt=l(),R=m("div"),f(je.$$.fragment),St=l(),Z=m("div"),f(Ue.$$.fragment),Yt=l(),Se=m("p"),Se.innerHTML=yn,Pt=l(),f(te.$$.fragment),Qt=l(),f(ne.$$.fragment),bt=l(),f(ze.$$.fragment),Tt=l(),Ye=m("p"),this.h()},l(e){const o=Jn("svelte-u9bgzb",document.head);t=u(o,"META",{name:!0,content:!0}),o.forEach(a),p=d(e),s=u(e,"P",{}),H(s).forEach(a),r=d(e),c=u(e,"P",{"data-svelte-h":!0}),T(c)!=="svelte-gf05rt"&&(c.innerHTML=n),k=d(e),F=u(e,"DIV",{style:!0,"data-svelte-h":!0}),T(F)!=="svelte-2m0t7r"&&(F.innerHTML=At),Qe=d(e),g(oe.$$.fragment,e),Ae=d(e),se=u(e,"P",{"data-svelte-h":!0}),T(se)!=="svelte-1yf1lnk"&&(se.innerHTML=Ot),Oe=d(e),ae=u(e,"P",{"data-svelte-h":!0}),T(ae)!=="svelte-1uu9w4l"&&(ae.innerHTML=Dt),De=d(e),re=u(e,"P",{"data-svelte-h":!0}),T(re)!=="svelte-ges01k"&&(re.innerHTML=Kt),Ke=d(e),g(E.$$.fragment,e),et=d(e),ie=u(e,"P",{"data-svelte-h":!0}),T(ie)!=="svelte-117tn4x"&&(ie.innerHTML=en),tt=d(e),g(S.$$.fragment,e),nt=d(e),le=u(e,"P",{"data-svelte-h":!0}),T(le)!=="svelte-w3z5ks"&&(le.innerHTML=tn),ot=d(e),g(de.$$.fragment,e),st=d(e),Y=u(e,"DIV",{class:!0,"data-svelte-h":!0}),T(Y)!=="svelte-ao08o7"&&(Y.innerHTML=nn),at=d(e),g(ce.$$.fragment,e),rt=d(e),pe=u(e,"UL",{"data-svelte-h":!0}),T(pe)!=="svelte-hawak0"&&(pe.innerHTML=on),it=d(e),g(me.$$.fragment,e),lt=d(e),ue=u(e,"UL",{"data-svelte-h":!0}),T(ue)!=="svelte-15k2x62"&&(ue.innerHTML=sn),dt=d(e),g(he.$$.fragment,e),ct=d(e),g(fe.$$.fragment,e),pt=d(e),j=u(e,"DIV",{class:!0});var N=H(j);g(ge.$$.fragment,N),vt=d(N),Ze=u(N,"P",{"data-svelte-h":!0}),T(Ze)!=="svelte-ytwbmp"&&(Ze.innerHTML=an),$t=d(N),We=u(N,"P",{"data-svelte-h":!0}),T(We)!=="svelte-1ek1ss9"&&(We.innerHTML=rn),Jt=d(N),g(P.$$.fragment,N),N.forEach(a),mt=d(e),g(_e.$$.fragment,e),ut=d(e),v=u(e,"DIV",{class:!0});var U=H(v);g(Me.$$.fragment,U),jt=d(U),Fe=u(U,"P",{"data-svelte-h":!0}),T(Fe)!=="svelte-1giw7lm"&&(Fe.innerHTML=ln),Ut=d(U),Ne=u(U,"P",{"data-svelte-h":!0}),T(Ne)!=="svelte-ntrhio"&&(Ne.innerHTML=dn),zt=d(U),g(Q.$$.fragment,U),Ct=d(U),A=u(U,"DIV",{class:!0});var Ce=H(A);g(ye.$$.fragment,Ce),xt=d(Ce),qe=u(Ce,"P",{"data-svelte-h":!0}),T(qe)!=="svelte-wv4s2m"&&(qe.textContent=cn),Ce.forEach(a),U.forEach(a),ht=d(e),g(be.$$.fragment,e),ft=d(e),$=u(e,"DIV",{class:!0});var z=H($);g(Te.$$.fragment,z),It=d(z),Be=u(z,"P",{"data-svelte-h":!0}),T(Be)!=="svelte-183di38"&&(Be.textContent=pn),Zt=d(z),Ge=u(z,"P",{"data-svelte-h":!0}),T(Ge)!=="svelte-q52n56"&&(Ge.innerHTML=mn),Wt=d(z),He=u(z,"P",{"data-svelte-h":!0}),T(He)!=="svelte-hswkmf"&&(He.innerHTML=un),Ft=d(z),x=u(z,"DIV",{class:!0});var q=H(x);g(ke.$$.fragment,q),Nt=d(q),Ve=u(q,"P",{"data-svelte-h":!0}),T(Ve)!=="svelte-k1ep3z"&&(Ve.innerHTML=hn),qt=d(q),g(O.$$.fragment,q),Bt=d(q),g(D.$$.fragment,q),q.forEach(a),z.forEach(a),gt=d(e),g(we.$$.fragment,e),_t=d(e),J=u(e,"DIV",{class:!0});var C=H(J);g(ve.$$.fragment,C),Gt=d(C),Xe=u(C,"P",{"data-svelte-h":!0}),T(Xe)!=="svelte-1k5jxq4"&&(Xe.textContent=fn),Ht=d(C),Re=u(C,"P",{"data-svelte-h":!0}),T(Re)!=="svelte-q52n56"&&(Re.innerHTML=gn),Vt=d(C),Le=u(C,"P",{"data-svelte-h":!0}),T(Le)!=="svelte-hswkmf"&&(Le.innerHTML=_n),Xt=d(C),I=u(C,"DIV",{class:!0});var B=H(I);g($e.$$.fragment,B),Rt=d(B),Ee=u(B,"P",{"data-svelte-h":!0}),T(Ee)!=="svelte-1e1yexr"&&(Ee.innerHTML=Mn),Lt=d(B),g(K.$$.fragment,B),Et=d(B),g(ee.$$.fragment,B),B.forEach(a),C.forEach(a),Mt=d(e),g(Je.$$.fragment,e),yt=d(e),R=u(e,"DIV",{class:!0});var xe=H(R);g(je.$$.fragment,xe),St=d(xe),Z=u(xe,"DIV",{class:!0});var G=H(Z);g(Ue.$$.fragment,G),Yt=d(G),Se=u(G,"P",{"data-svelte-h":!0}),T(Se)!=="svelte-93yfjv"&&(Se.innerHTML=yn),Pt=d(G),g(te.$$.fragment,G),Qt=d(G),g(ne.$$.fragment,G),G.forEach(a),xe.forEach(a),bt=d(e),g(ze.$$.fragment,e),Tt=d(e),Ye=u(e,"P",{}),H(Ye).forEach(a),this.h()},h(){W(t,"name","hf:doc:metadata"),W(t,"content",Rn),jn(F,"float","right"),W(Y,"class","flex justify-center"),W(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),W(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){h(document.head,t),i(e,p,o),i(e,s,o),i(e,r,o),i(e,c,o),i(e,k,o),i(e,F,o),i(e,Qe,o),_(oe,e,o),i(e,Ae,o),i(e,se,o),i(e,Oe,o),i(e,ae,o),i(e,De,o),i(e,re,o),i(e,Ke,o),_(E,e,o),i(e,et,o),i(e,ie,o),i(e,tt,o),_(S,e,o),i(e,nt,o),i(e,le,o),i(e,ot,o),_(de,e,o),i(e,st,o),i(e,Y,o),i(e,at,o),_(ce,e,o),i(e,rt,o),i(e,pe,o),i(e,it,o),_(me,e,o),i(e,lt,o),i(e,ue,o),i(e,dt,o),_(he,e,o),i(e,ct,o),_(fe,e,o),i(e,pt,o),i(e,j,o),_(ge,j,null),h(j,vt),h(j,Ze),h(j,$t),h(j,We),h(j,Jt),_(P,j,null),i(e,mt,o),_(_e,e,o),i(e,ut,o),i(e,v,o),_(Me,v,null),h(v,jt),h(v,Fe),h(v,Ut),h(v,Ne),h(v,zt),_(Q,v,null),h(v,Ct),h(v,A),_(ye,A,null),h(A,xt),h(A,qe),i(e,ht,o),_(be,e,o),i(e,ft,o),i(e,$,o),_(Te,$,null),h($,It),h($,Be),h($,Zt),h($,Ge),h($,Wt),h($,He),h($,Ft),h($,x),_(ke,x,null),h(x,Nt),h(x,Ve),h(x,qt),_(O,x,null),h(x,Bt),_(D,x,null),i(e,gt,o),_(we,e,o),i(e,_t,o),i(e,J,o),_(ve,J,null),h(J,Gt),h(J,Xe),h(J,Ht),h(J,Re),h(J,Vt),h(J,Le),h(J,Xt),h(J,I),_($e,I,null),h(I,Rt),h(I,Ee),h(I,Lt),_(K,I,null),h(I,Et),_(ee,I,null),i(e,Mt,o),_(Je,e,o),i(e,yt,o),i(e,R,o),_(je,R,null),h(R,St),h(R,Z),_(Ue,Z,null),h(Z,Yt),h(Z,Se),h(Z,Pt),_(te,Z,null),h(Z,Qt),_(ne,Z,null),i(e,bt,o),_(ze,e,o),i(e,Tt,o),i(e,Ye,o),kt=!0},p(e,[o]){const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),E.$set(N);const U={};o&2&&(U.$$scope={dirty:o,ctx:e}),S.$set(U);const Ce={};o&2&&(Ce.$$scope={dirty:o,ctx:e}),P.$set(Ce);const z={};o&2&&(z.$$scope={dirty:o,ctx:e}),Q.$set(z);const q={};o&2&&(q.$$scope={dirty:o,ctx:e}),O.$set(q);const C={};o&2&&(C.$$scope={dirty:o,ctx:e}),D.$set(C);const B={};o&2&&(B.$$scope={dirty:o,ctx:e}),K.$set(B);const xe={};o&2&&(xe.$$scope={dirty:o,ctx:e}),ee.$set(xe);const G={};o&2&&(G.$$scope={dirty:o,ctx:e}),te.$set(G);const bn={};o&2&&(bn.$$scope={dirty:o,ctx:e}),ne.$set(bn)},i(e){kt||(M(oe.$$.fragment,e),M(E.$$.fragment,e),M(S.$$.fragment,e),M(de.$$.fragment,e),M(ce.$$.fragment,e),M(me.$$.fragment,e),M(he.$$.fragment,e),M(fe.$$.fragment,e),M(ge.$$.fragment,e),M(P.$$.fragment,e),M(_e.$$.fragment,e),M(Me.$$.fragment,e),M(Q.$$.fragment,e),M(ye.$$.fragment,e),M(be.$$.fragment,e),M(Te.$$.fragment,e),M(ke.$$.fragment,e),M(O.$$.fragment,e),M(D.$$.fragment,e),M(we.$$.fragment,e),M(ve.$$.fragment,e),M($e.$$.fragment,e),M(K.$$.fragment,e),M(ee.$$.fragment,e),M(Je.$$.fragment,e),M(je.$$.fragment,e),M(Ue.$$.fragment,e),M(te.$$.fragment,e),M(ne.$$.fragment,e),M(ze.$$.fragment,e),kt=!0)},o(e){y(oe.$$.fragment,e),y(E.$$.fragment,e),y(S.$$.fragment,e),y(de.$$.fragment,e),y(ce.$$.fragment,e),y(me.$$.fragment,e),y(he.$$.fragment,e),y(fe.$$.fragment,e),y(ge.$$.fragment,e),y(P.$$.fragment,e),y(_e.$$.fragment,e),y(Me.$$.fragment,e),y(Q.$$.fragment,e),y(ye.$$.fragment,e),y(be.$$.fragment,e),y(Te.$$.fragment,e),y(ke.$$.fragment,e),y(O.$$.fragment,e),y(D.$$.fragment,e),y(we.$$.fragment,e),y(ve.$$.fragment,e),y($e.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(Je.$$.fragment,e),y(je.$$.fragment,e),y(Ue.$$.fragment,e),y(te.$$.fragment,e),y(ne.$$.fragment,e),y(ze.$$.fragment,e),kt=!1},d(e){e&&(a(p),a(s),a(r),a(c),a(k),a(F),a(Qe),a(Ae),a(se),a(Oe),a(ae),a(De),a(re),a(Ke),a(et),a(ie),a(tt),a(nt),a(le),a(ot),a(st),a(Y),a(at),a(rt),a(pe),a(it),a(lt),a(ue),a(dt),a(ct),a(pt),a(j),a(mt),a(ut),a(v),a(ht),a(ft),a($),a(gt),a(_t),a(J),a(Mt),a(yt),a(R),a(bt),a(Tt),a(Ye)),a(t),b(oe,e),b(E,e),b(S,e),b(de,e),b(ce,e),b(me,e),b(he,e),b(fe,e),b(ge),b(P),b(_e,e),b(Me),b(Q),b(ye),b(be,e),b(Te),b(ke),b(O),b(D),b(we,e),b(ve),b($e),b(K),b(ee),b(Je,e),b(je),b(Ue),b(te),b(ne),b(ze,e)}}}const Rn='{"title":"MarianMT","local":"marianmt","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"MarianConfig","local":"transformers.MarianConfig","sections":[],"depth":2},{"title":"MarianTokenizer","local":"transformers.MarianTokenizer","sections":[],"depth":2},{"title":"MarianModel","local":"transformers.MarianModel","sections":[],"depth":2},{"title":"MarianMTModel","local":"transformers.MarianMTModel","sections":[],"depth":2},{"title":"MarianForCausalLM","local":"transformers.MarianForCausalLM","sections":[],"depth":2}],"depth":1}';function Ln(w){return wn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Kn extends vn{constructor(t){super(),$n(this,t,Ln,Xn,kn,{})}}export{Kn as component};
