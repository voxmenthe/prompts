import{s as Rs,o as Ls,n as G}from"../chunks/scheduler.18a86fab.js";import{S as Xs,i as Ss,g as p,s as a,r as f,A as Qs,h as m,f as i,c as r,j as x,x as M,u as g,k as P,l as Es,y as l,a as c,v as _,d as T,t as b,w as y}from"../chunks/index.98837b22.js";import{T as de}from"../chunks/Tip.77304350.js";import{D as U}from"../chunks/Docstring.a1ef7999.js";import{C as N}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ce}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as K,E as Ds}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as As,a as Dn}from"../chunks/HfOption.6641485e.js";function Ys(v){let t,u="Click on the GPT-2 models in the right sidebar for more examples of how to apply GPT-2 to different language tasks.";return{c(){t=p("p"),t.textContent=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-1dbj8pt"&&(t.textContent=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function Os(v){let t,u;return t=new N({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMjBtb2RlbCUzRCUyMm9wZW5haS1jb21tdW5pdHklMkZncHQyJTIyJTJDJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTIwZGV2aWNlJTNEMCklMEFwaXBlbGluZSglMjJIZWxsbyUyQyUyMEknbSUyMGElMjBsYW5ndWFnZSUyMG1vZGVsJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(task=<span class="hljs-string">&quot;text-generation&quot;</span>, model=<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>, dtype=torch.float16, device=<span class="hljs-number">0</span>)
pipeline(<span class="hljs-string">&quot;Hello, I&#x27;m a language model&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){_(t,o,d),u=!0},p:G,i(o){u||(T(t.$$.fragment,o),u=!0)},o(o){b(t.$$.fragment,o),u=!1},d(o){y(t,o)}}}function Ks(v){let t,u;return t=new N({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haS1jb21tdW5pdHklMkZncHQyJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMEknbSUyMGElMjBsYW5ndWFnZSUyMG1vZGVsJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>, dtype=torch.float16, device_map=<span class="hljs-string">&quot;auto&quot;</span>, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

input_ids = tokenizer(<span class="hljs-string">&quot;Hello, I&#x27;m a language model&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){_(t,o,d),u=!0},p:G,i(o){u||(T(t.$$.fragment,o),u=!0)},o(o){b(t.$$.fragment,o),u=!1},d(o){y(t,o)}}}function ea(v){let t,u;return t=new N({props:{code:"ZWNobyUyMC1lJTIwJTIySGVsbG8lMkMlMjBJJ20lMjBhJTIwbGFuZ3VhZ2UlMjBtb2RlbCUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQtZ2VuZXJhdGlvbiUyMC0tbW9kZWwlMjBvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Hello, I&#x27;m a language model&quot;</span> | transformers run --task text-generation --model openai-community/gpt2 --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){_(t,o,d),u=!0},p:G,i(o){u||(T(t.$$.fragment,o),u=!0)},o(o){b(t.$$.fragment,o),u=!1},d(o){y(t,o)}}}function ta(v){let t,u,o,d,k,n;return t=new Dn({props:{id:"usage",option:"Pipeline",$$slots:{default:[Os]},$$scope:{ctx:v}}}),o=new Dn({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ks]},$$scope:{ctx:v}}}),k=new Dn({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ea]},$$scope:{ctx:v}}}),{c(){f(t.$$.fragment),u=a(),f(o.$$.fragment),d=a(),f(k.$$.fragment)},l(h){g(t.$$.fragment,h),u=r(h),g(o.$$.fragment,h),d=r(h),g(k.$$.fragment,h)},m(h,w){_(t,h,w),c(h,u,w),_(o,h,w),c(h,d,w),_(k,h,w),n=!0},p(h,w){const to={};w&2&&(to.$$scope={dirty:w,ctx:h}),t.$set(to);const xe={};w&2&&(xe.$$scope={dirty:w,ctx:h}),o.$set(xe);const ae={};w&2&&(ae.$$scope={dirty:w,ctx:h}),k.$set(ae)},i(h){n||(T(t.$$.fragment,h),T(o.$$.fragment,h),T(k.$$.fragment,h),n=!0)},o(h){b(t.$$.fragment,h),b(o.$$.fragment,h),b(k.$$.fragment,h),n=!1},d(h){h&&(i(u),i(d)),y(t,h),y(o,h),y(k,h)}}}function oa(v){let t,u="Example:",o,d,k;return d=new N({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdQVDJDb25maWclMkMlMjBHUFQyTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwR1BUMiUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwR1BUMkNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBHUFQyTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> GPT2Config, GPT2Model

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a GPT2 configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = GPT2Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function na(v){let t,u="be encoded differently whether it is at the beginning of the sentence (without space) or not:",o,d,k;return d=new N({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdQVDJUb2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBHUFQyVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMiklMEF0b2tlbml6ZXIoJTIySGVsbG8lMjB3b3JsZCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMEF0b2tlbml6ZXIoJTIyJTIwSGVsbG8lMjB3b3JsZCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> GPT2Tokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = GPT2Tokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">15496</span>, <span class="hljs-number">995</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">18435</span>, <span class="hljs-number">995</span>]`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-12atnao"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function sa(v){let t,u="When used with <code>is_split_into_words=True</code>, this tokenizer will add a space before each word (even the first one).";return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-jhmxzm"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function aa(v){let t,u="be encoded differently whether it is at the beginning of the sentence (without space) or not:",o,d,k;return d=new N({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdQVDJUb2tlbml6ZXJGYXN0JTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwR1BUMlRva2VuaXplckZhc3QuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haS1jb21tdW5pdHklMkZncHQyJTIyKSUwQXRva2VuaXplciglMjJIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUwQXRva2VuaXplciglMjIlMjBIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> GPT2TokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = GPT2TokenizerFast.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">15496</span>, <span class="hljs-number">995</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">18435</span>, <span class="hljs-number">995</span>]`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-12atnao"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function ra(v){let t,u="When used with <code>is_split_into_words=True</code>, this tokenizer needs to be instantiated with <code>add_prefix_space=True</code>.";return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-9gg91e"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function ia(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function la(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function da(v){let t,u="Example:",o,d,k;return d=new N({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEdQVDJMTUhlYWRNb2RlbCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haS1jb21tdW5pdHklMkZncHQyJTIyKSUwQW1vZGVsJTIwJTNEJTIwR1BUMkxNSGVhZE1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEaW5wdXRzJTVCJTIyaW5wdXRfaWRzJTIyJTVEKSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPT2LMHeadModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2LMHeadModel.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function ca(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function pa(v){let t,u="Example:",o,d,k;return d=new N({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEdQVDJEb3VibGVIZWFkc01vZGVsJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIpJTBBbW9kZWwlMjAlM0QlMjBHUFQyRG91YmxlSGVhZHNNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIpJTBBJTBBJTIzJTIwQWRkJTIwYSUyMCU1QkNMUyU1RCUyMHRvJTIwdGhlJTIwdm9jYWJ1bGFyeSUyMCh3ZSUyMHNob3VsZCUyMHRyYWluJTIwaXQlMjBhbHNvISklMEFudW1fYWRkZWRfdG9rZW5zJTIwJTNEJTIwdG9rZW5pemVyLmFkZF9zcGVjaWFsX3Rva2VucyglN0IlMjJjbHNfdG9rZW4lMjIlM0ElMjAlMjIlNUJDTFMlNUQlMjIlN0QpJTBBJTIzJTIwVXBkYXRlJTIwdGhlJTIwbW9kZWwlMjBlbWJlZGRpbmdzJTIwd2l0aCUyMHRoZSUyMG5ldyUyMHZvY2FidWxhcnklMjBzaXplJTBBZW1iZWRkaW5nX2xheWVyJTIwJTNEJTIwbW9kZWwucmVzaXplX3Rva2VuX2VtYmVkZGluZ3MobGVuKHRva2VuaXplcikpJTBBJTBBY2hvaWNlcyUyMCUzRCUyMCU1QiUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjAlNUJDTFMlNUQlMjIlMkMlMjAlMjJIZWxsbyUyQyUyMG15JTIwY2F0JTIwaXMlMjBjdXRlJTIwJTVCQ0xTJTVEJTIyJTVEJTBBZW5jb2RlZF9jaG9pY2VzJTIwJTNEJTIwJTVCdG9rZW5pemVyLmVuY29kZShzKSUyMGZvciUyMHMlMjBpbiUyMGNob2ljZXMlNUQlMEFjbHNfdG9rZW5fbG9jYXRpb24lMjAlM0QlMjAlNUJ0b2tlbnMuaW5kZXgodG9rZW5pemVyLmNsc190b2tlbl9pZCklMjBmb3IlMjB0b2tlbnMlMjBpbiUyMGVuY29kZWRfY2hvaWNlcyU1RCUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRvcmNoLnRlbnNvcihlbmNvZGVkX2Nob2ljZXMpLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUzQSUyMDElMkMlMjBudW1iZXIlMjBvZiUyMGNob2ljZXMlM0ElMjAyJTBBbWNfdG9rZW5faWRzJTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QmNsc190b2tlbl9sb2NhdGlvbiU1RCklMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlM0ElMjAxJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUyQyUyMG1jX3Rva2VuX2lkcyUzRG1jX3Rva2VuX2lkcyklMEFsbV9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQW1jX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubWNfbG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPT2DoubleHeadsModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2DoubleHeadsModel.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Add a [CLS] to the vocabulary (we should train it also!)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_added_tokens = tokenizer.add_special_tokens({<span class="hljs-string">&quot;cls_token&quot;</span>: <span class="hljs-string">&quot;[CLS]&quot;</span>})
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Update the model embeddings with the new vocabulary size</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>embedding_layer = model.resize_token_embeddings(<span class="hljs-built_in">len</span>(tokenizer))

<span class="hljs-meta">&gt;&gt;&gt; </span>choices = [<span class="hljs-string">&quot;Hello, my dog is cute [CLS]&quot;</span>, <span class="hljs-string">&quot;Hello, my cat is cute [CLS]&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>encoded_choices = [tokenizer.encode(s) <span class="hljs-keyword">for</span> s <span class="hljs-keyword">in</span> choices]
<span class="hljs-meta">&gt;&gt;&gt; </span>cls_token_location = [tokens.index(tokenizer.cls_token_id) <span class="hljs-keyword">for</span> tokens <span class="hljs-keyword">in</span> encoded_choices]

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = torch.tensor(encoded_choices).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># Batch size: 1, number of choices: 2</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mc_token_ids = torch.tensor([cls_token_location])  <span class="hljs-comment"># Batch size: 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids, mc_token_ids=mc_token_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>lm_logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>mc_logits = outputs.mc_logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function ma(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function ua(v){let t,u="Example:",o,d,k;return d=new N({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHUFQyRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haS1jb21tdW5pdHklMkZncHQyJTIyKSUwQW1vZGVsJTIwJTNEJTIwR1BUMkZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPT2ForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2ForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function ha(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function fa(v){let t,u="Example of single-label classification:",o,d,k;return d=new N({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEdQVDJGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIpJTBBbW9kZWwlMjAlM0QlMjBHUFQyRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMEdQVDJGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPT2ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-ykxpe4"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function ga(v){let t,u="Example of multi-label classification:",o,d,k;return d=new N({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEdQVDJGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIpJTBBbW9kZWwlMjAlM0QlMjBHUFQyRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMEdQVDJGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPT2ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2ForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;openai-community/gpt2&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-1l8e32d"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function _a(v){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:G,d(o){o&&i(t)}}}function Ta(v){let t,u="Example:",o,d,k;return d=new N({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHUFQyRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIpJTBBbW9kZWwlMjAlM0QlMjBHUFQyRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPT2ForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPT2ForTokenClassification.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=u),o=r(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),k=!0},p:G,i(n){k||(T(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(i(t),i(o)),y(d,n)}}}function ba(v){let t,u,o,d,k,n="<em>This model was released on 2019-02-14 and added to Hugging Face Transformers on 2020-11-16.</em>",h,w,to='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',xe,ae,so,Pe,An='<a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" rel="nofollow">GPT-2</a> is a scaled up version of GPT, a causal transformer language model, with 10x more parameters and training data. The model was pretrained on a 40GB dataset to predict the next word in a sequence based on all the previous words. This approach enabled the model to perform many downstream tasks in a zero-shot setting. The blog post released by OpenAI can be found <a href="https://openai.com/index/better-language-models/" rel="nofollow">here</a>.',ao,ze,Yn="The model architecture uses a unidirectional (causal) attention mechanism where each token can only attend to previous tokens, making it particularly effective for text generation tasks.",ro,We,On='You can find all the original GPT-2 checkpoints under the <a href="https://huggingface.co/openai-community?search_models=gpt" rel="nofollow">OpenAI community</a> organization.',io,pe,lo,Ue,Kn='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',co,me,po,Fe,es="One can also serve the model using vLLM with the <code>transformers backend</code>.",mo,Ie,uo,qe,ts='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',ho,He,os='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to 4-bits.',fo,Ze,go,Be,_o,Ne,ns='<li>Pad inputs on the right because GPT-2 uses absolute position embeddings.</li> <li>GPT-2 can reuse previously computed key-value attention pairs. Access this feature with the <a href="https://huggingface.co/docs/transformers//en/model_doc/gpt2#transformers.GPT2Model.forward.past_key_values" rel="nofollow">past_key_values</a> parameter in <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Model.forward">GPT2Model.forward()</a>.</li> <li>Enable the <a href="https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.scale_attn_by_inverse_layer_idx" rel="nofollow">scale_attn_by_inverse_layer_idx</a> and <a href="https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.reorder_and_upcast_attn" rel="nofollow">reorder_and_upcast_attn</a> parameters to apply the training stability improvements from <a href="./mistral">Mistral</a>.</li>',To,Ve,bo,V,Re,No,_t,ss=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Model">GPT2Model</a> or a <code>TFGPT2Model</code>. It is used to
instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the GPT-2
<a href="https://huggingface.co/openai-community/gpt2" rel="nofollow">openai-community/gpt2</a> architecture.`,Vo,Tt,as=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ro,ue,yo,Le,Mo,$,Xe,Lo,bt,rs="Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.",Xo,yt,is="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",So,he,Qo,Mt,ls=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.`,Eo,fe,Do,kt,ds=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,Ao,vt,Se,ko,Qe,vo,J,Ee,Yo,wt,cs=`Construct a “fast” GPT-2 tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on byte-level
Byte-Pair-Encoding.`,Oo,$t,ps="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",Ko,ge,en,Gt,ms=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer, but since
the model was not pretrained this way, it might yield a decrease in performance.`,tn,_e,on,Jt,us=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,wo,De,$o,ie,Ae,nn,jt,hs="Base class for outputs of models predicting if two sentences are consecutive or not.",Go,Ye,Jo,F,Oe,sn,Ct,fs="The bare Gpt2 Model outputting raw hidden-states without any specific head on top.",an,xt,gs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,rn,Pt,_s=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ln,re,Ke,dn,zt,Ts='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Model">GPT2Model</a> forward method, overrides the <code>__call__</code> special method.',cn,Te,jo,et,Co,I,tt,pn,Wt,bs=`The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`,mn,Ut,ys=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,un,Ft,Ms=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,hn,D,ot,fn,It,ks='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel">GPT2LMHeadModel</a> forward method, overrides the <code>__call__</code> special method.',gn,be,_n,ye,xo,nt,Po,q,st,Tn,qt,vs=`The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
input embeddings, the classification head takes as input the input of a specified classification token index in the
input sequence).`,bn,Ht,ws=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,yn,Zt,$s=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Mn,A,at,kn,Bt,Gs='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel">GPT2DoubleHeadsModel</a> forward method, overrides the <code>__call__</code> special method.',vn,Me,wn,ke,zo,rt,Wo,H,it,$n,Nt,Js=`The Gpt2 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Gn,Vt,js=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Jn,Rt,Cs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,jn,Y,lt,Cn,Lt,xs='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForQuestionAnswering">GPT2ForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',xn,ve,Pn,we,Uo,dt,Fo,j,ct,zn,Xt,Ps="The GPT2 Model transformer with a sequence classification head on top (linear layer).",Wn,St,zs=`<a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification">GPT2ForSequenceClassification</a> uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.`,Un,Qt,Ws=`Since it does classification on the last token, it requires to know the position of the last token. If a
<code>pad_token_id</code> is defined in the configuration, it finds the last token that is not a padding token in each row. If
no <code>pad_token_id</code> is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when <code>inputs_embeds</code> are passed instead of <code>input_ids</code>, it does the same (take the last value in
each row of the batch).`,Fn,Et,Us=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,In,Dt,Fs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,qn,B,pt,Hn,At,Is='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification">GPT2ForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Zn,$e,Bn,Ge,Nn,Je,Io,mt,qo,Z,ut,Vn,Yt,qs=`The Gpt2 transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Rn,Ot,Hs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ln,Kt,Zs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xn,O,ht,Sn,eo,Bs='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForTokenClassification">GPT2ForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Qn,je,En,Ce,Ho,ft,Zo,oo,Bo;return ae=new K({props:{title:"GPT-2",local:"gpt-2",headingTag:"h1"}}),pe=new de({props:{warning:!1,$$slots:{default:[Ys]},$$scope:{ctx:v}}}),me=new As({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[ta]},$$scope:{ctx:v}}}),Ie=new N({props:{code:"dmxsbSUyMHNlcnZlJTIwb3BlbmFpLWNvbW11bml0eSUyRmdwdDIlMjAtLW1vZGVsLWltcCUyMHRyYW5zZm9ybWVycw==",highlighted:'vllm serve openai-community/gpt2 <span class="hljs-comment">--model-imp transformers</span>',wrap:!1}}),Ze=new N({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwcGlwZWxpbmUlMEElMEFxdWFudGl6YXRpb25fY29uZmlnJTIwJTNEJTIwQml0c0FuZEJ5dGVzQ29uZmlnKCUwQSUyMCUyMCUyMCUyMGxvYWRfaW5fNGJpdCUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjBibmJfNGJpdF9xdWFudF90eXBlJTNEJTIybmY0JTIyJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfY29tcHV0ZV9kdHlwZSUzRCUyMmZsb2F0MTYlMjIlMkMlMEElMjAlMjAlMjAlMjBibmJfNGJpdF91c2VfZG91YmxlX3F1YW50JTNEVHJ1ZSUwQSklMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0Mi14bCUyMiUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RxdWFudGl6YXRpb25fY29uZmlnJTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMEEpJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpLWNvbW11bml0eSUyRmdwdDIteGwlMjIpJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMk9uY2UlMjB1cG9uJTIwYSUyMHRpbWUlMkMlMjB0aGVyZSUyMHdhcyUyMGElMjBtYWdpY2FsJTIwZm9yZXN0JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMTAwKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_compute_dtype=<span class="hljs-string">&quot;float16&quot;</span>,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>
)

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;openai-community/gpt2-xl&quot;</span>,
    quantization_config=quantization_config,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2-xl&quot;</span>)
inputs = tokenizer(<span class="hljs-string">&quot;Once upon a time, there was a magical forest&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),Be=new K({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Ve=new K({props:{title:"GPT2Config",local:"transformers.GPT2Config",headingTag:"h2"}}),Re=new U({props:{name:"class transformers.GPT2Config",anchor:"transformers.GPT2Config",parameters:[{name:"vocab_size",val:" = 50257"},{name:"n_positions",val:" = 1024"},{name:"n_embd",val:" = 768"},{name:"n_layer",val:" = 12"},{name:"n_head",val:" = 12"},{name:"n_inner",val:" = None"},{name:"activation_function",val:" = 'gelu_new'"},{name:"resid_pdrop",val:" = 0.1"},{name:"embd_pdrop",val:" = 0.1"},{name:"attn_pdrop",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-05"},{name:"initializer_range",val:" = 0.02"},{name:"summary_type",val:" = 'cls_index'"},{name:"summary_use_proj",val:" = True"},{name:"summary_activation",val:" = None"},{name:"summary_proj_to_labels",val:" = True"},{name:"summary_first_dropout",val:" = 0.1"},{name:"scale_attn_weights",val:" = True"},{name:"use_cache",val:" = True"},{name:"bos_token_id",val:" = 50256"},{name:"eos_token_id",val:" = 50256"},{name:"scale_attn_by_inverse_layer_idx",val:" = False"},{name:"reorder_and_upcast_attn",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GPT2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50257) &#x2014;
Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Model">GPT2Model</a> or <code>TFGPT2Model</code>.`,name:"vocab_size"},{anchor:"transformers.GPT2Config.n_positions",description:`<strong>n_positions</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"n_positions"},{anchor:"transformers.GPT2Config.n_embd",description:`<strong>n_embd</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the embeddings and hidden states.`,name:"n_embd"},{anchor:"transformers.GPT2Config.n_layer",description:`<strong>n_layer</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"n_layer"},{anchor:"transformers.GPT2Config.n_head",description:`<strong>n_head</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"n_head"},{anchor:"transformers.GPT2Config.n_inner",description:`<strong>n_inner</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Dimensionality of the inner feed-forward layers. <code>None</code> will set it to 4 times n_embd`,name:"n_inner"},{anchor:"transformers.GPT2Config.activation_function",description:`<strong>activation_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;gelu_new&quot;</code>) &#x2014;
Activation function, to be selected in the list <code>[&quot;relu&quot;, &quot;silu&quot;, &quot;gelu&quot;, &quot;tanh&quot;, &quot;gelu_new&quot;]</code>.`,name:"activation_function"},{anchor:"transformers.GPT2Config.resid_pdrop",description:`<strong>resid_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"resid_pdrop"},{anchor:"transformers.GPT2Config.embd_pdrop",description:`<strong>embd_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the embeddings.`,name:"embd_pdrop"},{anchor:"transformers.GPT2Config.attn_pdrop",description:`<strong>attn_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention.`,name:"attn_pdrop"},{anchor:"transformers.GPT2Config.layer_norm_epsilon",description:`<strong>layer_norm_epsilon</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon to use in the layer normalization layers.`,name:"layer_norm_epsilon"},{anchor:"transformers.GPT2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.GPT2Config.summary_type",description:`<strong>summary_type</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;cls_index&quot;</code>) &#x2014;
Argument used when doing sequence summary, used in the models <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel">GPT2DoubleHeadsModel</a> and
<code>TFGPT2DoubleHeadsModel</code>.</p>
<p>Has to be one of the following options:</p>
<ul>
<li><code>&quot;last&quot;</code>: Take the last token hidden state (like XLNet).</li>
<li><code>&quot;first&quot;</code>: Take the first token hidden state (like BERT).</li>
<li><code>&quot;mean&quot;</code>: Take the mean of all tokens hidden states.</li>
<li><code>&quot;cls_index&quot;</code>: Supply a Tensor of classification token position (like GPT/GPT-2).</li>
<li><code>&quot;attn&quot;</code>: Not implemented now, use multi-head attention.</li>
</ul>`,name:"summary_type"},{anchor:"transformers.GPT2Config.summary_use_proj",description:`<strong>summary_use_proj</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Argument used when doing sequence summary, used in the models <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel">GPT2DoubleHeadsModel</a> and
<code>TFGPT2DoubleHeadsModel</code>.</p>
<p>Whether or not to add a projection after the vector extraction.`,name:"summary_use_proj"},{anchor:"transformers.GPT2Config.summary_activation",description:`<strong>summary_activation</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Argument used when doing sequence summary. Used in for the multiple choice head in
<a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel">GPT2DoubleHeadsModel</a>.</p>
<p>Pass <code>&quot;tanh&quot;</code> for a tanh activation to the output, any other value will result in no activation.`,name:"summary_activation"},{anchor:"transformers.GPT2Config.summary_proj_to_labels",description:`<strong>summary_proj_to_labels</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Argument used when doing sequence summary, used in the models <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel">GPT2DoubleHeadsModel</a> and
<code>TFGPT2DoubleHeadsModel</code>.</p>
<p>Whether the projection outputs should have <code>config.num_labels</code> or <code>config.hidden_size</code> classes.`,name:"summary_proj_to_labels"},{anchor:"transformers.GPT2Config.summary_first_dropout",description:`<strong>summary_first_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Argument used when doing sequence summary, used in the models <a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel">GPT2DoubleHeadsModel</a> and
<code>TFGPT2DoubleHeadsModel</code>.</p>
<p>The dropout ratio to be used after the projection and activation.`,name:"summary_first_dropout"},{anchor:"transformers.GPT2Config.scale_attn_weights",description:`<strong>scale_attn_weights</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Scale attention weights by dividing by sqrt(hidden_size)..`,name:"scale_attn_weights"},{anchor:"transformers.GPT2Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.GPT2Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 50256) &#x2014;
Id of the beginning of sentence token in the vocabulary.`,name:"bos_token_id"},{anchor:"transformers.GPT2Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 50256) &#x2014;
Id of the end of sentence token in the vocabulary.`,name:"eos_token_id"},{anchor:"transformers.GPT2Config.scale_attn_by_inverse_layer_idx",description:`<strong>scale_attn_by_inverse_layer_idx</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to additionally scale attention weights by <code>1 / layer_idx + 1</code>.`,name:"scale_attn_by_inverse_layer_idx"},{anchor:"transformers.GPT2Config.reorder_and_upcast_attn",description:`<strong>reorder_and_upcast_attn</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
dot-product/softmax to float() when training with mixed precision.`,name:"reorder_and_upcast_attn"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/configuration_gpt2.py#L31"}}),ue=new ce({props:{anchor:"transformers.GPT2Config.example",$$slots:{default:[oa]},$$scope:{ctx:v}}}),Le=new K({props:{title:"GPT2Tokenizer",local:"transformers.GPT2Tokenizer",headingTag:"h2"}}),Xe=new U({props:{name:"class transformers.GPT2Tokenizer",anchor:"transformers.GPT2Tokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"errors",val:" = 'replace'"},{name:"unk_token",val:" = '<|endoftext|>'"},{name:"bos_token",val:" = '<|endoftext|>'"},{name:"eos_token",val:" = '<|endoftext|>'"},{name:"pad_token",val:" = None"},{name:"add_prefix_space",val:" = False"},{name:"add_bos_token",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GPT2Tokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.GPT2Tokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.GPT2Tokenizer.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.GPT2Tokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.GPT2Tokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The beginning of sequence token.`,name:"bos_token"},{anchor:"transformers.GPT2Tokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.GPT2Tokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.GPT2Tokenizer.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (GPT2 tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"},{anchor:"transformers.GPT2Tokenizer.add_bos_token",description:`<strong>add_bos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
word just as any other word.`,name:"add_bos_token"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/tokenization_gpt2.py#L75"}}),he=new ce({props:{anchor:"transformers.GPT2Tokenizer.example",$$slots:{default:[na]},$$scope:{ctx:v}}}),fe=new de({props:{$$slots:{default:[sa]},$$scope:{ctx:v}}}),Se=new U({props:{name:"save_vocabulary",anchor:"transformers.GPT2Tokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/tokenization_gpt2.py#L298"}}),Qe=new K({props:{title:"GPT2TokenizerFast",local:"transformers.GPT2TokenizerFast",headingTag:"h2"}}),Ee=new U({props:{name:"class transformers.GPT2TokenizerFast",anchor:"transformers.GPT2TokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"unk_token",val:" = '<|endoftext|>'"},{name:"bos_token",val:" = '<|endoftext|>'"},{name:"eos_token",val:" = '<|endoftext|>'"},{name:"add_prefix_space",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GPT2TokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.GPT2TokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.GPT2TokenizerFast.tokenizer_file",description:`<strong>tokenizer_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Path to <a href="https://github.com/huggingface/tokenizers" rel="nofollow">tokenizers</a> file (generally has a .json extension) that
contains everything needed to load the tokenizer.`,name:"tokenizer_file"},{anchor:"transformers.GPT2TokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.GPT2TokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The beginning of sequence token.`,name:"bos_token"},{anchor:"transformers.GPT2TokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.GPT2TokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (GPT2 tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/tokenization_gpt2_fast.py#L30"}}),ge=new ce({props:{anchor:"transformers.GPT2TokenizerFast.example",$$slots:{default:[aa]},$$scope:{ctx:v}}}),_e=new de({props:{$$slots:{default:[ra]},$$scope:{ctx:v}}}),De=new K({props:{title:"GPT2 specific outputs",local:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput",headingTag:"h2"}}),Ae=new U({props:{name:"class transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput",anchor:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mc_loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mc_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) &#x2014;
Language modeling loss.`,name:"loss"},{anchor:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput.mc_loss",description:`<strong>mc_loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>mc_labels</code> is provided) &#x2014;
Multiple choice classification loss.`,name:"mc_loss"},{anchor:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).`,name:"logits"},{anchor:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput.mc_logits",description:`<strong>mc_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) &#x2014;
Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).`,name:"mc_logits"},{anchor:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
Tuple of length <code>config.n_layers</code>, containing tuples of tensors of shape <code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.`,name:"past_key_values"},{anchor:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L615"}}),Ye=new K({props:{title:"GPT2Model",local:"transformers.GPT2Model",headingTag:"h2"}}),Oe=new U({props:{name:"class transformers.GPT2Model",anchor:"transformers.GPT2Model",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPT2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Model">GPT2Model</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L695"}}),Ke=new U({props:{name:"forward",anchor:"transformers.GPT2Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[tuple[tuple[torch.Tensor]], transformers.cache_utils.Cache, NoneType] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GPT2Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPT2Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[tuple[tuple[torch.Tensor]], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPT2Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.GPT2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPT2Model.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPT2Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPT2Model.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPT2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPT2Model.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.GPT2Model.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.GPT2Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPT2Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPT2Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPT2Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L776",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config"
>GPT2Config</a>) and inputs.</p>
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
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> and <code>config.add_cross_attention=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Te=new de({props:{$$slots:{default:[ia]},$$scope:{ctx:v}}}),et=new K({props:{title:"GPT2LMHeadModel",local:"transformers.GPT2LMHeadModel",headingTag:"h2"}}),tt=new U({props:{name:"class transformers.GPT2LMHeadModel",anchor:"transformers.GPT2LMHeadModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPT2LMHeadModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2LMHeadModel">GPT2LMHeadModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L983"}}),ot=new U({props:{name:"forward",anchor:"transformers.GPT2LMHeadModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GPT2LMHeadModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPT2LMHeadModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPT2LMHeadModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.GPT2LMHeadModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPT2LMHeadModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPT2LMHeadModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPT2LMHeadModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPT2LMHeadModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPT2LMHeadModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.GPT2LMHeadModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.GPT2LMHeadModel.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.GPT2LMHeadModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPT2LMHeadModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPT2LMHeadModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPT2LMHeadModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.GPT2LMHeadModel.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1029",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config"
>GPT2Config</a>) and inputs.</p>
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
`}}),be=new de({props:{$$slots:{default:[la]},$$scope:{ctx:v}}}),ye=new ce({props:{anchor:"transformers.GPT2LMHeadModel.forward.example",$$slots:{default:[da]},$$scope:{ctx:v}}}),nt=new K({props:{title:"GPT2DoubleHeadsModel",local:"transformers.GPT2DoubleHeadsModel",headingTag:"h2"}}),st=new U({props:{name:"class transformers.GPT2DoubleHeadsModel",anchor:"transformers.GPT2DoubleHeadsModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPT2DoubleHeadsModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel">GPT2DoubleHeadsModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1128"}}),at=new U({props:{name:"forward",anchor:"transformers.GPT2DoubleHeadsModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"mc_token_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"mc_labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GPT2DoubleHeadsModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.mc_token_ids",description:`<strong>mc_token_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices)</code>, <em>optional</em>, default to index of the last token of the input) &#x2014;
Index of the classification token in each input sequence. Selected in the range <code>[0, input_ids.size(-1) - 1]</code>.`,name:"mc_token_ids"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code>. Indices are selected in <code>[-100, 0, ..., config.vocab_size - 1]</code>. All labels set to
<code>-100</code> are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size - 1]</code>`,name:"labels"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.mc_labels",description:`<strong>mc_labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices]</code>
where <em>num_choices</em> is the size of the second dimension of the input tensors. (see <em>input_ids</em> above)`,name:"mc_labels"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPT2DoubleHeadsModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1178",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput"
>transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config"
>GPT2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>mc_loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>mc_labels</code> is provided) — Multiple choice classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>mc_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — Tuple of length <code>config.n_layers</code>, containing tuples of tensors of shape <code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
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
  href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput"
>transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Me=new de({props:{$$slots:{default:[ca]},$$scope:{ctx:v}}}),ke=new ce({props:{anchor:"transformers.GPT2DoubleHeadsModel.forward.example",$$slots:{default:[pa]},$$scope:{ctx:v}}}),rt=new K({props:{title:"GPT2ForQuestionAnswering",local:"transformers.GPT2ForQuestionAnswering",headingTag:"h2"}}),it=new U({props:{name:"class transformers.GPT2ForQuestionAnswering",anchor:"transformers.GPT2ForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPT2ForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForQuestionAnswering">GPT2ForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1537"}}),lt=new U({props:{name:"forward",anchor:"transformers.GPT2ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GPT2ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPT2ForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1551",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config"
>GPT2Config</a>) and inputs.</p>
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
`}}),ve=new de({props:{$$slots:{default:[ma]},$$scope:{ctx:v}}}),we=new ce({props:{anchor:"transformers.GPT2ForQuestionAnswering.forward.example",$$slots:{default:[ua]},$$scope:{ctx:v}}}),dt=new K({props:{title:"GPT2ForSequenceClassification",local:"transformers.GPT2ForSequenceClassification",headingTag:"h2"}}),ct=new U({props:{name:"class transformers.GPT2ForSequenceClassification",anchor:"transformers.GPT2ForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPT2ForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification">GPT2ForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1317"}}),pt=new U({props:{name:"forward",anchor:"transformers.GPT2ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GPT2ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPT2ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPT2ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPT2ForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPT2ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPT2ForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPT2ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPT2ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.GPT2ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPT2ForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPT2ForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPT2ForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1331",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config"
>GPT2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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


<p><code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),$e=new de({props:{$$slots:{default:[ha]},$$scope:{ctx:v}}}),Ge=new ce({props:{anchor:"transformers.GPT2ForSequenceClassification.forward.example",$$slots:{default:[fa]},$$scope:{ctx:v}}}),Je=new ce({props:{anchor:"transformers.GPT2ForSequenceClassification.forward.example-2",$$slots:{default:[ga]},$$scope:{ctx:v}}}),mt=new K({props:{title:"GPT2ForTokenClassification",local:"transformers.GPT2ForTokenClassification",headingTag:"h2"}}),ut=new U({props:{name:"class transformers.GPT2ForTokenClassification",anchor:"transformers.GPT2ForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPT2ForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2ForTokenClassification">GPT2ForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1442"}}),ht=new U({props:{name:"forward",anchor:"transformers.GPT2ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GPT2ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPT2ForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPT2ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPT2ForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPT2ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPT2ForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPT2ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPT2ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.GPT2ForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPT2ForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPT2ForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPT2ForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt2/modeling_gpt2.py#L1464",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt2#transformers.GPT2Config"
>GPT2Config</a>) and inputs.</p>
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
`}}),je=new de({props:{$$slots:{default:[_a]},$$scope:{ctx:v}}}),Ce=new ce({props:{anchor:"transformers.GPT2ForTokenClassification.forward.example",$$slots:{default:[Ta]},$$scope:{ctx:v}}}),ft=new Ds({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gpt2.md"}}),{c(){t=p("meta"),u=a(),o=p("p"),d=a(),k=p("p"),k.innerHTML=n,h=a(),w=p("div"),w.innerHTML=to,xe=a(),f(ae.$$.fragment),so=a(),Pe=p("p"),Pe.innerHTML=An,ao=a(),ze=p("p"),ze.textContent=Yn,ro=a(),We=p("p"),We.innerHTML=On,io=a(),f(pe.$$.fragment),lo=a(),Ue=p("p"),Ue.innerHTML=Kn,co=a(),f(me.$$.fragment),po=a(),Fe=p("p"),Fe.innerHTML=es,mo=a(),f(Ie.$$.fragment),uo=a(),qe=p("p"),qe.innerHTML=ts,ho=a(),He=p("p"),He.innerHTML=os,fo=a(),f(Ze.$$.fragment),go=a(),f(Be.$$.fragment),_o=a(),Ne=p("ul"),Ne.innerHTML=ns,To=a(),f(Ve.$$.fragment),bo=a(),V=p("div"),f(Re.$$.fragment),No=a(),_t=p("p"),_t.innerHTML=ss,Vo=a(),Tt=p("p"),Tt.innerHTML=as,Ro=a(),f(ue.$$.fragment),yo=a(),f(Le.$$.fragment),Mo=a(),$=p("div"),f(Xe.$$.fragment),Lo=a(),bt=p("p"),bt.textContent=rs,Xo=a(),yt=p("p"),yt.textContent=is,So=a(),f(he.$$.fragment),Qo=a(),Mt=p("p"),Mt.innerHTML=ls,Eo=a(),f(fe.$$.fragment),Do=a(),kt=p("p"),kt.innerHTML=ds,Ao=a(),vt=p("div"),f(Se.$$.fragment),ko=a(),f(Qe.$$.fragment),vo=a(),J=p("div"),f(Ee.$$.fragment),Yo=a(),wt=p("p"),wt.innerHTML=cs,Oo=a(),$t=p("p"),$t.textContent=ps,Ko=a(),f(ge.$$.fragment),en=a(),Gt=p("p"),Gt.innerHTML=ms,tn=a(),f(_e.$$.fragment),on=a(),Jt=p("p"),Jt.innerHTML=us,wo=a(),f(De.$$.fragment),$o=a(),ie=p("div"),f(Ae.$$.fragment),nn=a(),jt=p("p"),jt.textContent=hs,Go=a(),f(Ye.$$.fragment),Jo=a(),F=p("div"),f(Oe.$$.fragment),sn=a(),Ct=p("p"),Ct.textContent=fs,an=a(),xt=p("p"),xt.innerHTML=gs,rn=a(),Pt=p("p"),Pt.innerHTML=_s,ln=a(),re=p("div"),f(Ke.$$.fragment),dn=a(),zt=p("p"),zt.innerHTML=Ts,cn=a(),f(Te.$$.fragment),jo=a(),f(et.$$.fragment),Co=a(),I=p("div"),f(tt.$$.fragment),pn=a(),Wt=p("p"),Wt.textContent=bs,mn=a(),Ut=p("p"),Ut.innerHTML=ys,un=a(),Ft=p("p"),Ft.innerHTML=Ms,hn=a(),D=p("div"),f(ot.$$.fragment),fn=a(),It=p("p"),It.innerHTML=ks,gn=a(),f(be.$$.fragment),_n=a(),f(ye.$$.fragment),xo=a(),f(nt.$$.fragment),Po=a(),q=p("div"),f(st.$$.fragment),Tn=a(),qt=p("p"),qt.textContent=vs,bn=a(),Ht=p("p"),Ht.innerHTML=ws,yn=a(),Zt=p("p"),Zt.innerHTML=$s,Mn=a(),A=p("div"),f(at.$$.fragment),kn=a(),Bt=p("p"),Bt.innerHTML=Gs,vn=a(),f(Me.$$.fragment),wn=a(),f(ke.$$.fragment),zo=a(),f(rt.$$.fragment),Wo=a(),H=p("div"),f(it.$$.fragment),$n=a(),Nt=p("p"),Nt.innerHTML=Js,Gn=a(),Vt=p("p"),Vt.innerHTML=js,Jn=a(),Rt=p("p"),Rt.innerHTML=Cs,jn=a(),Y=p("div"),f(lt.$$.fragment),Cn=a(),Lt=p("p"),Lt.innerHTML=xs,xn=a(),f(ve.$$.fragment),Pn=a(),f(we.$$.fragment),Uo=a(),f(dt.$$.fragment),Fo=a(),j=p("div"),f(ct.$$.fragment),zn=a(),Xt=p("p"),Xt.textContent=Ps,Wn=a(),St=p("p"),St.innerHTML=zs,Un=a(),Qt=p("p"),Qt.innerHTML=Ws,Fn=a(),Et=p("p"),Et.innerHTML=Us,In=a(),Dt=p("p"),Dt.innerHTML=Fs,qn=a(),B=p("div"),f(pt.$$.fragment),Hn=a(),At=p("p"),At.innerHTML=Is,Zn=a(),f($e.$$.fragment),Bn=a(),f(Ge.$$.fragment),Nn=a(),f(Je.$$.fragment),Io=a(),f(mt.$$.fragment),qo=a(),Z=p("div"),f(ut.$$.fragment),Vn=a(),Yt=p("p"),Yt.textContent=qs,Rn=a(),Ot=p("p"),Ot.innerHTML=Hs,Ln=a(),Kt=p("p"),Kt.innerHTML=Zs,Xn=a(),O=p("div"),f(ht.$$.fragment),Sn=a(),eo=p("p"),eo.innerHTML=Bs,Qn=a(),f(je.$$.fragment),En=a(),f(Ce.$$.fragment),Ho=a(),f(ft.$$.fragment),Zo=a(),oo=p("p"),this.h()},l(e){const s=Qs("svelte-u9bgzb",document.head);t=m(s,"META",{name:!0,content:!0}),s.forEach(i),u=r(e),o=m(e,"P",{}),x(o).forEach(i),d=r(e),k=m(e,"P",{"data-svelte-h":!0}),M(k)!=="svelte-18fb9bq"&&(k.innerHTML=n),h=r(e),w=m(e,"DIV",{style:!0,"data-svelte-h":!0}),M(w)!=="svelte-1lhmk4n"&&(w.innerHTML=to),xe=r(e),g(ae.$$.fragment,e),so=r(e),Pe=m(e,"P",{"data-svelte-h":!0}),M(Pe)!=="svelte-11726rr"&&(Pe.innerHTML=An),ao=r(e),ze=m(e,"P",{"data-svelte-h":!0}),M(ze)!=="svelte-2ba78p"&&(ze.textContent=Yn),ro=r(e),We=m(e,"P",{"data-svelte-h":!0}),M(We)!=="svelte-113go8s"&&(We.innerHTML=On),io=r(e),g(pe.$$.fragment,e),lo=r(e),Ue=m(e,"P",{"data-svelte-h":!0}),M(Ue)!=="svelte-x9rs6r"&&(Ue.innerHTML=Kn),co=r(e),g(me.$$.fragment,e),po=r(e),Fe=m(e,"P",{"data-svelte-h":!0}),M(Fe)!=="svelte-1dowlbd"&&(Fe.innerHTML=es),mo=r(e),g(Ie.$$.fragment,e),uo=r(e),qe=m(e,"P",{"data-svelte-h":!0}),M(qe)!=="svelte-nf5ooi"&&(qe.innerHTML=ts),ho=r(e),He=m(e,"P",{"data-svelte-h":!0}),M(He)!=="svelte-60nsd0"&&(He.innerHTML=os),fo=r(e),g(Ze.$$.fragment,e),go=r(e),g(Be.$$.fragment,e),_o=r(e),Ne=m(e,"UL",{"data-svelte-h":!0}),M(Ne)!=="svelte-zz8igk"&&(Ne.innerHTML=ns),To=r(e),g(Ve.$$.fragment,e),bo=r(e),V=m(e,"DIV",{class:!0});var ee=x(V);g(Re.$$.fragment,ee),No=r(ee),_t=m(ee,"P",{"data-svelte-h":!0}),M(_t)!=="svelte-bg8afx"&&(_t.innerHTML=ss),Vo=r(ee),Tt=m(ee,"P",{"data-svelte-h":!0}),M(Tt)!=="svelte-1ek1ss9"&&(Tt.innerHTML=as),Ro=r(ee),g(ue.$$.fragment,ee),ee.forEach(i),yo=r(e),g(Le.$$.fragment,e),Mo=r(e),$=m(e,"DIV",{class:!0});var C=x($);g(Xe.$$.fragment,C),Lo=r(C),bt=m(C,"P",{"data-svelte-h":!0}),M(bt)!=="svelte-1tnpexf"&&(bt.textContent=rs),Xo=r(C),yt=m(C,"P",{"data-svelte-h":!0}),M(yt)!=="svelte-1s077p3"&&(yt.textContent=is),So=r(C),g(he.$$.fragment,C),Qo=r(C),Mt=m(C,"P",{"data-svelte-h":!0}),M(Mt)!=="svelte-1jfcabo"&&(Mt.innerHTML=ls),Eo=r(C),g(fe.$$.fragment,C),Do=r(C),kt=m(C,"P",{"data-svelte-h":!0}),M(kt)!=="svelte-ntrhio"&&(kt.innerHTML=ds),Ao=r(C),vt=m(C,"DIV",{class:!0});var no=x(vt);g(Se.$$.fragment,no),no.forEach(i),C.forEach(i),ko=r(e),g(Qe.$$.fragment,e),vo=r(e),J=m(e,"DIV",{class:!0});var z=x(J);g(Ee.$$.fragment,z),Yo=r(z),wt=m(z,"P",{"data-svelte-h":!0}),M(wt)!=="svelte-115toia"&&(wt.innerHTML=cs),Oo=r(z),$t=m(z,"P",{"data-svelte-h":!0}),M($t)!=="svelte-1s077p3"&&($t.textContent=ps),Ko=r(z),g(ge.$$.fragment,z),en=r(z),Gt=m(z,"P",{"data-svelte-h":!0}),M(Gt)!=="svelte-1afeqmz"&&(Gt.innerHTML=ms),tn=r(z),g(_e.$$.fragment,z),on=r(z),Jt=m(z,"P",{"data-svelte-h":!0}),M(Jt)!=="svelte-gxzj9w"&&(Jt.innerHTML=us),z.forEach(i),wo=r(e),g(De.$$.fragment,e),$o=r(e),ie=m(e,"DIV",{class:!0});var gt=x(ie);g(Ae.$$.fragment,gt),nn=r(gt),jt=m(gt,"P",{"data-svelte-h":!0}),M(jt)!=="svelte-1bm2i0r"&&(jt.textContent=hs),gt.forEach(i),Go=r(e),g(Ye.$$.fragment,e),Jo=r(e),F=m(e,"DIV",{class:!0});var R=x(F);g(Oe.$$.fragment,R),sn=r(R),Ct=m(R,"P",{"data-svelte-h":!0}),M(Ct)!=="svelte-1ai4d1p"&&(Ct.textContent=fs),an=r(R),xt=m(R,"P",{"data-svelte-h":!0}),M(xt)!=="svelte-q52n56"&&(xt.innerHTML=gs),rn=r(R),Pt=m(R,"P",{"data-svelte-h":!0}),M(Pt)!=="svelte-hswkmf"&&(Pt.innerHTML=_s),ln=r(R),re=m(R,"DIV",{class:!0});var le=x(re);g(Ke.$$.fragment,le),dn=r(le),zt=m(le,"P",{"data-svelte-h":!0}),M(zt)!=="svelte-layrqy"&&(zt.innerHTML=Ts),cn=r(le),g(Te.$$.fragment,le),le.forEach(i),R.forEach(i),jo=r(e),g(et.$$.fragment,e),Co=r(e),I=m(e,"DIV",{class:!0});var L=x(I);g(tt.$$.fragment,L),pn=r(L),Wt=m(L,"P",{"data-svelte-h":!0}),M(Wt)!=="svelte-jjb0s"&&(Wt.textContent=bs),mn=r(L),Ut=m(L,"P",{"data-svelte-h":!0}),M(Ut)!=="svelte-q52n56"&&(Ut.innerHTML=ys),un=r(L),Ft=m(L,"P",{"data-svelte-h":!0}),M(Ft)!=="svelte-hswkmf"&&(Ft.innerHTML=Ms),hn=r(L),D=m(L,"DIV",{class:!0});var te=x(D);g(ot.$$.fragment,te),fn=r(te),It=m(te,"P",{"data-svelte-h":!0}),M(It)!=="svelte-86j0yi"&&(It.innerHTML=ks),gn=r(te),g(be.$$.fragment,te),_n=r(te),g(ye.$$.fragment,te),te.forEach(i),L.forEach(i),xo=r(e),g(nt.$$.fragment,e),Po=r(e),q=m(e,"DIV",{class:!0});var X=x(q);g(st.$$.fragment,X),Tn=r(X),qt=m(X,"P",{"data-svelte-h":!0}),M(qt)!=="svelte-1q8023u"&&(qt.textContent=vs),bn=r(X),Ht=m(X,"P",{"data-svelte-h":!0}),M(Ht)!=="svelte-q52n56"&&(Ht.innerHTML=ws),yn=r(X),Zt=m(X,"P",{"data-svelte-h":!0}),M(Zt)!=="svelte-hswkmf"&&(Zt.innerHTML=$s),Mn=r(X),A=m(X,"DIV",{class:!0});var oe=x(A);g(at.$$.fragment,oe),kn=r(oe),Bt=m(oe,"P",{"data-svelte-h":!0}),M(Bt)!=="svelte-1nq1z9u"&&(Bt.innerHTML=Gs),vn=r(oe),g(Me.$$.fragment,oe),wn=r(oe),g(ke.$$.fragment,oe),oe.forEach(i),X.forEach(i),zo=r(e),g(rt.$$.fragment,e),Wo=r(e),H=m(e,"DIV",{class:!0});var S=x(H);g(it.$$.fragment,S),$n=r(S),Nt=m(S,"P",{"data-svelte-h":!0}),M(Nt)!=="svelte-1oclafy"&&(Nt.innerHTML=Js),Gn=r(S),Vt=m(S,"P",{"data-svelte-h":!0}),M(Vt)!=="svelte-q52n56"&&(Vt.innerHTML=js),Jn=r(S),Rt=m(S,"P",{"data-svelte-h":!0}),M(Rt)!=="svelte-hswkmf"&&(Rt.innerHTML=Cs),jn=r(S),Y=m(S,"DIV",{class:!0});var ne=x(Y);g(lt.$$.fragment,ne),Cn=r(ne),Lt=m(ne,"P",{"data-svelte-h":!0}),M(Lt)!=="svelte-m86pxi"&&(Lt.innerHTML=xs),xn=r(ne),g(ve.$$.fragment,ne),Pn=r(ne),g(we.$$.fragment,ne),ne.forEach(i),S.forEach(i),Uo=r(e),g(dt.$$.fragment,e),Fo=r(e),j=m(e,"DIV",{class:!0});var W=x(j);g(ct.$$.fragment,W),zn=r(W),Xt=m(W,"P",{"data-svelte-h":!0}),M(Xt)!=="svelte-1o27avp"&&(Xt.textContent=Ps),Wn=r(W),St=m(W,"P",{"data-svelte-h":!0}),M(St)!=="svelte-j8s62s"&&(St.innerHTML=zs),Un=r(W),Qt=m(W,"P",{"data-svelte-h":!0}),M(Qt)!=="svelte-10ugs3m"&&(Qt.innerHTML=Ws),Fn=r(W),Et=m(W,"P",{"data-svelte-h":!0}),M(Et)!=="svelte-q52n56"&&(Et.innerHTML=Us),In=r(W),Dt=m(W,"P",{"data-svelte-h":!0}),M(Dt)!=="svelte-hswkmf"&&(Dt.innerHTML=Fs),qn=r(W),B=m(W,"DIV",{class:!0});var Q=x(B);g(pt.$$.fragment,Q),Hn=r(Q),At=m(Q,"P",{"data-svelte-h":!0}),M(At)!=="svelte-2h436e"&&(At.innerHTML=Is),Zn=r(Q),g($e.$$.fragment,Q),Bn=r(Q),g(Ge.$$.fragment,Q),Nn=r(Q),g(Je.$$.fragment,Q),Q.forEach(i),W.forEach(i),Io=r(e),g(mt.$$.fragment,e),qo=r(e),Z=m(e,"DIV",{class:!0});var E=x(Z);g(ut.$$.fragment,E),Vn=r(E),Yt=m(E,"P",{"data-svelte-h":!0}),M(Yt)!=="svelte-1yoz5oj"&&(Yt.textContent=qs),Rn=r(E),Ot=m(E,"P",{"data-svelte-h":!0}),M(Ot)!=="svelte-q52n56"&&(Ot.innerHTML=Hs),Ln=r(E),Kt=m(E,"P",{"data-svelte-h":!0}),M(Kt)!=="svelte-hswkmf"&&(Kt.innerHTML=Zs),Xn=r(E),O=m(E,"DIV",{class:!0});var se=x(O);g(ht.$$.fragment,se),Sn=r(se),eo=m(se,"P",{"data-svelte-h":!0}),M(eo)!=="svelte-z2g27g"&&(eo.innerHTML=Bs),Qn=r(se),g(je.$$.fragment,se),En=r(se),g(Ce.$$.fragment,se),se.forEach(i),E.forEach(i),Ho=r(e),g(ft.$$.fragment,e),Zo=r(e),oo=m(e,"P",{}),x(oo).forEach(i),this.h()},h(){P(t,"name","hf:doc:metadata"),P(t,"content",ya),Es(w,"float","right"),P(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){l(document.head,t),c(e,u,s),c(e,o,s),c(e,d,s),c(e,k,s),c(e,h,s),c(e,w,s),c(e,xe,s),_(ae,e,s),c(e,so,s),c(e,Pe,s),c(e,ao,s),c(e,ze,s),c(e,ro,s),c(e,We,s),c(e,io,s),_(pe,e,s),c(e,lo,s),c(e,Ue,s),c(e,co,s),_(me,e,s),c(e,po,s),c(e,Fe,s),c(e,mo,s),_(Ie,e,s),c(e,uo,s),c(e,qe,s),c(e,ho,s),c(e,He,s),c(e,fo,s),_(Ze,e,s),c(e,go,s),_(Be,e,s),c(e,_o,s),c(e,Ne,s),c(e,To,s),_(Ve,e,s),c(e,bo,s),c(e,V,s),_(Re,V,null),l(V,No),l(V,_t),l(V,Vo),l(V,Tt),l(V,Ro),_(ue,V,null),c(e,yo,s),_(Le,e,s),c(e,Mo,s),c(e,$,s),_(Xe,$,null),l($,Lo),l($,bt),l($,Xo),l($,yt),l($,So),_(he,$,null),l($,Qo),l($,Mt),l($,Eo),_(fe,$,null),l($,Do),l($,kt),l($,Ao),l($,vt),_(Se,vt,null),c(e,ko,s),_(Qe,e,s),c(e,vo,s),c(e,J,s),_(Ee,J,null),l(J,Yo),l(J,wt),l(J,Oo),l(J,$t),l(J,Ko),_(ge,J,null),l(J,en),l(J,Gt),l(J,tn),_(_e,J,null),l(J,on),l(J,Jt),c(e,wo,s),_(De,e,s),c(e,$o,s),c(e,ie,s),_(Ae,ie,null),l(ie,nn),l(ie,jt),c(e,Go,s),_(Ye,e,s),c(e,Jo,s),c(e,F,s),_(Oe,F,null),l(F,sn),l(F,Ct),l(F,an),l(F,xt),l(F,rn),l(F,Pt),l(F,ln),l(F,re),_(Ke,re,null),l(re,dn),l(re,zt),l(re,cn),_(Te,re,null),c(e,jo,s),_(et,e,s),c(e,Co,s),c(e,I,s),_(tt,I,null),l(I,pn),l(I,Wt),l(I,mn),l(I,Ut),l(I,un),l(I,Ft),l(I,hn),l(I,D),_(ot,D,null),l(D,fn),l(D,It),l(D,gn),_(be,D,null),l(D,_n),_(ye,D,null),c(e,xo,s),_(nt,e,s),c(e,Po,s),c(e,q,s),_(st,q,null),l(q,Tn),l(q,qt),l(q,bn),l(q,Ht),l(q,yn),l(q,Zt),l(q,Mn),l(q,A),_(at,A,null),l(A,kn),l(A,Bt),l(A,vn),_(Me,A,null),l(A,wn),_(ke,A,null),c(e,zo,s),_(rt,e,s),c(e,Wo,s),c(e,H,s),_(it,H,null),l(H,$n),l(H,Nt),l(H,Gn),l(H,Vt),l(H,Jn),l(H,Rt),l(H,jn),l(H,Y),_(lt,Y,null),l(Y,Cn),l(Y,Lt),l(Y,xn),_(ve,Y,null),l(Y,Pn),_(we,Y,null),c(e,Uo,s),_(dt,e,s),c(e,Fo,s),c(e,j,s),_(ct,j,null),l(j,zn),l(j,Xt),l(j,Wn),l(j,St),l(j,Un),l(j,Qt),l(j,Fn),l(j,Et),l(j,In),l(j,Dt),l(j,qn),l(j,B),_(pt,B,null),l(B,Hn),l(B,At),l(B,Zn),_($e,B,null),l(B,Bn),_(Ge,B,null),l(B,Nn),_(Je,B,null),c(e,Io,s),_(mt,e,s),c(e,qo,s),c(e,Z,s),_(ut,Z,null),l(Z,Vn),l(Z,Yt),l(Z,Rn),l(Z,Ot),l(Z,Ln),l(Z,Kt),l(Z,Xn),l(Z,O),_(ht,O,null),l(O,Sn),l(O,eo),l(O,Qn),_(je,O,null),l(O,En),_(Ce,O,null),c(e,Ho,s),_(ft,e,s),c(e,Zo,s),c(e,oo,s),Bo=!0},p(e,[s]){const ee={};s&2&&(ee.$$scope={dirty:s,ctx:e}),pe.$set(ee);const C={};s&2&&(C.$$scope={dirty:s,ctx:e}),me.$set(C);const no={};s&2&&(no.$$scope={dirty:s,ctx:e}),ue.$set(no);const z={};s&2&&(z.$$scope={dirty:s,ctx:e}),he.$set(z);const gt={};s&2&&(gt.$$scope={dirty:s,ctx:e}),fe.$set(gt);const R={};s&2&&(R.$$scope={dirty:s,ctx:e}),ge.$set(R);const le={};s&2&&(le.$$scope={dirty:s,ctx:e}),_e.$set(le);const L={};s&2&&(L.$$scope={dirty:s,ctx:e}),Te.$set(L);const te={};s&2&&(te.$$scope={dirty:s,ctx:e}),be.$set(te);const X={};s&2&&(X.$$scope={dirty:s,ctx:e}),ye.$set(X);const oe={};s&2&&(oe.$$scope={dirty:s,ctx:e}),Me.$set(oe);const S={};s&2&&(S.$$scope={dirty:s,ctx:e}),ke.$set(S);const ne={};s&2&&(ne.$$scope={dirty:s,ctx:e}),ve.$set(ne);const W={};s&2&&(W.$$scope={dirty:s,ctx:e}),we.$set(W);const Q={};s&2&&(Q.$$scope={dirty:s,ctx:e}),$e.$set(Q);const E={};s&2&&(E.$$scope={dirty:s,ctx:e}),Ge.$set(E);const se={};s&2&&(se.$$scope={dirty:s,ctx:e}),Je.$set(se);const Ns={};s&2&&(Ns.$$scope={dirty:s,ctx:e}),je.$set(Ns);const Vs={};s&2&&(Vs.$$scope={dirty:s,ctx:e}),Ce.$set(Vs)},i(e){Bo||(T(ae.$$.fragment,e),T(pe.$$.fragment,e),T(me.$$.fragment,e),T(Ie.$$.fragment,e),T(Ze.$$.fragment,e),T(Be.$$.fragment,e),T(Ve.$$.fragment,e),T(Re.$$.fragment,e),T(ue.$$.fragment,e),T(Le.$$.fragment,e),T(Xe.$$.fragment,e),T(he.$$.fragment,e),T(fe.$$.fragment,e),T(Se.$$.fragment,e),T(Qe.$$.fragment,e),T(Ee.$$.fragment,e),T(ge.$$.fragment,e),T(_e.$$.fragment,e),T(De.$$.fragment,e),T(Ae.$$.fragment,e),T(Ye.$$.fragment,e),T(Oe.$$.fragment,e),T(Ke.$$.fragment,e),T(Te.$$.fragment,e),T(et.$$.fragment,e),T(tt.$$.fragment,e),T(ot.$$.fragment,e),T(be.$$.fragment,e),T(ye.$$.fragment,e),T(nt.$$.fragment,e),T(st.$$.fragment,e),T(at.$$.fragment,e),T(Me.$$.fragment,e),T(ke.$$.fragment,e),T(rt.$$.fragment,e),T(it.$$.fragment,e),T(lt.$$.fragment,e),T(ve.$$.fragment,e),T(we.$$.fragment,e),T(dt.$$.fragment,e),T(ct.$$.fragment,e),T(pt.$$.fragment,e),T($e.$$.fragment,e),T(Ge.$$.fragment,e),T(Je.$$.fragment,e),T(mt.$$.fragment,e),T(ut.$$.fragment,e),T(ht.$$.fragment,e),T(je.$$.fragment,e),T(Ce.$$.fragment,e),T(ft.$$.fragment,e),Bo=!0)},o(e){b(ae.$$.fragment,e),b(pe.$$.fragment,e),b(me.$$.fragment,e),b(Ie.$$.fragment,e),b(Ze.$$.fragment,e),b(Be.$$.fragment,e),b(Ve.$$.fragment,e),b(Re.$$.fragment,e),b(ue.$$.fragment,e),b(Le.$$.fragment,e),b(Xe.$$.fragment,e),b(he.$$.fragment,e),b(fe.$$.fragment,e),b(Se.$$.fragment,e),b(Qe.$$.fragment,e),b(Ee.$$.fragment,e),b(ge.$$.fragment,e),b(_e.$$.fragment,e),b(De.$$.fragment,e),b(Ae.$$.fragment,e),b(Ye.$$.fragment,e),b(Oe.$$.fragment,e),b(Ke.$$.fragment,e),b(Te.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(ot.$$.fragment,e),b(be.$$.fragment,e),b(ye.$$.fragment,e),b(nt.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b(Me.$$.fragment,e),b(ke.$$.fragment,e),b(rt.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(ve.$$.fragment,e),b(we.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b($e.$$.fragment,e),b(Ge.$$.fragment,e),b(Je.$$.fragment,e),b(mt.$$.fragment,e),b(ut.$$.fragment,e),b(ht.$$.fragment,e),b(je.$$.fragment,e),b(Ce.$$.fragment,e),b(ft.$$.fragment,e),Bo=!1},d(e){e&&(i(u),i(o),i(d),i(k),i(h),i(w),i(xe),i(so),i(Pe),i(ao),i(ze),i(ro),i(We),i(io),i(lo),i(Ue),i(co),i(po),i(Fe),i(mo),i(uo),i(qe),i(ho),i(He),i(fo),i(go),i(_o),i(Ne),i(To),i(bo),i(V),i(yo),i(Mo),i($),i(ko),i(vo),i(J),i(wo),i($o),i(ie),i(Go),i(Jo),i(F),i(jo),i(Co),i(I),i(xo),i(Po),i(q),i(zo),i(Wo),i(H),i(Uo),i(Fo),i(j),i(Io),i(qo),i(Z),i(Ho),i(Zo),i(oo)),i(t),y(ae,e),y(pe,e),y(me,e),y(Ie,e),y(Ze,e),y(Be,e),y(Ve,e),y(Re),y(ue),y(Le,e),y(Xe),y(he),y(fe),y(Se),y(Qe,e),y(Ee),y(ge),y(_e),y(De,e),y(Ae),y(Ye,e),y(Oe),y(Ke),y(Te),y(et,e),y(tt),y(ot),y(be),y(ye),y(nt,e),y(st),y(at),y(Me),y(ke),y(rt,e),y(it),y(lt),y(ve),y(we),y(dt,e),y(ct),y(pt),y($e),y(Ge),y(Je),y(mt,e),y(ut),y(ht),y(je),y(Ce),y(ft,e)}}}const ya='{"title":"GPT-2","local":"gpt-2","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"GPT2Config","local":"transformers.GPT2Config","sections":[],"depth":2},{"title":"GPT2Tokenizer","local":"transformers.GPT2Tokenizer","sections":[],"depth":2},{"title":"GPT2TokenizerFast","local":"transformers.GPT2TokenizerFast","sections":[],"depth":2},{"title":"GPT2 specific outputs","local":"transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput","sections":[],"depth":2},{"title":"GPT2Model","local":"transformers.GPT2Model","sections":[],"depth":2},{"title":"GPT2LMHeadModel","local":"transformers.GPT2LMHeadModel","sections":[],"depth":2},{"title":"GPT2DoubleHeadsModel","local":"transformers.GPT2DoubleHeadsModel","sections":[],"depth":2},{"title":"GPT2ForQuestionAnswering","local":"transformers.GPT2ForQuestionAnswering","sections":[],"depth":2},{"title":"GPT2ForSequenceClassification","local":"transformers.GPT2ForSequenceClassification","sections":[],"depth":2},{"title":"GPT2ForTokenClassification","local":"transformers.GPT2ForTokenClassification","sections":[],"depth":2}],"depth":1}';function Ma(v){return Ls(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class xa extends Xs{constructor(t){super(),Ss(this,t,Ma,ba,Rs,{})}}export{xa as component};
