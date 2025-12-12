import{s as dr,o as cr,n as R}from"../chunks/scheduler.18a86fab.js";import{S as mr,i as pr,g as c,s as r,r as f,A as hr,h as m,f as d,c as a,j as v,x as M,u as g,k as F,l as ur,y as s,a as p,v as _,d as b,t as y,w as T}from"../chunks/index.98837b22.js";import{T as qe}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as Y}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as pe}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as te,E as fr}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as gr,a as fs}from"../chunks/HfOption.6641485e.js";function _r(w){let t,h="Click on the RoFormer models in the right sidebar for more examples of how to apply RoFormer to different language tasks.";return{c(){t=c("p"),t.textContent=h},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-174ylnb"&&(t.textContent=h)},m(n,i){p(n,t,i)},p:R,d(n){n&&d(t)}}}function br(w){let t,h;return t=new Y({props:{code:"JTIzJTIwdW5jb21tZW50JTIwdG8lMjBpbnN0YWxsJTIwcmppZWJhJTIwd2hpY2glMjBpcyUyMG5lZWRlZCUyMGZvciUyMHRoZSUyMHRva2VuaXplciUwQSUyMyUyMCFwaXAlMjBpbnN0YWxsJTIwcmppZWJhJTBBaW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMmZpbGwtbWFzayUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUwQSklMEFvdXRwdXQlMjAlM0QlMjBwaXBlKCUyMiVFNiVCMCVCNCVFNSU5QyVBOCVFOSU5QiVCNiVFNSVCQSVBNiVFNiU5NyVCNiVFNCVCQyU5QSU1Qk1BU0slNUQlMjIpJTBBcHJpbnQob3V0cHV0KQ==",highlighted:`<span class="hljs-comment"># uncomment to install rjieba which is needed for the tokenizer</span>
<span class="hljs-comment"># !pip install rjieba</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    task=<span class="hljs-string">&quot;fill-mask&quot;</span>,
    model=<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
output = pipe(<span class="hljs-string">&quot;水在零度时会[MASK]&quot;</span>)
<span class="hljs-built_in">print</span>(output)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){_(t,n,i),h=!0},p:R,i(n){h||(b(t.$$.fragment,n),h=!0)},o(n){y(t.$$.fragment,n),h=!1},d(n){T(t,n)}}}function yr(w){let t,h;return t=new Y({props:{code:"JTIzJTIwdW5jb21tZW50JTIwdG8lMjBpbnN0YWxsJTIwcmppZWJhJTIwd2hpY2glMjBpcyUyMG5lZWRlZCUyMGZvciUyMHRoZSUyMHRva2VuaXplciUwQSUyMyUyMCFwaXAlMjBpbnN0YWxsJTIwcmppZWJhJTBBaW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIyJTJDJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTBBKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMiklMEElMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyJUU2JUIwJUI0JUU1JTlDJUE4JUU5JTlCJUI2JUU1JUJBJUE2JUU2JTk3JUI2JUU0JUJDJTlBJTVCTUFTSyU1RCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dF9pZHMpJTBBZGVjb2RlZCUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUob3V0cHV0cy5sb2dpdHMuYXJnbWF4KC0xKSUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQXByaW50KGRlY29kZWQp",highlighted:`<span class="hljs-comment"># uncomment to install rjieba which is needed for the tokenizer</span>
<span class="hljs-comment"># !pip install rjieba</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>, dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)

input_ids = tokenizer(<span class="hljs-string">&quot;水在零度时会[MASK]&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model(**input_ids)
decoded = tokenizer.batch_decode(outputs.logits.argmax(-<span class="hljs-number">1</span>), skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(decoded)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){_(t,n,i),h=!0},p:R,i(n){h||(b(t.$$.fragment,n),h=!0)},o(n){y(t.$$.fragment,n),h=!1},d(n){T(t,n)}}}function Tr(w){let t,h;return t=new Y({props:{code:"ZWNobyUyMC1lJTIwJTIyJUU2JUIwJUI0JUU1JTlDJUE4JUU5JTlCJUI2JUU1JUJBJUE2JUU2JTk3JUI2JUU0JUJDJTlBJTVCTUFTSyU1RCUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycy1jbGklMjBydW4lMjAtLXRhc2slMjBmaWxsLW1hc2slMjAtLW1vZGVsJTIwanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIwLS1kZXZpY2UlMjAw",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;水在零度时会[MASK]&quot;</span> | transformers-cli run --task fill-mask --model junnyu/roformer_chinese_base --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){_(t,n,i),h=!0},p:R,i(n){h||(b(t.$$.fragment,n),h=!0)},o(n){y(t.$$.fragment,n),h=!1},d(n){T(t,n)}}}function Mr(w){let t,h,n,i,k,o;return t=new fs({props:{id:"usage",option:"Pipeline",$$slots:{default:[br]},$$scope:{ctx:w}}}),n=new fs({props:{id:"usage",option:"AutoModel",$$slots:{default:[yr]},$$scope:{ctx:w}}}),k=new fs({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Tr]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),h=r(),f(n.$$.fragment),i=r(),f(k.$$.fragment)},l(u){g(t.$$.fragment,u),h=a(u),g(n.$$.fragment,u),i=a(u),g(k.$$.fragment,u)},m(u,J){_(t,u,J),p(u,h,J),_(n,u,J),p(u,i,J),_(k,u,J),o=!0},p(u,J){const _o={};J&2&&(_o.$$scope={dirty:J,ctx:u}),t.$set(_o);const Ne={};J&2&&(Ne.$$scope={dirty:J,ctx:u}),n.$set(Ne);const ae={};J&2&&(ae.$$scope={dirty:J,ctx:u}),k.$set(ae)},i(u){o||(b(t.$$.fragment,u),b(n.$$.fragment,u),b(k.$$.fragment,u),o=!0)},o(u){y(t.$$.fragment,u),y(n.$$.fragment,u),y(k.$$.fragment,u),o=!1},d(u){u&&(d(h),d(i)),T(t,u),T(n,u),T(k,u)}}}function kr(w){let t,h="Example:",n,i,k;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJvRm9ybWVyTW9kZWwlMkMlMjBSb0Zvcm1lckNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBSb0Zvcm1lciUyMGp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBSb0Zvcm1lckNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBqdW5ueXUlMkZyb2Zvcm1lcl9jaGluZXNlX2Jhc2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMFJvRm9ybWVyTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RoFormerModel, RoFormerConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a RoFormer junnyu/roformer_chinese_base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = RoFormerConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the junnyu/roformer_chinese_base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function wr(w){let t,h="Example:",n,i,k;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJvRm9ybWVyVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwUm9Gb3JtZXJUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMiklMEF0b2tlbml6ZXIudG9rZW5pemUoJTIyJUU0JUJCJThBJUU1JUE0JUE5JUU1JUE0JUE5JUU2JUIwJTk0JUU5JTlEJTlFJUU1JUI4JUI4JUU1JUE1JUJEJUUzJTgwJTgyJTIyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RoFormerTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = RoFormerTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.tokenize(<span class="hljs-string">&quot;今天天气非常好。&quot;</span>)
[<span class="hljs-string">&#x27;今&#x27;</span>, <span class="hljs-string">&#x27;天&#x27;</span>, <span class="hljs-string">&#x27;天&#x27;</span>, <span class="hljs-string">&#x27;气&#x27;</span>, <span class="hljs-string">&#x27;非常&#x27;</span>, <span class="hljs-string">&#x27;好&#x27;</span>, <span class="hljs-string">&#x27;。&#x27;</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function vr(w){let t,h="Example:",n,i,k;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJvRm9ybWVyVG9rZW5pemVyRmFzdCUwQSUwQXRva2VuaXplciUyMCUzRCUyMFJvRm9ybWVyVG9rZW5pemVyRmFzdC5mcm9tX3ByZXRyYWluZWQoJTIyanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIyKSUwQXRva2VuaXplci50b2tlbml6ZSglMjIlRTQlQkIlOEElRTUlQTQlQTklRTUlQTQlQTklRTYlQjAlOTQlRTklOUQlOUUlRTUlQjglQjglRTUlQTUlQkQlRTMlODAlODIlMjIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RoFormerTokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = RoFormerTokenizerFast.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.tokenize(<span class="hljs-string">&quot;今天天气非常好。&quot;</span>)
[<span class="hljs-string">&#x27;今&#x27;</span>, <span class="hljs-string">&#x27;天&#x27;</span>, <span class="hljs-string">&#x27;天&#x27;</span>, <span class="hljs-string">&#x27;气&#x27;</span>, <span class="hljs-string">&#x27;非常&#x27;</span>, <span class="hljs-string">&#x27;好&#x27;</span>, <span class="hljs-string">&#x27;。&#x27;</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function Fr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(n,i){p(n,t,i)},p:R,d(n){n&&d(t)}}}function $r(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(n,i){p(n,t,i)},p:R,d(n){n&&d(t)}}}function Jr(w){let t,h="Example:",n,i,k;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0Zvcm1lckZvckNhdXNhbExNJTJDJTIwUm9Gb3JtZXJDb25maWclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMiklMEFjb25maWclMjAlM0QlMjBSb0Zvcm1lckNvbmZpZy5mcm9tX3ByZXRyYWluZWQoJTIyanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIyKSUwQWNvbmZpZy5pc19kZWNvZGVyJTIwJTNEJTIwVHJ1ZSUwQW1vZGVsJTIwJTNEJTIwUm9Gb3JtZXJGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIyJTJDJTIwY29uZmlnJTNEY29uZmlnKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjIlRTQlQkIlOEElRTUlQTQlQTklRTUlQTQlQTklRTYlQjAlOTQlRTklOUQlOUUlRTUlQjglQjglRTUlQTUlQkQlRTMlODAlODIlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoFormerForCausalLM, RoFormerConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = RoFormerConfig.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForCausalLM.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;今天天气非常好。&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function Ur(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(n,i){p(n,t,i)},p:R,d(n){n&&d(t)}}}function Rr(w){let t,h="Example:",n,i,k;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0Zvcm1lckZvck1hc2tlZExNJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJqdW5ueXUlMkZyb2Zvcm1lcl9jaGluZXNlX2Jhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb0Zvcm1lckZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJqdW5ueXUlMkZyb2Zvcm1lcl9jaGluZXNlX2Jhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoFormerForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForMaskedLM.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function jr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(n,i){p(n,t,i)},p:R,d(n){n&&d(t)}}}function Cr(w){let t,h="Example of single-label classification:",n,i,k;return i=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFJvRm9ybWVyRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMFJvRm9ybWVyRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBSb0Zvcm1lckZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoFormerForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-ykxpe4"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function zr(w){let t,h="Example of multi-label classification:",n,i,k;return i=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFJvRm9ybWVyRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMFJvRm9ybWVyRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBSb0Zvcm1lckZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoFormerForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-1l8e32d"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function xr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(n,i){p(n,t,i)},p:R,d(n){n&&d(t)}}}function Zr(w){let t,h="Example:",n,i,k;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0Zvcm1lckZvck11bHRpcGxlQ2hvaWNlJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJqdW5ueXUlMkZyb2Zvcm1lcl9jaGluZXNlX2Jhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb0Zvcm1lckZvck11bHRpcGxlQ2hvaWNlLmZyb21fcHJldHJhaW5lZCglMjJqdW5ueXUlMkZyb2Zvcm1lcl9jaGluZXNlX2Jhc2UlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoFormerForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function Wr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(n,i){p(n,t,i)},p:R,d(n){n&&d(t)}}}function Ir(w){let t,h="Example:",n,i,k;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0Zvcm1lckZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmp1bm55dSUyRnJvZm9ybWVyX2NoaW5lc2VfYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMFJvRm9ybWVyRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyanVubnl1JTJGcm9mb3JtZXJfY2hpbmVzZV9iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoFormerForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForTokenClassification.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function Br(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(n,i){p(n,t,i)},p:R,d(n){n&&d(t)}}}function Vr(w){let t,h="Example:",n,i,k;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0Zvcm1lckZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJqdW5ueXUlMkZyb2Zvcm1lcl9jaGluZXNlX2Jhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb0Zvcm1lckZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJqdW5ueXUlMkZyb2Zvcm1lcl9jaGluZXNlX2Jhc2UlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoFormerForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoFormerForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;junnyu/roformer_chinese_base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),n=a(o),g(i.$$.fragment,o)},m(o,u){p(o,t,u),p(o,n,u),_(i,o,u),k=!0},p:R,i(o){k||(b(i.$$.fragment,o),k=!0)},o(o){y(i.$$.fragment,o),k=!1},d(o){o&&(d(t),d(n)),T(i,o)}}}function qr(w){let t,h,n,i,k,o="<em>This model was released on 2021-04-20 and added to Hugging Face Transformers on 2021-05-20.</em>",u,J,_o='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ne,ae,To,Le,gs='<a href="https://huggingface.co/papers/2104.09864" rel="nofollow">RoFormer</a> introduces Rotary Position Embedding (RoPE) to encode token positions by rotating the inputs in 2D space. This allows a model to track absolute positions and model relative relationships. RoPE can scale to longer sequences, account for the natural decay of token dependencies, and works with the more efficient linear self-attention.',Mo,Ge,_s='You can find all the RoFormer checkpoints on the <a href="https://huggingface.co/models?search=roformer" rel="nofollow">Hub</a>.',ko,_e,wo,Xe,bs='The example below demonstrates how to predict the <code>[MASK]</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',vo,be,Fo,Se,$o,He,ys='<li>The current RoFormer implementation is an encoder-only model. The original code can be found in the <a href="https://github.com/ZhuiyiTechnology/roformer" rel="nofollow">ZhuiyiTechnology/roformer</a> repository.</li>',Jo,Qe,Uo,G,Ee,Po,$t,Ts=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerModel">RoFormerModel</a>. It is used to instantiate an
RoFormer model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the RoFormer
<a href="https://huggingface.co/junnyu/roformer_chinese_base" rel="nofollow">junnyu/roformer_chinese_base</a> architecture.`,Oo,Jt,Ms=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Do,ye,Ro,Ae,jo,U,Ye,Ko,Ut,ks='Construct a RoFormer tokenizer. Based on <a href="https://pypi.org/project/rjieba/" rel="nofollow">Rust Jieba</a>.',en,Rt,ws=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,tn,Te,on,ie,Pe,nn,jt,vs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A RoFormer sequence has the following format:`,sn,Ct,Fs="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",rn,Me,Oe,an,zt,$s=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,ln,le,De,dn,xt,Js=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,cn,Zt,Us="Should be overridden in a subclass if the model has a special way of building those.",mn,Wt,Ke,Co,et,zo,C,tt,pn,It,Rs="Construct a “fast” RoFormer tokenizer (backed by HuggingFace’s <em>tokenizers</em> library).",hn,Bt,js=`<a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerTokenizerFast">RoFormerTokenizerFast</a> is almost identical to <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizerFast">BertTokenizerFast</a> and runs end-to-end tokenization:
punctuation splitting and wordpiece. There are some difference between them when tokenizing Chinese.`,un,Vt,Cs=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,fn,ke,gn,de,ot,_n,qt,zs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A RoFormer sequence has the following format:`,bn,Nt,xs="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",xo,nt,Zo,z,st,yn,Lt,Zs=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention is
all you need</a> by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`,Tn,Gt,Ws=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to initialized with both <code>is_decoder</code> argument and
<code>add_cross_attention</code> set to <code>True</code>; an <code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,Mn,Xt,Is=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,kn,St,Bs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wn,ce,rt,vn,Ht,Vs='The <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerModel">RoFormerModel</a> forward method, overrides the <code>__call__</code> special method.',Fn,we,Wo,at,Io,x,it,$n,Qt,qs="RoFormer Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Jn,Et,Ns=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Un,At,Ls=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Rn,P,lt,jn,Yt,Gs='The <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForCausalLM">RoFormerForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Cn,ve,zn,Fe,Bo,dt,Vo,Z,ct,xn,Pt,Xs="The Roformer Model with a <code>language modeling</code> head on top.”",Zn,Ot,Ss=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Wn,Dt,Hs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,In,O,mt,Bn,Kt,Qs='The <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForMaskedLM">RoFormerForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Vn,$e,qn,Je,qo,pt,No,W,ht,Nn,eo,Es=`RoFormer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,Ln,to,As=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Gn,oo,Ys=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xn,L,ut,Sn,no,Ps='The <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForSequenceClassification">RoFormerForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Hn,Ue,Qn,Re,En,je,Lo,ft,Go,I,gt,An,so,Os=`The Roformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Yn,ro,Ds=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Pn,ao,Ks=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,On,D,_t,Dn,io,er='The <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForMultipleChoice">RoFormerForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Kn,Ce,es,ze,Xo,bt,So,B,yt,ts,lo,tr=`The Roformer transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,os,co,or=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ns,mo,nr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ss,K,Tt,rs,po,sr='The <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForTokenClassification">RoFormerForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',as,xe,is,Ze,Ho,Mt,Qo,V,kt,ls,ho,rr=`The Roformer transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ds,uo,ar=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,cs,fo,ir=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ms,ee,wt,ps,go,lr='The <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForQuestionAnswering">RoFormerForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',hs,We,us,Ie,Eo,vt,Ao,bo,Yo;return ae=new te({props:{title:"RoFormer",local:"roformer",headingTag:"h1"}}),_e=new qe({props:{warning:!1,$$slots:{default:[_r]},$$scope:{ctx:w}}}),be=new gr({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Mr]},$$scope:{ctx:w}}}),Se=new te({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Qe=new te({props:{title:"RoFormerConfig",local:"transformers.RoFormerConfig",headingTag:"h2"}}),Ee=new $({props:{name:"class transformers.RoFormerConfig",anchor:"transformers.RoFormerConfig",parameters:[{name:"vocab_size",val:" = 50000"},{name:"embedding_size",val:" = None"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 1536"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"rotary_value",val:" = False"},{name:"use_cache",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RoFormerConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50000) &#x2014;
Vocabulary size of the RoFormer model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerModel">RoFormerModel</a> or <code>TFRoFormerModel</code>.`,name:"vocab_size"},{anchor:"transformers.RoFormerConfig.embedding_size",description:`<strong>embedding_size</strong> (<code>int</code>, <em>optional</em>, defaults to None) &#x2014;
Dimensionality of the encoder layers and the pooler layer. Defaults to the <code>hidden_size</code> if not provided.`,name:"embedding_size"},{anchor:"transformers.RoFormerConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.RoFormerConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.RoFormerConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.RoFormerConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.RoFormerConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.RoFormerConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.RoFormerConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.RoFormerConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1536) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 1536).`,name:"max_position_embeddings"},{anchor:"transformers.RoFormerConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerModel">RoFormerModel</a> or <code>TFRoFormerModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.RoFormerConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.RoFormerConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.RoFormerConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.RoFormerConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.RoFormerConfig.rotary_value",description:`<strong>rotary_value</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not apply rotary position embeddings on value layer.`,name:"rotary_value"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/configuration_roformer.py#L28"}}),ye=new pe({props:{anchor:"transformers.RoFormerConfig.example",$$slots:{default:[kr]},$$scope:{ctx:w}}}),Ae=new te({props:{title:"RoFormerTokenizer",local:"transformers.RoFormerTokenizer",headingTag:"h2"}}),Ye=new $({props:{name:"class transformers.RoFormerTokenizer",anchor:"transformers.RoFormerTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RoFormerTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.RoFormerTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.RoFormerTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.RoFormerTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.RoFormerTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.RoFormerTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.RoFormerTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.RoFormerTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.RoFormerTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.RoFormerTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.RoFormerTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"strip_accents"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/tokenization_roformer.py#L273"}}),Te=new pe({props:{anchor:"transformers.RoFormerTokenizer.example",$$slots:{default:[wr]},$$scope:{ctx:w}}}),Pe=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.RoFormerTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.RoFormerTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.RoFormerTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/tokenization_roformer.py#L437",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Oe=new $({props:{name:"get_special_tokens_mask",anchor:"transformers.RoFormerTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.RoFormerTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.RoFormerTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.RoFormerTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/tokenization_roformer.py#L462",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),De=new $({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.RoFormerTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.RoFormerTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.RoFormerTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ke=new $({props:{name:"save_vocabulary",anchor:"transformers.RoFormerTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/tokenization_roformer.py#L490"}}),et=new te({props:{title:"RoFormerTokenizerFast",local:"transformers.RoFormerTokenizerFast",headingTag:"h2"}}),tt=new $({props:{name:"class transformers.RoFormerTokenizerFast",anchor:"transformers.RoFormerTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/tokenization_roformer_fast.py#L34"}}),ke=new pe({props:{anchor:"transformers.RoFormerTokenizerFast.example",$$slots:{default:[vr]},$$scope:{ctx:w}}}),ot=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.RoFormerTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.RoFormerTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.RoFormerTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/tokenization_roformer_fast.py#L111",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),nt=new te({props:{title:"RoFormerModel",local:"transformers.RoFormerModel",headingTag:"h2"}}),st=new $({props:{name:"class transformers.RoFormerModel",anchor:"transformers.RoFormerModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoFormerModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerModel">RoFormerModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L802"}}),rt=new $({props:{name:"forward",anchor:"transformers.RoFormerModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.RoFormerModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoFormerModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoFormerModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoFormerModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoFormerModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoFormerModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RoFormerModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RoFormerModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RoFormerModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RoFormerModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoFormerModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoFormerModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.RoFormerModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L830",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig"
>RoFormerConfig</a>) and inputs.</p>
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
`}}),we=new qe({props:{$$slots:{default:[Fr]},$$scope:{ctx:w}}}),at=new te({props:{title:"RoFormerForCausalLM",local:"transformers.RoFormerForCausalLM",headingTag:"h2"}}),it=new $({props:{name:"class transformers.RoFormerForCausalLM",anchor:"transformers.RoFormerForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoFormerForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForCausalLM">RoFormerForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1040"}}),lt=new $({props:{name:"forward",anchor:"transformers.RoFormerForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RoFormerForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoFormerForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoFormerForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoFormerForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoFormerForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RoFormerForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RoFormerForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoFormerForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.RoFormerForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RoFormerForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.RoFormerForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RoFormerForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoFormerForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoFormerForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.RoFormerForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1062",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig"
>RoFormerConfig</a>) and inputs.</p>
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
`}}),ve=new qe({props:{$$slots:{default:[$r]},$$scope:{ctx:w}}}),Fe=new pe({props:{anchor:"transformers.RoFormerForCausalLM.forward.example",$$slots:{default:[Jr]},$$scope:{ctx:w}}}),dt=new te({props:{title:"RoFormerForMaskedLM",local:"transformers.RoFormerForMaskedLM",headingTag:"h2"}}),ct=new $({props:{name:"class transformers.RoFormerForMaskedLM",anchor:"transformers.RoFormerForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoFormerForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForMaskedLM">RoFormerForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L940"}}),mt=new $({props:{name:"forward",anchor:"transformers.RoFormerForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoFormerForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoFormerForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoFormerForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoFormerForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoFormerForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoFormerForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RoFormerForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RoFormerForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.RoFormerForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoFormerForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoFormerForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L965",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig"
>RoFormerConfig</a>) and inputs.</p>
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
`}}),$e=new qe({props:{$$slots:{default:[Ur]},$$scope:{ctx:w}}}),Je=new pe({props:{anchor:"transformers.RoFormerForMaskedLM.forward.example",$$slots:{default:[Rr]},$$scope:{ctx:w}}}),pt=new te({props:{title:"RoFormerForSequenceClassification",local:"transformers.RoFormerForSequenceClassification",headingTag:"h2"}}),ht=new $({props:{name:"class transformers.RoFormerForSequenceClassification",anchor:"transformers.RoFormerForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoFormerForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForSequenceClassification">RoFormerForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1175"}}),ut=new $({props:{name:"forward",anchor:"transformers.RoFormerForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoFormerForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoFormerForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoFormerForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoFormerForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoFormerForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoFormerForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.RoFormerForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoFormerForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoFormerForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1185",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig"
>RoFormerConfig</a>) and inputs.</p>
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
`}}),Ue=new qe({props:{$$slots:{default:[jr]},$$scope:{ctx:w}}}),Re=new pe({props:{anchor:"transformers.RoFormerForSequenceClassification.forward.example",$$slots:{default:[Cr]},$$scope:{ctx:w}}}),je=new pe({props:{anchor:"transformers.RoFormerForSequenceClassification.forward.example-2",$$slots:{default:[zr]},$$scope:{ctx:w}}}),ft=new te({props:{title:"RoFormerForMultipleChoice",local:"transformers.RoFormerForMultipleChoice",headingTag:"h2"}}),gt=new $({props:{name:"class transformers.RoFormerForMultipleChoice",anchor:"transformers.RoFormerForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoFormerForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForMultipleChoice">RoFormerForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1255"}}),_t=new $({props:{name:"forward",anchor:"transformers.RoFormerForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoFormerForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoFormerForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoFormerForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoFormerForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoFormerForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoFormerForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.RoFormerForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoFormerForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoFormerForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1266",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig"
>RoFormerConfig</a>) and inputs.</p>
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
`}}),Ce=new qe({props:{$$slots:{default:[xr]},$$scope:{ctx:w}}}),ze=new pe({props:{anchor:"transformers.RoFormerForMultipleChoice.forward.example",$$slots:{default:[Zr]},$$scope:{ctx:w}}}),bt=new te({props:{title:"RoFormerForTokenClassification",local:"transformers.RoFormerForTokenClassification",headingTag:"h2"}}),yt=new $({props:{name:"class transformers.RoFormerForTokenClassification",anchor:"transformers.RoFormerForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoFormerForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForTokenClassification">RoFormerForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1352"}}),Tt=new $({props:{name:"forward",anchor:"transformers.RoFormerForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoFormerForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoFormerForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoFormerForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoFormerForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoFormerForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoFormerForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.RoFormerForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoFormerForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoFormerForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1364",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig"
>RoFormerConfig</a>) and inputs.</p>
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
`}}),xe=new qe({props:{$$slots:{default:[Wr]},$$scope:{ctx:w}}}),Ze=new pe({props:{anchor:"transformers.RoFormerForTokenClassification.forward.example",$$slots:{default:[Ir]},$$scope:{ctx:w}}}),Mt=new te({props:{title:"RoFormerForQuestionAnswering",local:"transformers.RoFormerForQuestionAnswering",headingTag:"h2"}}),kt=new $({props:{name:"class transformers.RoFormerForQuestionAnswering",anchor:"transformers.RoFormerForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoFormerForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerForQuestionAnswering">RoFormerForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1417"}}),wt=new $({props:{name:"forward",anchor:"transformers.RoFormerForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoFormerForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoFormerForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roformer/modeling_roformer.py#L1430",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roformer#transformers.RoFormerConfig"
>RoFormerConfig</a>) and inputs.</p>
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
`}}),We=new qe({props:{$$slots:{default:[Br]},$$scope:{ctx:w}}}),Ie=new pe({props:{anchor:"transformers.RoFormerForQuestionAnswering.forward.example",$$slots:{default:[Vr]},$$scope:{ctx:w}}}),vt=new fr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/roformer.md"}}),{c(){t=c("meta"),h=r(),n=c("p"),i=r(),k=c("p"),k.innerHTML=o,u=r(),J=c("div"),J.innerHTML=_o,Ne=r(),f(ae.$$.fragment),To=r(),Le=c("p"),Le.innerHTML=gs,Mo=r(),Ge=c("p"),Ge.innerHTML=_s,ko=r(),f(_e.$$.fragment),wo=r(),Xe=c("p"),Xe.innerHTML=bs,vo=r(),f(be.$$.fragment),Fo=r(),f(Se.$$.fragment),$o=r(),He=c("ul"),He.innerHTML=ys,Jo=r(),f(Qe.$$.fragment),Uo=r(),G=c("div"),f(Ee.$$.fragment),Po=r(),$t=c("p"),$t.innerHTML=Ts,Oo=r(),Jt=c("p"),Jt.innerHTML=Ms,Do=r(),f(ye.$$.fragment),Ro=r(),f(Ae.$$.fragment),jo=r(),U=c("div"),f(Ye.$$.fragment),Ko=r(),Ut=c("p"),Ut.innerHTML=ks,en=r(),Rt=c("p"),Rt.innerHTML=ws,tn=r(),f(Te.$$.fragment),on=r(),ie=c("div"),f(Pe.$$.fragment),nn=r(),jt=c("p"),jt.textContent=vs,sn=r(),Ct=c("ul"),Ct.innerHTML=Fs,rn=r(),Me=c("div"),f(Oe.$$.fragment),an=r(),zt=c("p"),zt.innerHTML=$s,ln=r(),le=c("div"),f(De.$$.fragment),dn=r(),xt=c("p"),xt.innerHTML=Js,cn=r(),Zt=c("p"),Zt.textContent=Us,mn=r(),Wt=c("div"),f(Ke.$$.fragment),Co=r(),f(et.$$.fragment),zo=r(),C=c("div"),f(tt.$$.fragment),pn=r(),It=c("p"),It.innerHTML=Rs,hn=r(),Bt=c("p"),Bt.innerHTML=js,un=r(),Vt=c("p"),Vt.innerHTML=Cs,fn=r(),f(ke.$$.fragment),gn=r(),de=c("div"),f(ot.$$.fragment),_n=r(),qt=c("p"),qt.textContent=zs,bn=r(),Nt=c("ul"),Nt.innerHTML=xs,xo=r(),f(nt.$$.fragment),Zo=r(),z=c("div"),f(st.$$.fragment),yn=r(),Lt=c("p"),Lt.innerHTML=Zs,Tn=r(),Gt=c("p"),Gt.innerHTML=Ws,Mn=r(),Xt=c("p"),Xt.innerHTML=Is,kn=r(),St=c("p"),St.innerHTML=Bs,wn=r(),ce=c("div"),f(rt.$$.fragment),vn=r(),Ht=c("p"),Ht.innerHTML=Vs,Fn=r(),f(we.$$.fragment),Wo=r(),f(at.$$.fragment),Io=r(),x=c("div"),f(it.$$.fragment),$n=r(),Qt=c("p"),Qt.innerHTML=qs,Jn=r(),Et=c("p"),Et.innerHTML=Ns,Un=r(),At=c("p"),At.innerHTML=Ls,Rn=r(),P=c("div"),f(lt.$$.fragment),jn=r(),Yt=c("p"),Yt.innerHTML=Gs,Cn=r(),f(ve.$$.fragment),zn=r(),f(Fe.$$.fragment),Bo=r(),f(dt.$$.fragment),Vo=r(),Z=c("div"),f(ct.$$.fragment),xn=r(),Pt=c("p"),Pt.innerHTML=Xs,Zn=r(),Ot=c("p"),Ot.innerHTML=Ss,Wn=r(),Dt=c("p"),Dt.innerHTML=Hs,In=r(),O=c("div"),f(mt.$$.fragment),Bn=r(),Kt=c("p"),Kt.innerHTML=Qs,Vn=r(),f($e.$$.fragment),qn=r(),f(Je.$$.fragment),qo=r(),f(pt.$$.fragment),No=r(),W=c("div"),f(ht.$$.fragment),Nn=r(),eo=c("p"),eo.textContent=Es,Ln=r(),to=c("p"),to.innerHTML=As,Gn=r(),oo=c("p"),oo.innerHTML=Ys,Xn=r(),L=c("div"),f(ut.$$.fragment),Sn=r(),no=c("p"),no.innerHTML=Ps,Hn=r(),f(Ue.$$.fragment),Qn=r(),f(Re.$$.fragment),En=r(),f(je.$$.fragment),Lo=r(),f(ft.$$.fragment),Go=r(),I=c("div"),f(gt.$$.fragment),An=r(),so=c("p"),so.textContent=Os,Yn=r(),ro=c("p"),ro.innerHTML=Ds,Pn=r(),ao=c("p"),ao.innerHTML=Ks,On=r(),D=c("div"),f(_t.$$.fragment),Dn=r(),io=c("p"),io.innerHTML=er,Kn=r(),f(Ce.$$.fragment),es=r(),f(ze.$$.fragment),Xo=r(),f(bt.$$.fragment),So=r(),B=c("div"),f(yt.$$.fragment),ts=r(),lo=c("p"),lo.textContent=tr,os=r(),co=c("p"),co.innerHTML=or,ns=r(),mo=c("p"),mo.innerHTML=nr,ss=r(),K=c("div"),f(Tt.$$.fragment),rs=r(),po=c("p"),po.innerHTML=sr,as=r(),f(xe.$$.fragment),is=r(),f(Ze.$$.fragment),Ho=r(),f(Mt.$$.fragment),Qo=r(),V=c("div"),f(kt.$$.fragment),ls=r(),ho=c("p"),ho.innerHTML=rr,ds=r(),uo=c("p"),uo.innerHTML=ar,cs=r(),fo=c("p"),fo.innerHTML=ir,ms=r(),ee=c("div"),f(wt.$$.fragment),ps=r(),go=c("p"),go.innerHTML=lr,hs=r(),f(We.$$.fragment),us=r(),f(Ie.$$.fragment),Eo=r(),f(vt.$$.fragment),Ao=r(),bo=c("p"),this.h()},l(e){const l=hr("svelte-u9bgzb",document.head);t=m(l,"META",{name:!0,content:!0}),l.forEach(d),h=a(e),n=m(e,"P",{}),v(n).forEach(d),i=a(e),k=m(e,"P",{"data-svelte-h":!0}),M(k)!=="svelte-hlwusb"&&(k.innerHTML=o),u=a(e),J=m(e,"DIV",{style:!0,"data-svelte-h":!0}),M(J)!=="svelte-383xsf"&&(J.innerHTML=_o),Ne=a(e),g(ae.$$.fragment,e),To=a(e),Le=m(e,"P",{"data-svelte-h":!0}),M(Le)!=="svelte-1udicf6"&&(Le.innerHTML=gs),Mo=a(e),Ge=m(e,"P",{"data-svelte-h":!0}),M(Ge)!=="svelte-el5arb"&&(Ge.innerHTML=_s),ko=a(e),g(_e.$$.fragment,e),wo=a(e),Xe=m(e,"P",{"data-svelte-h":!0}),M(Xe)!=="svelte-lqa8w5"&&(Xe.innerHTML=bs),vo=a(e),g(be.$$.fragment,e),Fo=a(e),g(Se.$$.fragment,e),$o=a(e),He=m(e,"UL",{"data-svelte-h":!0}),M(He)!=="svelte-miew09"&&(He.innerHTML=ys),Jo=a(e),g(Qe.$$.fragment,e),Uo=a(e),G=m(e,"DIV",{class:!0});var oe=v(G);g(Ee.$$.fragment,oe),Po=a(oe),$t=m(oe,"P",{"data-svelte-h":!0}),M($t)!=="svelte-i1pgpn"&&($t.innerHTML=Ts),Oo=a(oe),Jt=m(oe,"P",{"data-svelte-h":!0}),M(Jt)!=="svelte-1ek1ss9"&&(Jt.innerHTML=Ms),Do=a(oe),g(ye.$$.fragment,oe),oe.forEach(d),Ro=a(e),g(Ae.$$.fragment,e),jo=a(e),U=m(e,"DIV",{class:!0});var j=v(U);g(Ye.$$.fragment,j),Ko=a(j),Ut=m(j,"P",{"data-svelte-h":!0}),M(Ut)!=="svelte-1ygx4dd"&&(Ut.innerHTML=ks),en=a(j),Rt=m(j,"P",{"data-svelte-h":!0}),M(Rt)!=="svelte-ntrhio"&&(Rt.innerHTML=ws),tn=a(j),g(Te.$$.fragment,j),on=a(j),ie=m(j,"DIV",{class:!0});var he=v(ie);g(Pe.$$.fragment,he),nn=a(he),jt=m(he,"P",{"data-svelte-h":!0}),M(jt)!=="svelte-1h5xar7"&&(jt.textContent=vs),sn=a(he),Ct=m(he,"UL",{"data-svelte-h":!0}),M(Ct)!=="svelte-xi6653"&&(Ct.innerHTML=Fs),he.forEach(d),rn=a(j),Me=m(j,"DIV",{class:!0});var Ft=v(Me);g(Oe.$$.fragment,Ft),an=a(Ft),zt=m(Ft,"P",{"data-svelte-h":!0}),M(zt)!=="svelte-1f4f5kp"&&(zt.innerHTML=$s),Ft.forEach(d),ln=a(j),le=m(j,"DIV",{class:!0});var ue=v(le);g(De.$$.fragment,ue),dn=a(ue),xt=m(ue,"P",{"data-svelte-h":!0}),M(xt)!=="svelte-zj1vf1"&&(xt.innerHTML=Js),cn=a(ue),Zt=m(ue,"P",{"data-svelte-h":!0}),M(Zt)!=="svelte-9vptpw"&&(Zt.textContent=Us),ue.forEach(d),mn=a(j),Wt=m(j,"DIV",{class:!0});var yo=v(Wt);g(Ke.$$.fragment,yo),yo.forEach(d),j.forEach(d),Co=a(e),g(et.$$.fragment,e),zo=a(e),C=m(e,"DIV",{class:!0});var q=v(C);g(tt.$$.fragment,q),pn=a(q),It=m(q,"P",{"data-svelte-h":!0}),M(It)!=="svelte-6k11b2"&&(It.innerHTML=Rs),hn=a(q),Bt=m(q,"P",{"data-svelte-h":!0}),M(Bt)!=="svelte-1rgjkk9"&&(Bt.innerHTML=js),un=a(q),Vt=m(q,"P",{"data-svelte-h":!0}),M(Vt)!=="svelte-gxzj9w"&&(Vt.innerHTML=Cs),fn=a(q),g(ke.$$.fragment,q),gn=a(q),de=m(q,"DIV",{class:!0});var fe=v(de);g(ot.$$.fragment,fe),_n=a(fe),qt=m(fe,"P",{"data-svelte-h":!0}),M(qt)!=="svelte-1h5xar7"&&(qt.textContent=zs),bn=a(fe),Nt=m(fe,"UL",{"data-svelte-h":!0}),M(Nt)!=="svelte-xi6653"&&(Nt.innerHTML=xs),fe.forEach(d),q.forEach(d),xo=a(e),g(nt.$$.fragment,e),Zo=a(e),z=m(e,"DIV",{class:!0});var N=v(z);g(st.$$.fragment,N),yn=a(N),Lt=m(N,"P",{"data-svelte-h":!0}),M(Lt)!=="svelte-1854dma"&&(Lt.innerHTML=Zs),Tn=a(N),Gt=m(N,"P",{"data-svelte-h":!0}),M(Gt)!=="svelte-174erte"&&(Gt.innerHTML=Ws),Mn=a(N),Xt=m(N,"P",{"data-svelte-h":!0}),M(Xt)!=="svelte-q52n56"&&(Xt.innerHTML=Is),kn=a(N),St=m(N,"P",{"data-svelte-h":!0}),M(St)!=="svelte-hswkmf"&&(St.innerHTML=Bs),wn=a(N),ce=m(N,"DIV",{class:!0});var ge=v(ce);g(rt.$$.fragment,ge),vn=a(ge),Ht=m(ge,"P",{"data-svelte-h":!0}),M(Ht)!=="svelte-1qjouyx"&&(Ht.innerHTML=Vs),Fn=a(ge),g(we.$$.fragment,ge),ge.forEach(d),N.forEach(d),Wo=a(e),g(at.$$.fragment,e),Io=a(e),x=m(e,"DIV",{class:!0});var X=v(x);g(it.$$.fragment,X),$n=a(X),Qt=m(X,"P",{"data-svelte-h":!0}),M(Qt)!=="svelte-fci1kp"&&(Qt.innerHTML=qs),Jn=a(X),Et=m(X,"P",{"data-svelte-h":!0}),M(Et)!=="svelte-q52n56"&&(Et.innerHTML=Ns),Un=a(X),At=m(X,"P",{"data-svelte-h":!0}),M(At)!=="svelte-hswkmf"&&(At.innerHTML=Ls),Rn=a(X),P=m(X,"DIV",{class:!0});var ne=v(P);g(lt.$$.fragment,ne),jn=a(ne),Yt=m(ne,"P",{"data-svelte-h":!0}),M(Yt)!=="svelte-8gdu45"&&(Yt.innerHTML=Gs),Cn=a(ne),g(ve.$$.fragment,ne),zn=a(ne),g(Fe.$$.fragment,ne),ne.forEach(d),X.forEach(d),Bo=a(e),g(dt.$$.fragment,e),Vo=a(e),Z=m(e,"DIV",{class:!0});var S=v(Z);g(ct.$$.fragment,S),xn=a(S),Pt=m(S,"P",{"data-svelte-h":!0}),M(Pt)!=="svelte-ttklty"&&(Pt.innerHTML=Xs),Zn=a(S),Ot=m(S,"P",{"data-svelte-h":!0}),M(Ot)!=="svelte-q52n56"&&(Ot.innerHTML=Ss),Wn=a(S),Dt=m(S,"P",{"data-svelte-h":!0}),M(Dt)!=="svelte-hswkmf"&&(Dt.innerHTML=Hs),In=a(S),O=m(S,"DIV",{class:!0});var se=v(O);g(mt.$$.fragment,se),Bn=a(se),Kt=m(se,"P",{"data-svelte-h":!0}),M(Kt)!=="svelte-k2u65p"&&(Kt.innerHTML=Qs),Vn=a(se),g($e.$$.fragment,se),qn=a(se),g(Je.$$.fragment,se),se.forEach(d),S.forEach(d),qo=a(e),g(pt.$$.fragment,e),No=a(e),W=m(e,"DIV",{class:!0});var H=v(W);g(ht.$$.fragment,H),Nn=a(H),eo=m(H,"P",{"data-svelte-h":!0}),M(eo)!=="svelte-db97bs"&&(eo.textContent=Es),Ln=a(H),to=m(H,"P",{"data-svelte-h":!0}),M(to)!=="svelte-q52n56"&&(to.innerHTML=As),Gn=a(H),oo=m(H,"P",{"data-svelte-h":!0}),M(oo)!=="svelte-hswkmf"&&(oo.innerHTML=Ys),Xn=a(H),L=m(H,"DIV",{class:!0});var Q=v(L);g(ut.$$.fragment,Q),Sn=a(Q),no=m(Q,"P",{"data-svelte-h":!0}),M(no)!=="svelte-1oastcd"&&(no.innerHTML=Ps),Hn=a(Q),g(Ue.$$.fragment,Q),Qn=a(Q),g(Re.$$.fragment,Q),En=a(Q),g(je.$$.fragment,Q),Q.forEach(d),H.forEach(d),Lo=a(e),g(ft.$$.fragment,e),Go=a(e),I=m(e,"DIV",{class:!0});var E=v(I);g(gt.$$.fragment,E),An=a(E),so=m(E,"P",{"data-svelte-h":!0}),M(so)!=="svelte-19s7tib"&&(so.textContent=Os),Yn=a(E),ro=m(E,"P",{"data-svelte-h":!0}),M(ro)!=="svelte-q52n56"&&(ro.innerHTML=Ds),Pn=a(E),ao=m(E,"P",{"data-svelte-h":!0}),M(ao)!=="svelte-hswkmf"&&(ao.innerHTML=Ks),On=a(E),D=m(E,"DIV",{class:!0});var re=v(D);g(_t.$$.fragment,re),Dn=a(re),io=m(re,"P",{"data-svelte-h":!0}),M(io)!=="svelte-hky2nd"&&(io.innerHTML=er),Kn=a(re),g(Ce.$$.fragment,re),es=a(re),g(ze.$$.fragment,re),re.forEach(d),E.forEach(d),Xo=a(e),g(bt.$$.fragment,e),So=a(e),B=m(e,"DIV",{class:!0});var A=v(B);g(yt.$$.fragment,A),ts=a(A),lo=m(A,"P",{"data-svelte-h":!0}),M(lo)!=="svelte-27n2oi"&&(lo.textContent=tr),os=a(A),co=m(A,"P",{"data-svelte-h":!0}),M(co)!=="svelte-q52n56"&&(co.innerHTML=or),ns=a(A),mo=m(A,"P",{"data-svelte-h":!0}),M(mo)!=="svelte-hswkmf"&&(mo.innerHTML=nr),ss=a(A),K=m(A,"DIV",{class:!0});var Be=v(K);g(Tt.$$.fragment,Be),rs=a(Be),po=m(Be,"P",{"data-svelte-h":!0}),M(po)!=="svelte-rdrlft"&&(po.innerHTML=sr),as=a(Be),g(xe.$$.fragment,Be),is=a(Be),g(Ze.$$.fragment,Be),Be.forEach(d),A.forEach(d),Ho=a(e),g(Mt.$$.fragment,e),Qo=a(e),V=m(e,"DIV",{class:!0});var me=v(V);g(kt.$$.fragment,me),ls=a(me),ho=m(me,"P",{"data-svelte-h":!0}),M(ho)!=="svelte-1rs78mv"&&(ho.innerHTML=rr),ds=a(me),uo=m(me,"P",{"data-svelte-h":!0}),M(uo)!=="svelte-q52n56"&&(uo.innerHTML=ar),cs=a(me),fo=m(me,"P",{"data-svelte-h":!0}),M(fo)!=="svelte-hswkmf"&&(fo.innerHTML=ir),ms=a(me),ee=m(me,"DIV",{class:!0});var Ve=v(ee);g(wt.$$.fragment,Ve),ps=a(Ve),go=m(Ve,"P",{"data-svelte-h":!0}),M(go)!=="svelte-sh7ayv"&&(go.innerHTML=lr),hs=a(Ve),g(We.$$.fragment,Ve),us=a(Ve),g(Ie.$$.fragment,Ve),Ve.forEach(d),me.forEach(d),Eo=a(e),g(vt.$$.fragment,e),Ao=a(e),bo=m(e,"P",{}),v(bo).forEach(d),this.h()},h(){F(t,"name","hf:doc:metadata"),F(t,"content",Nr),ur(J,"float","right"),F(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(Wt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,l){s(document.head,t),p(e,h,l),p(e,n,l),p(e,i,l),p(e,k,l),p(e,u,l),p(e,J,l),p(e,Ne,l),_(ae,e,l),p(e,To,l),p(e,Le,l),p(e,Mo,l),p(e,Ge,l),p(e,ko,l),_(_e,e,l),p(e,wo,l),p(e,Xe,l),p(e,vo,l),_(be,e,l),p(e,Fo,l),_(Se,e,l),p(e,$o,l),p(e,He,l),p(e,Jo,l),_(Qe,e,l),p(e,Uo,l),p(e,G,l),_(Ee,G,null),s(G,Po),s(G,$t),s(G,Oo),s(G,Jt),s(G,Do),_(ye,G,null),p(e,Ro,l),_(Ae,e,l),p(e,jo,l),p(e,U,l),_(Ye,U,null),s(U,Ko),s(U,Ut),s(U,en),s(U,Rt),s(U,tn),_(Te,U,null),s(U,on),s(U,ie),_(Pe,ie,null),s(ie,nn),s(ie,jt),s(ie,sn),s(ie,Ct),s(U,rn),s(U,Me),_(Oe,Me,null),s(Me,an),s(Me,zt),s(U,ln),s(U,le),_(De,le,null),s(le,dn),s(le,xt),s(le,cn),s(le,Zt),s(U,mn),s(U,Wt),_(Ke,Wt,null),p(e,Co,l),_(et,e,l),p(e,zo,l),p(e,C,l),_(tt,C,null),s(C,pn),s(C,It),s(C,hn),s(C,Bt),s(C,un),s(C,Vt),s(C,fn),_(ke,C,null),s(C,gn),s(C,de),_(ot,de,null),s(de,_n),s(de,qt),s(de,bn),s(de,Nt),p(e,xo,l),_(nt,e,l),p(e,Zo,l),p(e,z,l),_(st,z,null),s(z,yn),s(z,Lt),s(z,Tn),s(z,Gt),s(z,Mn),s(z,Xt),s(z,kn),s(z,St),s(z,wn),s(z,ce),_(rt,ce,null),s(ce,vn),s(ce,Ht),s(ce,Fn),_(we,ce,null),p(e,Wo,l),_(at,e,l),p(e,Io,l),p(e,x,l),_(it,x,null),s(x,$n),s(x,Qt),s(x,Jn),s(x,Et),s(x,Un),s(x,At),s(x,Rn),s(x,P),_(lt,P,null),s(P,jn),s(P,Yt),s(P,Cn),_(ve,P,null),s(P,zn),_(Fe,P,null),p(e,Bo,l),_(dt,e,l),p(e,Vo,l),p(e,Z,l),_(ct,Z,null),s(Z,xn),s(Z,Pt),s(Z,Zn),s(Z,Ot),s(Z,Wn),s(Z,Dt),s(Z,In),s(Z,O),_(mt,O,null),s(O,Bn),s(O,Kt),s(O,Vn),_($e,O,null),s(O,qn),_(Je,O,null),p(e,qo,l),_(pt,e,l),p(e,No,l),p(e,W,l),_(ht,W,null),s(W,Nn),s(W,eo),s(W,Ln),s(W,to),s(W,Gn),s(W,oo),s(W,Xn),s(W,L),_(ut,L,null),s(L,Sn),s(L,no),s(L,Hn),_(Ue,L,null),s(L,Qn),_(Re,L,null),s(L,En),_(je,L,null),p(e,Lo,l),_(ft,e,l),p(e,Go,l),p(e,I,l),_(gt,I,null),s(I,An),s(I,so),s(I,Yn),s(I,ro),s(I,Pn),s(I,ao),s(I,On),s(I,D),_(_t,D,null),s(D,Dn),s(D,io),s(D,Kn),_(Ce,D,null),s(D,es),_(ze,D,null),p(e,Xo,l),_(bt,e,l),p(e,So,l),p(e,B,l),_(yt,B,null),s(B,ts),s(B,lo),s(B,os),s(B,co),s(B,ns),s(B,mo),s(B,ss),s(B,K),_(Tt,K,null),s(K,rs),s(K,po),s(K,as),_(xe,K,null),s(K,is),_(Ze,K,null),p(e,Ho,l),_(Mt,e,l),p(e,Qo,l),p(e,V,l),_(kt,V,null),s(V,ls),s(V,ho),s(V,ds),s(V,uo),s(V,cs),s(V,fo),s(V,ms),s(V,ee),_(wt,ee,null),s(ee,ps),s(ee,go),s(ee,hs),_(We,ee,null),s(ee,us),_(Ie,ee,null),p(e,Eo,l),_(vt,e,l),p(e,Ao,l),p(e,bo,l),Yo=!0},p(e,[l]){const oe={};l&2&&(oe.$$scope={dirty:l,ctx:e}),_e.$set(oe);const j={};l&2&&(j.$$scope={dirty:l,ctx:e}),be.$set(j);const he={};l&2&&(he.$$scope={dirty:l,ctx:e}),ye.$set(he);const Ft={};l&2&&(Ft.$$scope={dirty:l,ctx:e}),Te.$set(Ft);const ue={};l&2&&(ue.$$scope={dirty:l,ctx:e}),ke.$set(ue);const yo={};l&2&&(yo.$$scope={dirty:l,ctx:e}),we.$set(yo);const q={};l&2&&(q.$$scope={dirty:l,ctx:e}),ve.$set(q);const fe={};l&2&&(fe.$$scope={dirty:l,ctx:e}),Fe.$set(fe);const N={};l&2&&(N.$$scope={dirty:l,ctx:e}),$e.$set(N);const ge={};l&2&&(ge.$$scope={dirty:l,ctx:e}),Je.$set(ge);const X={};l&2&&(X.$$scope={dirty:l,ctx:e}),Ue.$set(X);const ne={};l&2&&(ne.$$scope={dirty:l,ctx:e}),Re.$set(ne);const S={};l&2&&(S.$$scope={dirty:l,ctx:e}),je.$set(S);const se={};l&2&&(se.$$scope={dirty:l,ctx:e}),Ce.$set(se);const H={};l&2&&(H.$$scope={dirty:l,ctx:e}),ze.$set(H);const Q={};l&2&&(Q.$$scope={dirty:l,ctx:e}),xe.$set(Q);const E={};l&2&&(E.$$scope={dirty:l,ctx:e}),Ze.$set(E);const re={};l&2&&(re.$$scope={dirty:l,ctx:e}),We.$set(re);const A={};l&2&&(A.$$scope={dirty:l,ctx:e}),Ie.$set(A)},i(e){Yo||(b(ae.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(Se.$$.fragment,e),b(Qe.$$.fragment,e),b(Ee.$$.fragment,e),b(ye.$$.fragment,e),b(Ae.$$.fragment,e),b(Ye.$$.fragment,e),b(Te.$$.fragment,e),b(Pe.$$.fragment,e),b(Oe.$$.fragment,e),b(De.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(ke.$$.fragment,e),b(ot.$$.fragment,e),b(nt.$$.fragment,e),b(st.$$.fragment,e),b(rt.$$.fragment,e),b(we.$$.fragment,e),b(at.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(ve.$$.fragment,e),b(Fe.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(mt.$$.fragment,e),b($e.$$.fragment,e),b(Je.$$.fragment,e),b(pt.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(Ue.$$.fragment,e),b(Re.$$.fragment,e),b(je.$$.fragment,e),b(ft.$$.fragment,e),b(gt.$$.fragment,e),b(_t.$$.fragment,e),b(Ce.$$.fragment,e),b(ze.$$.fragment,e),b(bt.$$.fragment,e),b(yt.$$.fragment,e),b(Tt.$$.fragment,e),b(xe.$$.fragment,e),b(Ze.$$.fragment,e),b(Mt.$$.fragment,e),b(kt.$$.fragment,e),b(wt.$$.fragment,e),b(We.$$.fragment,e),b(Ie.$$.fragment,e),b(vt.$$.fragment,e),Yo=!0)},o(e){y(ae.$$.fragment,e),y(_e.$$.fragment,e),y(be.$$.fragment,e),y(Se.$$.fragment,e),y(Qe.$$.fragment,e),y(Ee.$$.fragment,e),y(ye.$$.fragment,e),y(Ae.$$.fragment,e),y(Ye.$$.fragment,e),y(Te.$$.fragment,e),y(Pe.$$.fragment,e),y(Oe.$$.fragment,e),y(De.$$.fragment,e),y(Ke.$$.fragment,e),y(et.$$.fragment,e),y(tt.$$.fragment,e),y(ke.$$.fragment,e),y(ot.$$.fragment,e),y(nt.$$.fragment,e),y(st.$$.fragment,e),y(rt.$$.fragment,e),y(we.$$.fragment,e),y(at.$$.fragment,e),y(it.$$.fragment,e),y(lt.$$.fragment,e),y(ve.$$.fragment,e),y(Fe.$$.fragment,e),y(dt.$$.fragment,e),y(ct.$$.fragment,e),y(mt.$$.fragment,e),y($e.$$.fragment,e),y(Je.$$.fragment,e),y(pt.$$.fragment,e),y(ht.$$.fragment,e),y(ut.$$.fragment,e),y(Ue.$$.fragment,e),y(Re.$$.fragment,e),y(je.$$.fragment,e),y(ft.$$.fragment,e),y(gt.$$.fragment,e),y(_t.$$.fragment,e),y(Ce.$$.fragment,e),y(ze.$$.fragment,e),y(bt.$$.fragment,e),y(yt.$$.fragment,e),y(Tt.$$.fragment,e),y(xe.$$.fragment,e),y(Ze.$$.fragment,e),y(Mt.$$.fragment,e),y(kt.$$.fragment,e),y(wt.$$.fragment,e),y(We.$$.fragment,e),y(Ie.$$.fragment,e),y(vt.$$.fragment,e),Yo=!1},d(e){e&&(d(h),d(n),d(i),d(k),d(u),d(J),d(Ne),d(To),d(Le),d(Mo),d(Ge),d(ko),d(wo),d(Xe),d(vo),d(Fo),d($o),d(He),d(Jo),d(Uo),d(G),d(Ro),d(jo),d(U),d(Co),d(zo),d(C),d(xo),d(Zo),d(z),d(Wo),d(Io),d(x),d(Bo),d(Vo),d(Z),d(qo),d(No),d(W),d(Lo),d(Go),d(I),d(Xo),d(So),d(B),d(Ho),d(Qo),d(V),d(Eo),d(Ao),d(bo)),d(t),T(ae,e),T(_e,e),T(be,e),T(Se,e),T(Qe,e),T(Ee),T(ye),T(Ae,e),T(Ye),T(Te),T(Pe),T(Oe),T(De),T(Ke),T(et,e),T(tt),T(ke),T(ot),T(nt,e),T(st),T(rt),T(we),T(at,e),T(it),T(lt),T(ve),T(Fe),T(dt,e),T(ct),T(mt),T($e),T(Je),T(pt,e),T(ht),T(ut),T(Ue),T(Re),T(je),T(ft,e),T(gt),T(_t),T(Ce),T(ze),T(bt,e),T(yt),T(Tt),T(xe),T(Ze),T(Mt,e),T(kt),T(wt),T(We),T(Ie),T(vt,e)}}}const Nr='{"title":"RoFormer","local":"roformer","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"RoFormerConfig","local":"transformers.RoFormerConfig","sections":[],"depth":2},{"title":"RoFormerTokenizer","local":"transformers.RoFormerTokenizer","sections":[],"depth":2},{"title":"RoFormerTokenizerFast","local":"transformers.RoFormerTokenizerFast","sections":[],"depth":2},{"title":"RoFormerModel","local":"transformers.RoFormerModel","sections":[],"depth":2},{"title":"RoFormerForCausalLM","local":"transformers.RoFormerForCausalLM","sections":[],"depth":2},{"title":"RoFormerForMaskedLM","local":"transformers.RoFormerForMaskedLM","sections":[],"depth":2},{"title":"RoFormerForSequenceClassification","local":"transformers.RoFormerForSequenceClassification","sections":[],"depth":2},{"title":"RoFormerForMultipleChoice","local":"transformers.RoFormerForMultipleChoice","sections":[],"depth":2},{"title":"RoFormerForTokenClassification","local":"transformers.RoFormerForTokenClassification","sections":[],"depth":2},{"title":"RoFormerForQuestionAnswering","local":"transformers.RoFormerForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Lr(w){return cr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Pr extends mr{constructor(t){super(),pr(this,t,Lr,qr,dr,{})}}export{Pr as component};
