import{s as Ps,o as Os,n as j}from"../chunks/scheduler.18a86fab.js";import{S as Ks,i as ea,g as p,s as l,r as g,A as ta,h as m,f as a,c as i,j as k,x as w,u as M,k as J,l as na,y as d,a as c,v as _,d as T,t as b,w as y}from"../chunks/index.98837b22.js";import{T as $t}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as W}from"../chunks/CodeBlock.8d0c2e8a.js";import{E}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ie,E as oa}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as sa,a as ls}from"../chunks/HfOption.6641485e.js";function aa(v){let n,u='This model was contributed by <a href="https://huggingface.co/patrickvonplaten" rel="nofollow">patrickvonplaten</a>.',o,s,f="Click on the mT5 models in the right sidebar for more examples of how to apply mT5 to different language tasks.";return{c(){n=p("p"),n.innerHTML=u,o=l(),s=p("p"),s.textContent=f},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-vqdfz5"&&(n.innerHTML=u),o=i(t),s=m(t,"P",{"data-svelte-h":!0}),w(s)!=="svelte-17v49qx"&&(s.textContent=f)},m(t,h){c(t,n,h),c(t,o,h),c(t,s,h)},p:j,d(t){t&&(a(n),a(o),a(s))}}}function ra(v){let n,u;return n=new W({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0MnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyY3NlYnVldG5scCUyRm1UNV9tdWx0aWxpbmd1YWxfWExTdW0lMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUyMiUyMiUyMlBsYW50cyUyMGFyZSUyMHJlbWFya2FibGUlMjBvcmdhbmlzbXMlMjB0aGF0JTIwcHJvZHVjZSUyMHRoZWlyJTIwb3duJTIwZm9vZCUyMHVzaW5nJTIwYSUyMG1ldGhvZCUyMGNhbGxlZCUyMHBob3Rvc3ludGhlc2lzLiUwQVRoaXMlMjBwcm9jZXNzJTIwaW52b2x2ZXMlMjBjb252ZXJ0aW5nJTIwc3VubGlnaHQlMkMlMjBjYXJib24lMjBkaW94aWRlJTJDJTIwYW5kJTIwd2F0ZXIlMjBpbnRvJTIwZ2x1Y29zZSUyQyUyMHdoaWNoJTIwcHJvdmlkZXMlMjBlbmVyZ3klMjBmb3IlMjBncm93dGguJTBBUGxhbnRzJTIwcGxheSUyMGElMjBjcnVjaWFsJTIwcm9sZSUyMGluJTIwc3VzdGFpbmluZyUyMGxpZmUlMjBvbiUyMEVhcnRoJTIwYnklMjBnZW5lcmF0aW5nJTIwb3h5Z2VuJTIwYW5kJTIwc2VydmluZyUyMGFzJTIwdGhlJTIwZm91bmRhdGlvbiUyMG9mJTIwbW9zdCUyMGVjb3N5c3RlbXMuJTIyJTIyJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text2text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;csebuetnlp/mT5_multilingual_XLSum&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;&quot;&quot;Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.&quot;&quot;&quot;</span>)`,wrap:!1}}),{c(){g(n.$$.fragment)},l(o){M(n.$$.fragment,o)},m(o,s){_(n,o,s),u=!0},p:j,i(o){u||(T(n.$$.fragment,o),u=!0)},o(o){b(n.$$.fragment,o),u=!1},d(o){y(n,o)}}}function la(v){let n,u;return n=new W({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmNzZWJ1ZXRubHAlMkZtVDVfbXVsdGlsaW5ndWFsX1hMU3VtJTIyJTBBKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJjc2VidWV0bmxwJTJGbVQ1X211bHRpbGluZ3VhbF9YTFN1bSUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBKSUwQSUwQWlucHV0X3RleHQlMjAlM0QlMjAlMjIlMjIlMjJQbGFudHMlMjBhcmUlMjByZW1hcmthYmxlJTIwb3JnYW5pc21zJTIwdGhhdCUyMHByb2R1Y2UlMjB0aGVpciUyMG93biUyMGZvb2QlMjB1c2luZyUyMGElMjBtZXRob2QlMjBjYWxsZWQlMjBwaG90b3N5bnRoZXNpcy4lMEFUaGlzJTIwcHJvY2VzcyUyMGludm9sdmVzJTIwY29udmVydGluZyUyMHN1bmxpZ2h0JTJDJTIwY2FyYm9uJTIwZGlveGlkZSUyQyUyMGFuZCUyMHdhdGVyJTIwaW50byUyMGdsdWNvc2UlMkMlMjB3aGljaCUyMHByb3ZpZGVzJTIwZW5lcmd5JTIwZm9yJTIwZ3Jvd3RoLiUwQVBsYW50cyUyMHBsYXklMjBhJTIwY3J1Y2lhbCUyMHJvbGUlMjBpbiUyMHN1c3RhaW5pbmclMjBsaWZlJTIwb24lMjBFYXJ0aCUyMGJ5JTIwZ2VuZXJhdGluZyUyMG94eWdlbiUyMGFuZCUyMHNlcnZpbmclMjBhcyUyMHRoZSUyMGZvdW5kYXRpb24lMjBvZiUyMG1vc3QlMjBlY29zeXN0ZW1zLiUyMiUyMiUyMiUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcihpbnB1dF90ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;csebuetnlp/mT5_multilingual_XLSum&quot;</span>
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;csebuetnlp/mT5_multilingual_XLSum&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)

input_text = <span class="hljs-string">&quot;&quot;&quot;Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.&quot;&quot;&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){g(n.$$.fragment)},l(o){M(n.$$.fragment,o)},m(o,s){_(n,o,s),u=!0},p:j,i(o){u||(T(n.$$.fragment,o),u=!0)},o(o){b(n.$$.fragment,o),u=!1},d(o){y(n,o)}}}function ia(v){let n,u;return n=new W({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwYXJlJTIwcmVtYXJrYWJsZSUyMG9yZ2FuaXNtcyUyMHRoYXQlMjBwcm9kdWNlJTIwdGhlaXIlMjBvd24lMjBmb29kJTIwdXNpbmclMjBhJTIwbWV0aG9kJTIwY2FsbGVkJTIwcGhvdG9zeW50aGVzaXMuJTIyJTIwJTdDJTIwdHJhbnNmb3JtZXJzJTIwcnVuJTIwLS10YXNrJTIwdGV4dDJ0ZXh0LWdlbmVyYXRpb24lMjAtLW1vZGVsJTIwY3NlYnVldG5scCUyRm1UNV9tdWx0aWxpbmd1YWxfWExTdW0lMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants are remarkable organisms that produce their own food using a method called photosynthesis.&quot;</span> | transformers run --task text2text-generation --model csebuetnlp/mT5_multilingual_XLSum --device 0',wrap:!1}}),{c(){g(n.$$.fragment)},l(o){M(n.$$.fragment,o)},m(o,s){_(n,o,s),u=!0},p:j,i(o){u||(T(n.$$.fragment,o),u=!0)},o(o){b(n.$$.fragment,o),u=!1},d(o){y(n,o)}}}function da(v){let n,u,o,s,f,t;return n=new ls({props:{id:"usage",option:"Pipeline",$$slots:{default:[ra]},$$scope:{ctx:v}}}),o=new ls({props:{id:"usage",option:"AutoModel",$$slots:{default:[la]},$$scope:{ctx:v}}}),f=new ls({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ia]},$$scope:{ctx:v}}}),{c(){g(n.$$.fragment),u=l(),g(o.$$.fragment),s=l(),g(f.$$.fragment)},l(h){M(n.$$.fragment,h),u=i(h),M(o.$$.fragment,h),s=i(h),M(f.$$.fragment,h)},m(h,U){_(n,h,U),c(h,u,U),_(o,h,U),c(h,s,U),_(f,h,U),t=!0},p(h,U){const pn={};U&2&&(pn.$$scope={dirty:U,ctx:h}),n.$set(pn);const Ve={};U&2&&(Ve.$$scope={dirty:U,ctx:h}),o.$set(Ve);const de={};U&2&&(de.$$scope={dirty:U,ctx:h}),f.$set(de)},i(h){t||(T(n.$$.fragment,h),T(o.$$.fragment,h),T(f.$$.fragment,h),t=!0)},o(h){b(n.$$.fragment,h),b(o.$$.fragment,h),b(f.$$.fragment,h),t=!1},d(h){h&&(a(u),a(s)),y(n,h),y(o,h),y(f,h)}}}function ca(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"JTIzJTIwT24lMjBhJTIwNCUyMEdQVSUyMG1hY2hpbmUlMjB3aXRoJTIwbXQ1LXhsJTNBJTBBbW9kZWwlMjAlM0QlMjBNVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMk10NS14bCUyMiklMEFkZXZpY2VfbWFwJTIwJTNEJTIwJTdCJTBBJTIwJTIwJTIwJTIwMCUzQSUyMCU1QjAlMkMlMjAxJTJDJTIwMiU1RCUyQyUwQSUyMCUyMCUyMCUyMDElM0ElMjAlNUIzJTJDJTIwNCUyQyUyMDUlMkMlMjA2JTJDJTIwNyUyQyUyMDglMkMlMjA5JTVEJTJDJTBBJTIwJTIwJTIwJTIwMiUzQSUyMCU1QjEwJTJDJTIwMTElMkMlMjAxMiUyQyUyMDEzJTJDJTIwMTQlMkMlMjAxNSUyQyUyMDE2JTVEJTJDJTBBJTIwJTIwJTIwJTIwMyUzQSUyMCU1QjE3JTJDJTIwMTglMkMlMjAxOSUyQyUyMDIwJTJDJTIwMjElMkMlMjAyMiUyQyUyMDIzJTVEJTJDJTBBJTdEJTBBbW9kZWwucGFyYWxsZWxpemUoZGV2aWNlX21hcCklMjAlMjAlMjMlMjBTcGxpdHMlMjB0aGUlMjBtb2RlbCUyMGFjcm9zcyUyMHNldmVyYWwlMjBkZXZpY2VzJTBBbW9kZWwuZGVwYXJhbGxlbGl6ZSgpJTIwJTIwJTIzJTIwUHV0JTIwdGhlJTIwbW9kZWwlMjBiYWNrJTIwb24lMjBjcHUlMjBhbmQlMjBjbGVhbnMlMjBtZW1vcnklMjBieSUyMGNhbGxpbmclMjB0b3JjaC5jdWRhLmVtcHR5X2NhY2hlKCk=",highlighted:`<span class="hljs-comment"># On a 4 GPU machine with mt5-xl:</span>
model = MT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;Mt5-xl&quot;</span>)
device_map = {
    <span class="hljs-number">0</span>: [<span class="hljs-number">0</span>, <span class="hljs-number">1</span>, <span class="hljs-number">2</span>],
    <span class="hljs-number">1</span>: [<span class="hljs-number">3</span>, <span class="hljs-number">4</span>, <span class="hljs-number">5</span>, <span class="hljs-number">6</span>, <span class="hljs-number">7</span>, <span class="hljs-number">8</span>, <span class="hljs-number">9</span>],
    <span class="hljs-number">2</span>: [<span class="hljs-number">10</span>, <span class="hljs-number">11</span>, <span class="hljs-number">12</span>, <span class="hljs-number">13</span>, <span class="hljs-number">14</span>, <span class="hljs-number">15</span>, <span class="hljs-number">16</span>],
    <span class="hljs-number">3</span>: [<span class="hljs-number">17</span>, <span class="hljs-number">18</span>, <span class="hljs-number">19</span>, <span class="hljs-number">20</span>, <span class="hljs-number">21</span>, <span class="hljs-number">22</span>, <span class="hljs-number">23</span>],
}
model.parallelize(device_map)  <span class="hljs-comment"># Splits the model across several devices</span>
model.deparallelize()  <span class="hljs-comment"># Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()</span>`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function pa(v){let n,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=u},l(o){n=m(o,"P",{"data-svelte-h":!0}),w(n)!=="svelte-fincs2"&&(n.innerHTML=u)},m(o,s){c(o,n,s)},p:j,d(o){o&&a(n)}}}function ma(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNVDVNb2RlbCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm10NS1zbWFsbCUyMiklMEFtb2RlbCUyMCUzRCUyME1UNU1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIpJTBBJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMlN0dWRpZXMlMjBoYXZlJTIwYmVlbiUyMHNob3duJTIwdGhhdCUyMG93bmluZyUyMGElMjBkb2clMjBpcyUyMGdvb2QlMjBmb3IlMjB5b3UlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBZGVjb2Rlcl9pbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyU3R1ZGllcyUyMHNob3clMjB0aGF0JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikuaW5wdXRfaWRzJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQSUwQSUyMyUyMHByZXByb2Nlc3MlM0ElMjBQcmVwZW5kJTIwZGVjb2Rlcl9pbnB1dF9pZHMlMjB3aXRoJTIwc3RhcnQlMjB0b2tlbiUyMHdoaWNoJTIwaXMlMjBwYWQlMjB0b2tlbiUyMGZvciUyME1UNU1vZGVsLiUwQSUyMyUyMFRoaXMlMjBpcyUyMG5vdCUyMG5lZWRlZCUyMGZvciUyMHRvcmNoJ3MlMjBNVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMjBhcyUyMGl0JTIwZG9lcyUyMHRoaXMlMjBpbnRlcm5hbGx5JTIwdXNpbmclMjBsYWJlbHMlMjBhcmcuJTBBZGVjb2Rlcl9pbnB1dF9pZHMlMjAlM0QlMjBtb2RlbC5fc2hpZnRfcmlnaHQoZGVjb2Rlcl9pbnB1dF9pZHMpJTBBJTBBJTIzJTIwZm9yd2FyZCUyMHBhc3MlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RkZWNvZGVyX2lucHV0X2lkcyklMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MT5Model

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5Model.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># preprocess: Prepend decoder_input_ids with start token which is pad token for MT5Model.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># This is not needed for torch&#x27;s MT5ForConditionalGeneration as it does this internally using labels arg.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = model._shift_right(decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function ha(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"JTIzJTIwSGVyZSUyMGlzJTIwYW4lMjBleGFtcGxlJTIwb2YlMjBhJTIwZGV2aWNlJTIwbWFwJTIwb24lMjBhJTIwbWFjaGluZSUyMHdpdGglMjA0JTIwR1BVcyUyMHVzaW5nJTIwbXQ1LXhsJTJDJTIwd2hpY2glMjBoYXMlMjBhJTIwdG90YWwlMjBvZiUyMDI0JTIwYXR0ZW50aW9uJTIwbW9kdWxlcyUzQSUwQW1vZGVsJTIwJTNEJTIwTVQ1Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtdDUteGwlMjIpJTBBZGV2aWNlX21hcCUyMCUzRCUyMCU3QiUwQSUyMCUyMCUyMCUyMDAlM0ElMjAlNUIwJTJDJTIwMSUyQyUyMDIlNUQlMkMlMEElMjAlMjAlMjAlMjAxJTNBJTIwJTVCMyUyQyUyMDQlMkMlMjA1JTJDJTIwNiUyQyUyMDclMkMlMjA4JTJDJTIwOSU1RCUyQyUwQSUyMCUyMCUyMCUyMDIlM0ElMjAlNUIxMCUyQyUyMDExJTJDJTIwMTIlMkMlMjAxMyUyQyUyMDE0JTJDJTIwMTUlMkMlMjAxNiU1RCUyQyUwQSUyMCUyMCUyMCUyMDMlM0ElMjAlNUIxNyUyQyUyMDE4JTJDJTIwMTklMkMlMjAyMCUyQyUyMDIxJTJDJTIwMjIlMkMlMjAyMyU1RCUyQyUwQSU3RCUwQW1vZGVsLnBhcmFsbGVsaXplKGRldmljZV9tYXAp",highlighted:`<span class="hljs-comment"># Here is an example of a device map on a machine with 4 GPUs using mt5-xl, which has a total of 24 attention modules:</span>
model = MT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;mt5-xl&quot;</span>)
device_map = {
    <span class="hljs-number">0</span>: [<span class="hljs-number">0</span>, <span class="hljs-number">1</span>, <span class="hljs-number">2</span>],
    <span class="hljs-number">1</span>: [<span class="hljs-number">3</span>, <span class="hljs-number">4</span>, <span class="hljs-number">5</span>, <span class="hljs-number">6</span>, <span class="hljs-number">7</span>, <span class="hljs-number">8</span>, <span class="hljs-number">9</span>],
    <span class="hljs-number">2</span>: [<span class="hljs-number">10</span>, <span class="hljs-number">11</span>, <span class="hljs-number">12</span>, <span class="hljs-number">13</span>, <span class="hljs-number">14</span>, <span class="hljs-number">15</span>, <span class="hljs-number">16</span>],
    <span class="hljs-number">3</span>: [<span class="hljs-number">17</span>, <span class="hljs-number">18</span>, <span class="hljs-number">19</span>, <span class="hljs-number">20</span>, <span class="hljs-number">21</span>, <span class="hljs-number">22</span>, <span class="hljs-number">23</span>],
}
model.parallelize(device_map)`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function ua(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"JTIzJTIwT24lMjBhJTIwNCUyMEdQVSUyMG1hY2hpbmUlMjB3aXRoJTIwbXQ1LXhsJTNBJTBBbW9kZWwlMjAlM0QlMjBNVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMk10NS14bCUyMiklMEFkZXZpY2VfbWFwJTIwJTNEJTIwJTdCJTBBJTIwJTIwJTIwJTIwMCUzQSUyMCU1QjAlMkMlMjAxJTJDJTIwMiU1RCUyQyUwQSUyMCUyMCUyMCUyMDElM0ElMjAlNUIzJTJDJTIwNCUyQyUyMDUlMkMlMjA2JTJDJTIwNyUyQyUyMDglMkMlMjA5JTVEJTJDJTBBJTIwJTIwJTIwJTIwMiUzQSUyMCU1QjEwJTJDJTIwMTElMkMlMjAxMiUyQyUyMDEzJTJDJTIwMTQlMkMlMjAxNSUyQyUyMDE2JTVEJTJDJTBBJTIwJTIwJTIwJTIwMyUzQSUyMCU1QjE3JTJDJTIwMTglMkMlMjAxOSUyQyUyMDIwJTJDJTIwMjElMkMlMjAyMiUyQyUyMDIzJTVEJTJDJTBBJTdEJTBBbW9kZWwucGFyYWxsZWxpemUoZGV2aWNlX21hcCklMjAlMjAlMjMlMjBTcGxpdHMlMjB0aGUlMjBtb2RlbCUyMGFjcm9zcyUyMHNldmVyYWwlMjBkZXZpY2VzJTBBbW9kZWwuZGVwYXJhbGxlbGl6ZSgpJTIwJTIwJTIzJTIwUHV0JTIwdGhlJTIwbW9kZWwlMjBiYWNrJTIwb24lMjBjcHUlMjBhbmQlMjBjbGVhbnMlMjBtZW1vcnklMjBieSUyMGNhbGxpbmclMjB0b3JjaC5jdWRhLmVtcHR5X2NhY2hlKCk=",highlighted:`<span class="hljs-comment"># On a 4 GPU machine with mt5-xl:</span>
model = MT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;Mt5-xl&quot;</span>)
device_map = {
    <span class="hljs-number">0</span>: [<span class="hljs-number">0</span>, <span class="hljs-number">1</span>, <span class="hljs-number">2</span>],
    <span class="hljs-number">1</span>: [<span class="hljs-number">3</span>, <span class="hljs-number">4</span>, <span class="hljs-number">5</span>, <span class="hljs-number">6</span>, <span class="hljs-number">7</span>, <span class="hljs-number">8</span>, <span class="hljs-number">9</span>],
    <span class="hljs-number">2</span>: [<span class="hljs-number">10</span>, <span class="hljs-number">11</span>, <span class="hljs-number">12</span>, <span class="hljs-number">13</span>, <span class="hljs-number">14</span>, <span class="hljs-number">15</span>, <span class="hljs-number">16</span>],
    <span class="hljs-number">3</span>: [<span class="hljs-number">17</span>, <span class="hljs-number">18</span>, <span class="hljs-number">19</span>, <span class="hljs-number">20</span>, <span class="hljs-number">21</span>, <span class="hljs-number">22</span>, <span class="hljs-number">23</span>],
}
model.parallelize(device_map)  <span class="hljs-comment"># Splits the model across several devices</span>
model.deparallelize()  <span class="hljs-comment"># Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()</span>`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function fa(v){let n,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=u},l(o){n=m(o,"P",{"data-svelte-h":!0}),w(n)!=="svelte-fincs2"&&(n.innerHTML=u)},m(o,s){c(o,n,s)},p:j,d(o){o&&a(n)}}}function ga(v){let n,u="Examples:",o,s,f;return s=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBNVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm10NS1zbWFsbCUyMiklMEElMEElMjMlMjB0cmFpbmluZyUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjAlM0NleHRyYV9pZF8wJTNFJTIwd2Fsa3MlMjBpbiUyMCUzQ2V4dHJhX2lkXzElM0UlMjBwYXJrJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikuaW5wdXRfaWRzJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMiUzQ2V4dHJhX2lkXzAlM0UlMjBjdXRlJTIwZG9nJTIwJTNDZXh0cmFfaWRfMSUzRSUyMHRoZSUyMCUzQ2V4dHJhX2lkXzIlM0UlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQSUwQSUyMyUyMGluZmVyZW5jZSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJzdW1tYXJpemUlM0ElMjBzdHVkaWVzJTIwaGF2ZSUyMHNob3duJTIwdGhhdCUyMG93bmluZyUyMGElMjBkb2clMjBpcyUyMGdvb2QlMjBmb3IlMjB5b3UlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0X2lkcyklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKSUwQSUyMyUyMHN0dWRpZXMlMjBoYXZlJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdS4=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MT5ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># training</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(<span class="hljs-string">&quot;The &lt;extra_id_0&gt; walks in &lt;extra_id_1&gt; park&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;&lt;extra_id_0&gt; cute dog &lt;extra_id_1&gt; the &lt;extra_id_2&gt;&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># inference</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;summarize: studies have shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.generate(input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># studies have shown that owning a dog is good for you.</span>`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-kvfsh7"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function Ma(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"JTIzJTIwSGVyZSUyMGlzJTIwYW4lMjBleGFtcGxlJTIwb2YlMjBhJTIwZGV2aWNlJTIwbWFwJTIwb24lMjBhJTIwbWFjaGluZSUyMHdpdGglMjA0JTIwR1BVcyUyMHVzaW5nJTIwbXQ1LXhsJTJDJTIwd2hpY2glMjBoYXMlMjBhJTIwdG90YWwlMjBvZiUyMDI0JTIwYXR0ZW50aW9uJTIwbW9kdWxlcyUzQSUwQW1vZGVsJTIwJTNEJTIwTVQ1Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtdDUteGwlMjIpJTBBZGV2aWNlX21hcCUyMCUzRCUyMCU3QiUwQSUyMCUyMCUyMCUyMDAlM0ElMjAlNUIwJTJDJTIwMSUyQyUyMDIlNUQlMkMlMEElMjAlMjAlMjAlMjAxJTNBJTIwJTVCMyUyQyUyMDQlMkMlMjA1JTJDJTIwNiUyQyUyMDclMkMlMjA4JTJDJTIwOSU1RCUyQyUwQSUyMCUyMCUyMCUyMDIlM0ElMjAlNUIxMCUyQyUyMDExJTJDJTIwMTIlMkMlMjAxMyUyQyUyMDE0JTJDJTIwMTUlMkMlMjAxNiU1RCUyQyUwQSUyMCUyMCUyMCUyMDMlM0ElMjAlNUIxNyUyQyUyMDE4JTJDJTIwMTklMkMlMjAyMCUyQyUyMDIxJTJDJTIwMjIlMkMlMjAyMyU1RCUyQyUwQSU3RCUwQW1vZGVsLnBhcmFsbGVsaXplKGRldmljZV9tYXAp",highlighted:`<span class="hljs-comment"># Here is an example of a device map on a machine with 4 GPUs using mt5-xl, which has a total of 24 attention modules:</span>
model = MT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;mt5-xl&quot;</span>)
device_map = {
    <span class="hljs-number">0</span>: [<span class="hljs-number">0</span>, <span class="hljs-number">1</span>, <span class="hljs-number">2</span>],
    <span class="hljs-number">1</span>: [<span class="hljs-number">3</span>, <span class="hljs-number">4</span>, <span class="hljs-number">5</span>, <span class="hljs-number">6</span>, <span class="hljs-number">7</span>, <span class="hljs-number">8</span>, <span class="hljs-number">9</span>],
    <span class="hljs-number">2</span>: [<span class="hljs-number">10</span>, <span class="hljs-number">11</span>, <span class="hljs-number">12</span>, <span class="hljs-number">13</span>, <span class="hljs-number">14</span>, <span class="hljs-number">15</span>, <span class="hljs-number">16</span>],
    <span class="hljs-number">3</span>: [<span class="hljs-number">17</span>, <span class="hljs-number">18</span>, <span class="hljs-number">19</span>, <span class="hljs-number">20</span>, <span class="hljs-number">21</span>, <span class="hljs-number">22</span>, <span class="hljs-number">23</span>],
}
model.parallelize(device_map)`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function _a(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"JTIzJTIwT24lMjBhJTIwNCUyMEdQVSUyMG1hY2hpbmUlMjB3aXRoJTIwbXQ1LXhsJTNBJTBBbW9kZWwlMjAlM0QlMjBNVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMk10NS14bCUyMiklMEFkZXZpY2VfbWFwJTIwJTNEJTIwJTdCJTBBJTIwJTIwJTIwJTIwMCUzQSUyMCU1QjAlMkMlMjAxJTJDJTIwMiU1RCUyQyUwQSUyMCUyMCUyMCUyMDElM0ElMjAlNUIzJTJDJTIwNCUyQyUyMDUlMkMlMjA2JTJDJTIwNyUyQyUyMDglMkMlMjA5JTVEJTJDJTBBJTIwJTIwJTIwJTIwMiUzQSUyMCU1QjEwJTJDJTIwMTElMkMlMjAxMiUyQyUyMDEzJTJDJTIwMTQlMkMlMjAxNSUyQyUyMDE2JTVEJTJDJTBBJTIwJTIwJTIwJTIwMyUzQSUyMCU1QjE3JTJDJTIwMTglMkMlMjAxOSUyQyUyMDIwJTJDJTIwMjElMkMlMjAyMiUyQyUyMDIzJTVEJTJDJTBBJTdEJTBBbW9kZWwucGFyYWxsZWxpemUoZGV2aWNlX21hcCklMjAlMjAlMjMlMjBTcGxpdHMlMjB0aGUlMjBtb2RlbCUyMGFjcm9zcyUyMHNldmVyYWwlMjBkZXZpY2VzJTBBbW9kZWwuZGVwYXJhbGxlbGl6ZSgpJTIwJTIwJTIzJTIwUHV0JTIwdGhlJTIwbW9kZWwlMjBiYWNrJTIwb24lMjBjcHUlMjBhbmQlMjBjbGVhbnMlMjBtZW1vcnklMjBieSUyMGNhbGxpbmclMjB0b3JjaC5jdWRhLmVtcHR5X2NhY2hlKCk=",highlighted:`<span class="hljs-comment"># On a 4 GPU machine with mt5-xl:</span>
model = MT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;Mt5-xl&quot;</span>)
device_map = {
    <span class="hljs-number">0</span>: [<span class="hljs-number">0</span>, <span class="hljs-number">1</span>, <span class="hljs-number">2</span>],
    <span class="hljs-number">1</span>: [<span class="hljs-number">3</span>, <span class="hljs-number">4</span>, <span class="hljs-number">5</span>, <span class="hljs-number">6</span>, <span class="hljs-number">7</span>, <span class="hljs-number">8</span>, <span class="hljs-number">9</span>],
    <span class="hljs-number">2</span>: [<span class="hljs-number">10</span>, <span class="hljs-number">11</span>, <span class="hljs-number">12</span>, <span class="hljs-number">13</span>, <span class="hljs-number">14</span>, <span class="hljs-number">15</span>, <span class="hljs-number">16</span>],
    <span class="hljs-number">3</span>: [<span class="hljs-number">17</span>, <span class="hljs-number">18</span>, <span class="hljs-number">19</span>, <span class="hljs-number">20</span>, <span class="hljs-number">21</span>, <span class="hljs-number">22</span>, <span class="hljs-number">23</span>],
}
model.parallelize(device_map)  <span class="hljs-comment"># Splits the model across several devices</span>
model.deparallelize()  <span class="hljs-comment"># Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()</span>`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function Ta(v){let n,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=u},l(o){n=m(o,"P",{"data-svelte-h":!0}),w(n)!=="svelte-fincs2"&&(n.innerHTML=u)},m(o,s){c(o,n,s)},p:j,d(o){o&&a(n)}}}function ba(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNVDVFbmNvZGVyTW9kZWwlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBNVDVFbmNvZGVyTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm10NS1zbWFsbCUyMiklMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIyU3R1ZGllcyUyMGhhdmUlMjBiZWVuJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzKSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MT5EncoderModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5EncoderModel.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function ya(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"JTIzJTIwSGVyZSUyMGlzJTIwYW4lMjBleGFtcGxlJTIwb2YlMjBhJTIwZGV2aWNlJTIwbWFwJTIwb24lMjBhJTIwbWFjaGluZSUyMHdpdGglMjA0JTIwR1BVcyUyMHVzaW5nJTIwbXQ1LXhsJTJDJTIwd2hpY2glMjBoYXMlMjBhJTIwdG90YWwlMjBvZiUyMDI0JTIwYXR0ZW50aW9uJTIwbW9kdWxlcyUzQSUwQW1vZGVsJTIwJTNEJTIwTVQ1Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtdDUteGwlMjIpJTBBZGV2aWNlX21hcCUyMCUzRCUyMCU3QiUwQSUyMCUyMCUyMCUyMDAlM0ElMjAlNUIwJTJDJTIwMSUyQyUyMDIlNUQlMkMlMEElMjAlMjAlMjAlMjAxJTNBJTIwJTVCMyUyQyUyMDQlMkMlMjA1JTJDJTIwNiUyQyUyMDclMkMlMjA4JTJDJTIwOSU1RCUyQyUwQSUyMCUyMCUyMCUyMDIlM0ElMjAlNUIxMCUyQyUyMDExJTJDJTIwMTIlMkMlMjAxMyUyQyUyMDE0JTJDJTIwMTUlMkMlMjAxNiU1RCUyQyUwQSUyMCUyMCUyMCUyMDMlM0ElMjAlNUIxNyUyQyUyMDE4JTJDJTIwMTklMkMlMjAyMCUyQyUyMDIxJTJDJTIwMjIlMkMlMjAyMyU1RCUyQyUwQSU3RCUwQW1vZGVsLnBhcmFsbGVsaXplKGRldmljZV9tYXAp",highlighted:`<span class="hljs-comment"># Here is an example of a device map on a machine with 4 GPUs using mt5-xl, which has a total of 24 attention modules:</span>
model = MT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;mt5-xl&quot;</span>)
device_map = {
    <span class="hljs-number">0</span>: [<span class="hljs-number">0</span>, <span class="hljs-number">1</span>, <span class="hljs-number">2</span>],
    <span class="hljs-number">1</span>: [<span class="hljs-number">3</span>, <span class="hljs-number">4</span>, <span class="hljs-number">5</span>, <span class="hljs-number">6</span>, <span class="hljs-number">7</span>, <span class="hljs-number">8</span>, <span class="hljs-number">9</span>],
    <span class="hljs-number">2</span>: [<span class="hljs-number">10</span>, <span class="hljs-number">11</span>, <span class="hljs-number">12</span>, <span class="hljs-number">13</span>, <span class="hljs-number">14</span>, <span class="hljs-number">15</span>, <span class="hljs-number">16</span>],
    <span class="hljs-number">3</span>: [<span class="hljs-number">17</span>, <span class="hljs-number">18</span>, <span class="hljs-number">19</span>, <span class="hljs-number">20</span>, <span class="hljs-number">21</span>, <span class="hljs-number">22</span>, <span class="hljs-number">23</span>],
}
model.parallelize(device_map)`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function wa(v){let n,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=u},l(o){n=m(o,"P",{"data-svelte-h":!0}),w(n)!=="svelte-fincs2"&&(n.innerHTML=u)},m(o,s){c(o,n,s)},p:j,d(o){o&&a(n)}}}function va(v){let n,u="Example of single-label classification:",o,s,f;return s=new W({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1UNUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBNVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyME1UNUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm10NS1zbWFsbCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MT5ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-ykxpe4"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function ka(v){let n,u="Example of multi-label classification:",o,s,f;return s=new W({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1UNUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBNVDVGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyME1UNUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRm10NS1zbWFsbCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MT5ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5ForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/mt5-small&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-1l8e32d"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function Ja(v){let n,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=u},l(o){n=m(o,"P",{"data-svelte-h":!0}),w(n)!=="svelte-fincs2"&&(n.innerHTML=u)},m(o,s){c(o,n,s)},p:j,d(o){o&&a(n)}}}function ja(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNVDVGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBNVDVGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZtdDUtc21hbGwlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MT5ForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5ForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)

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
...`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function Ua(v){let n,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=u},l(o){n=m(o,"P",{"data-svelte-h":!0}),w(n)!=="svelte-fincs2"&&(n.innerHTML=u)},m(o,s){c(o,n,s)},p:j,d(o){o&&a(n)}}}function $a(v){let n,u="Example:",o,s,f;return s=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNVDVGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbXQ1LXNtYWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwTVQ1Rm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm10NS1zbWFsbCUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MT5ForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MT5ForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google/mt5-small&quot;</span>)

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
...`,wrap:!1}}),{c(){n=p("p"),n.textContent=u,o=l(),g(s.$$.fragment)},l(t){n=m(t,"P",{"data-svelte-h":!0}),w(n)!=="svelte-11lpom8"&&(n.textContent=u),o=i(t),M(s.$$.fragment,t)},m(t,h){c(t,n,h),c(t,o,h),_(s,t,h),f=!0},p:j,i(t){f||(T(s.$$.fragment,t),f=!0)},o(t){b(s.$$.fragment,t),f=!1},d(t){t&&(a(n),a(o)),y(s,t)}}}function Ca(v){let n,u,o,s,f,t="<em>This model was released on 2020-10-22 and added to Hugging Face Transformers on 2020-11-17.</em>",h,U,pn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ve,de,fn,Ne,is='<a href="https://huggingface.co/papers/2010.11934" rel="nofollow">mT5</a> is a multilingual variant of <a href="./t5">T5</a>, training on 101 languages. It also incorporates a new “accidental translation” technique to prevent the model from incorrectly translating predictions into the wrong language.',gn,qe,ds='You can find all the original [mT5] checkpoints under the <a href="https://huggingface.co/collections/google/mt5-release-65005f1a520f8d7b4d039509" rel="nofollow">mT5</a> collection.',Mn,Me,_n,Re,cs='The example below demonstrates how to summarize text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Tn,_e,bn,Xe,ps='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',yn,Qe,ms='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to int4.',wn,Ee,vn,He,kn,Se,hs='<li>mT5 must be fine-tuned for downstream tasks because it was only pretrained on the <a href="https://huggingface.co/datasets/mc4" rel="nofollow">mc4</a> dataset.</li>',Jn,Ae,jn,K,De,Dn,Ct,us=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Model">MT5Model</a> or a <code>TFMT5Model</code>. It is used to
instantiate a mT5 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the mT5
<a href="https://huggingface.co/google/mt5-small" rel="nofollow">google/mt5-small</a> architecture.`,Yn,It,fs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Un,Ye,$n,Le,Pe,Cn,Oe,gs='See <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer">T5Tokenizer</a> for all details.',In,Ke,xn,et,tt,zn,nt,Ms='See <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5TokenizerFast">T5TokenizerFast</a> for all details.',Zn,ot,Gn,C,st,Ln,xt,_s="The bare Mt5 Model outputting raw hidden-states without any specific head on top.",Pn,zt,Ts=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,On,Zt,bs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Kn,ce,at,eo,Gt,ys="Moves the model to cpu from a model parallel state.",to,Te,no,H,rt,oo,Wt,ws='The <a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Model">MT5Model</a> forward method, overrides the <code>__call__</code> special method.',so,be,ao,ye,ro,S,lt,lo,Bt,vs="This is an experimental feature and is a subject to change at a moment’s notice.",io,Ft,ks=`Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
it will evenly distribute blocks across all devices.`,co,we,Wn,it,Bn,I,dt,po,Vt,Js="MT5 Model with a <code>language modeling</code> head on top.",mo,Nt,js=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ho,qt,Us=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,uo,pe,ct,fo,Rt,$s="Moves the model to cpu from a model parallel state.",go,ve,Mo,A,pt,_o,Xt,Cs='The <a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForConditionalGeneration">MT5ForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',To,ke,bo,Je,yo,D,mt,wo,Qt,Is="This is an experimental feature and is a subject to change at a moment’s notice.",vo,Et,xs=`Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
it will evenly distribute blocks across all devices.`,ko,je,Fn,ht,Vn,x,ut,Jo,Ht,zs="The bare Mt5 Model outputting raw hidden-states without any specific head on top.",jo,St,Zs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Uo,At,Gs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$o,me,ft,Co,Dt,Ws="Moves the model to cpu from a model parallel state.",Io,Ue,xo,Y,gt,zo,Yt,Bs='The <a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5EncoderModel">MT5EncoderModel</a> forward method, overrides the <code>__call__</code> special method.',Zo,$e,Go,Ce,Wo,L,Mt,Bo,Lt,Fs="This is an experimental feature and is a subject to change at a moment’s notice.",Fo,Pt,Vs=`Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
it will evenly distribute blocks across all devices.`,Vo,Ie,Nn,_t,qn,B,Tt,No,Ot,Ns=`MT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.`,qo,Kt,qs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ro,en,Rs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xo,N,bt,Qo,tn,Xs='The <a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForSequenceClassification">MT5ForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Eo,xe,Ho,ze,So,Ze,Rn,yt,Xn,F,wt,Ao,nn,Qs=`The Mt5 transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Do,on,Es=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Yo,sn,Hs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Lo,P,vt,Po,an,Ss='The <a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForTokenClassification">MT5ForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Oo,Ge,Ko,We,Qn,kt,En,V,Jt,es,rn,As=`The Mt5 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ts,ln,Ds=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ns,dn,Ys=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,os,O,jt,ss,cn,Ls='The <a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5ForQuestionAnswering">MT5ForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',as,Be,rs,Fe,Hn,Ut,Sn,mn,An;return de=new ie({props:{title:"mT5",local:"mt5",headingTag:"h1"}}),Me=new $t({props:{warning:!1,$$slots:{default:[aa]},$$scope:{ctx:v}}}),_e=new sa({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[da]},$$scope:{ctx:v}}}),Ee=new W({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBCaXRzQW5kQnl0ZXNDb25maWcoJTBBJTIwJTIwJTIwJTIwbG9hZF9pbl80Yml0JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3F1YW50X3R5cGUlM0QlMjJuZjQlMjIlMEEpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXEyU2VxTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmNzZWJ1ZXRubHAlMkZtVDVfbXVsdGlsaW5ndWFsX1hMU3VtJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRHF1YW50aXphdGlvbl9jb25maWclMEEpJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyY3NlYnVldG5scCUyRm1UNV9tdWx0aWxpbmd1YWxfWExTdW0lMjIlMEEpJTBBaW5wdXRfdGV4dCUyMCUzRCUyMCUyMiUyMiUyMlBsYW50cyUyMGFyZSUyMHJlbWFya2FibGUlMjBvcmdhbmlzbXMlMjB0aGF0JTIwcHJvZHVjZSUyMHRoZWlyJTIwb3duJTIwZm9vZCUyMHVzaW5nJTIwYSUyMG1ldGhvZCUyMGNhbGxlZCUyMHBob3Rvc3ludGhlc2lzLiUwQVRoaXMlMjBwcm9jZXNzJTIwaW52b2x2ZXMlMjBjb252ZXJ0aW5nJTIwc3VubGlnaHQlMkMlMjBjYXJib24lMjBkaW94aWRlJTJDJTIwYW5kJTIwd2F0ZXIlMjBpbnRvJTIwZ2x1Y29zZSUyQyUyMHdoaWNoJTIwcHJvdmlkZXMlMjBlbmVyZ3klMjBmb3IlMjBncm93dGguJTBBUGxhbnRzJTIwcGxheSUyMGElMjBjcnVjaWFsJTIwcm9sZSUyMGluJTIwc3VzdGFpbmluZyUyMGxpZmUlMjBvbiUyMEVhcnRoJTIwYnklMjBnZW5lcmF0aW5nJTIwb3h5Z2VuJTIwYW5kJTIwc2VydmluZyUyMGFzJTIwdGhlJTIwZm91bmRhdGlvbiUyMG9mJTIwbW9zdCUyMGVjb3N5c3RlbXMuJTIyJTIyJTIyJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKGlucHV0X3RleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dF9pZHMlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;csebuetnlp/mT5_multilingual_XLSum&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;csebuetnlp/mT5_multilingual_XLSum&quot;</span>
)
input_text = <span class="hljs-string">&quot;&quot;&quot;Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.&quot;&quot;&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),He=new ie({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Ae=new ie({props:{title:"MT5Config",local:"transformers.MT5Config",headingTag:"h2"}}),De=new $({props:{name:"class transformers.MT5Config",anchor:"transformers.MT5Config",parameters:[{name:"vocab_size",val:" = 250112"},{name:"d_model",val:" = 512"},{name:"d_kv",val:" = 64"},{name:"d_ff",val:" = 1024"},{name:"num_layers",val:" = 8"},{name:"num_decoder_layers",val:" = None"},{name:"num_heads",val:" = 6"},{name:"relative_attention_num_buckets",val:" = 32"},{name:"relative_attention_max_distance",val:" = 128"},{name:"dropout_rate",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-06"},{name:"initializer_factor",val:" = 1.0"},{name:"feed_forward_proj",val:" = 'gated-gelu'"},{name:"is_encoder_decoder",val:" = True"},{name:"use_cache",val:" = True"},{name:"tokenizer_class",val:" = 'T5Tokenizer'"},{name:"tie_word_embeddings",val:" = False"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"decoder_start_token_id",val:" = 0"},{name:"classifier_dropout",val:" = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MT5Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 250112) &#x2014;
Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Model">T5Model</a> or <code>TFT5Model</code>.`,name:"vocab_size"},{anchor:"transformers.MT5Config.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Size of the encoder layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.MT5Config.d_kv",description:`<strong>d_kv</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Size of the key, query, value projections per attention head. In the conventional context, it is typically expected that <code>d_kv</code> has to be equal to <code>d_model // num_heads</code>.
But in the architecture of mt5-small, <code>d_kv</code> is not equal to <code>d_model //num_heads</code>. The <code>inner_dim</code> of the projection layer will be defined as <code>num_heads * d_kv</code>.`,name:"d_kv"},{anchor:"transformers.MT5Config.d_ff",description:`<strong>d_ff</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Size of the intermediate feed forward layer in each <code>T5Block</code>.`,name:"d_ff"},{anchor:"transformers.MT5Config.num_layers",description:`<strong>num_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_layers"},{anchor:"transformers.MT5Config.num_decoder_layers",description:`<strong>num_decoder_layers</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Number of hidden layers in the Transformer decoder. Will use the same value as <code>num_layers</code> if not set.`,name:"num_decoder_layers"},{anchor:"transformers.MT5Config.num_heads",description:`<strong>num_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_heads"},{anchor:"transformers.MT5Config.relative_attention_num_buckets",description:`<strong>relative_attention_num_buckets</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of buckets to use for each attention layer.`,name:"relative_attention_num_buckets"},{anchor:"transformers.MT5Config.relative_attention_max_distance",description:`<strong>relative_attention_max_distance</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The maximum distance of the longer sequences for the bucket separation.`,name:"relative_attention_max_distance"},{anchor:"transformers.MT5Config.dropout_rate",description:`<strong>dropout_rate</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The ratio for all dropout layers.`,name:"dropout_rate"},{anchor:"transformers.MT5Config.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.MT5Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-6) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.MT5Config.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"},{anchor:"transformers.MT5Config.feed_forward_proj",description:`<strong>feed_forward_proj</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;gated-gelu&quot;</code>) &#x2014;
Type of feed forward layer to be used. Should be one of <code>&quot;relu&quot;</code> or <code>&quot;gated-gelu&quot;</code>.`,name:"feed_forward_proj"},{anchor:"transformers.MT5Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/configuration_mt5.py#L27"}}),Ye=new ie({props:{title:"MT5Tokenizer",local:"transformers.MT5Tokenizer",headingTag:"h2"}}),Pe=new $({props:{name:"class transformers.MT5Tokenizer",anchor:"transformers.MT5Tokenizer",parameters:[{name:"vocab_file",val:""},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"extra_ids",val:" = 100"},{name:"additional_special_tokens",val:" = None"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"legacy",val:" = None"},{name:"add_prefix_space",val:" = True"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/tokenization_mt5.py#L20"}}),Ke=new ie({props:{title:"MT5TokenizerFast",local:"transformers.MT5TokenizerFast",headingTag:"h2"}}),tt=new $({props:{name:"class transformers.MT5TokenizerFast",anchor:"transformers.MT5TokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"extra_ids",val:" = 100"},{name:"additional_special_tokens",val:" = None"},{name:"add_prefix_space",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/tokenization_mt5_fast.py#L20"}}),ot=new ie({props:{title:"MT5Model",local:"transformers.MT5Model",headingTag:"h2"}}),st=new $({props:{name:"class transformers.MT5Model",anchor:"transformers.MT5Model",parameters:[{name:"config",val:": MT5Config"}],parametersDescription:[{anchor:"transformers.MT5Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config">MT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1285"}}),at=new $({props:{name:"deparallelize",anchor:"transformers.MT5Model.deparallelize",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1352"}}),Te=new E({props:{anchor:"transformers.MT5Model.deparallelize.example",$$slots:{default:[ca]},$$scope:{ctx:v}}}),rt=new $({props:{name:"forward",anchor:"transformers.MT5Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.MT5Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./mt5#training">MT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.MT5Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MT5Model.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>MT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./mt5#training">MT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.MT5Model.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MT5Model.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MT5Model.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MT5Model.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MT5Model.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MT5Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MT5Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MT5Model.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MT5Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MT5Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MT5Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MT5Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MT5Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1390",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config"
>MT5Config</a>) and inputs.</p>
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
`}}),be=new $t({props:{$$slots:{default:[pa]},$$scope:{ctx:v}}}),ye=new E({props:{anchor:"transformers.MT5Model.forward.example",$$slots:{default:[ma]},$$scope:{ctx:v}}}),lt=new $({props:{name:"parallelize",anchor:"transformers.MT5Model.parallelize",parameters:[{name:"device_map",val:" = None"}],parametersDescription:[{anchor:"transformers.MT5Model.parallelize.device_map",description:`<strong>device_map</strong> (<code>dict[int, list]</code>, <em>optional</em>) &#x2014;
A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
automatically mapped to the first device (for esoteric reasons). That means that the first device should
have fewer attention modules mapped to it than other devices. For reference, the mt5 models have the
following number of attention modules:</p>
<ul>
<li>mt5-small: 6</li>
<li>mt5-base: 12</li>
<li>mt5-large: 24</li>
<li>mt5-xl: 24</li>
<li>mt5-xxl: 24</li>
</ul>`,name:"device_map"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1332"}}),we=new E({props:{anchor:"transformers.MT5Model.parallelize.example",$$slots:{default:[ha]},$$scope:{ctx:v}}}),it=new ie({props:{title:"MT5ForConditionalGeneration",local:"transformers.MT5ForConditionalGeneration",headingTag:"h2"}}),dt=new $({props:{name:"class transformers.MT5ForConditionalGeneration",anchor:"transformers.MT5ForConditionalGeneration",parameters:[{name:"config",val:": MT5Config"}],parametersDescription:[{anchor:"transformers.MT5ForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config">MT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1549"}}),ct=new $({props:{name:"deparallelize",anchor:"transformers.MT5ForConditionalGeneration.deparallelize",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1620"}}),ve=new E({props:{anchor:"transformers.MT5ForConditionalGeneration.deparallelize.example",$$slots:{default:[ua]},$$scope:{ctx:v}}}),pt=new $({props:{name:"forward",anchor:"transformers.MT5ForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.MT5ForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./mt5#training">MT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.MT5ForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MT5ForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>MT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./mt5#training">MT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.MT5ForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MT5ForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MT5ForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MT5ForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MT5ForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MT5ForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MT5ForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MT5ForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MT5ForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[-100, 0, ..., config.vocab_size - 1]</code>. All labels set to <code>-100</code> are ignored (masked), the loss is only computed for
labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MT5ForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MT5ForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MT5ForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MT5ForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MT5ForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1648",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config"
>MT5Config</a>) and inputs.</p>
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
`}}),ke=new $t({props:{$$slots:{default:[fa]},$$scope:{ctx:v}}}),Je=new E({props:{anchor:"transformers.MT5ForConditionalGeneration.forward.example",$$slots:{default:[ga]},$$scope:{ctx:v}}}),mt=new $({props:{name:"parallelize",anchor:"transformers.MT5ForConditionalGeneration.parallelize",parameters:[{name:"device_map",val:" = None"}],parametersDescription:[{anchor:"transformers.MT5ForConditionalGeneration.parallelize.device_map",description:`<strong>device_map</strong> (<code>dict[int, list]</code>, <em>optional</em>) &#x2014;
A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
automatically mapped to the first device (for esoteric reasons). That means that the first device should
have fewer attention modules mapped to it than other devices. For reference, the mt5 models have the
following number of attention modules:</p>
<ul>
<li>mt5-small: 6</li>
<li>mt5-base: 12</li>
<li>mt5-large: 24</li>
<li>mt5-xl: 24</li>
<li>mt5-xxl: 24</li>
</ul>`,name:"device_map"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1599"}}),je=new E({props:{anchor:"transformers.MT5ForConditionalGeneration.parallelize.example",$$slots:{default:[Ma]},$$scope:{ctx:v}}}),ht=new ie({props:{title:"MT5EncoderModel",local:"transformers.MT5EncoderModel",headingTag:"h2"}}),ut=new $({props:{name:"class transformers.MT5EncoderModel",anchor:"transformers.MT5EncoderModel",parameters:[{name:"config",val:": MT5Config"}],parametersDescription:[{anchor:"transformers.MT5EncoderModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config">MT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1847"}}),ft=new $({props:{name:"deparallelize",anchor:"transformers.MT5EncoderModel.deparallelize",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1902"}}),Ue=new E({props:{anchor:"transformers.MT5EncoderModel.deparallelize.example",$$slots:{default:[_a]},$$scope:{ctx:v}}}),gt=new $({props:{name:"forward",anchor:"transformers.MT5EncoderModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MT5EncoderModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./mt5#training">MT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.MT5EncoderModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MT5EncoderModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MT5EncoderModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MT5EncoderModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MT5EncoderModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MT5EncoderModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1937",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config"
>MT5Config</a>) and inputs.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),$e=new $t({props:{$$slots:{default:[Ta]},$$scope:{ctx:v}}}),Ce=new E({props:{anchor:"transformers.MT5EncoderModel.forward.example",$$slots:{default:[ba]},$$scope:{ctx:v}}}),Mt=new $({props:{name:"parallelize",anchor:"transformers.MT5EncoderModel.parallelize",parameters:[{name:"device_map",val:" = None"}],parametersDescription:[{anchor:"transformers.MT5EncoderModel.parallelize.device_map",description:`<strong>device_map</strong> (<code>dict[int, list]</code>, <em>optional</em>) &#x2014;
A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
automatically mapped to the first device (for esoteric reasons). That means that the first device should
have fewer attention modules mapped to it than other devices. For reference, the mt5 models have the
following number of attention modules:</p>
<ul>
<li>mt5-small: 6</li>
<li>mt5-base: 12</li>
<li>mt5-large: 24</li>
<li>mt5-xl: 24</li>
<li>mt5-xxl: 24</li>
</ul>`,name:"device_map"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1883"}}),Ie=new E({props:{anchor:"transformers.MT5EncoderModel.parallelize.example",$$slots:{default:[ya]},$$scope:{ctx:v}}}),_t=new ie({props:{title:"MT5ForSequenceClassification",local:"transformers.MT5ForSequenceClassification",headingTag:"h2"}}),Tt=new $({props:{name:"class transformers.MT5ForSequenceClassification",anchor:"transformers.MT5ForSequenceClassification",parameters:[{name:"config",val:": MT5Config"}],parametersDescription:[{anchor:"transformers.MT5ForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config">MT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L1993"}}),bt=new $({props:{name:"forward",anchor:"transformers.MT5ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MT5ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./mt5#training">MT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.MT5ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MT5ForSequenceClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>MT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./mt5#training">MT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.MT5ForSequenceClassification.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MT5ForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MT5ForSequenceClassification.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MT5ForSequenceClassification.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MT5ForSequenceClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MT5ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MT5ForSequenceClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MT5ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.MT5ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MT5ForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MT5ForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MT5ForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L2008",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config"
>MT5Config</a>) and inputs.</p>
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
`}}),xe=new $t({props:{$$slots:{default:[wa]},$$scope:{ctx:v}}}),ze=new E({props:{anchor:"transformers.MT5ForSequenceClassification.forward.example",$$slots:{default:[va]},$$scope:{ctx:v}}}),Ze=new E({props:{anchor:"transformers.MT5ForSequenceClassification.forward.example-2",$$slots:{default:[ka]},$$scope:{ctx:v}}}),yt=new ie({props:{title:"MT5ForTokenClassification",local:"transformers.MT5ForTokenClassification",headingTag:"h2"}}),wt=new $({props:{name:"class transformers.MT5ForTokenClassification",anchor:"transformers.MT5ForTokenClassification",parameters:[{name:"config",val:": MT5Config"}],parametersDescription:[{anchor:"transformers.MT5ForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config">MT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L2158"}}),vt=new $({props:{name:"forward",anchor:"transformers.MT5ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MT5ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. MT5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./t5#training">MT5 Training</a>.`,name:"input_ids"},{anchor:"transformers.MT5ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MT5ForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MT5ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MT5ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.MT5ForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MT5ForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MT5ForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L2173",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config"
>MT5Config</a>) and inputs.</p>
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
`}}),Ge=new $t({props:{$$slots:{default:[Ja]},$$scope:{ctx:v}}}),We=new E({props:{anchor:"transformers.MT5ForTokenClassification.forward.example",$$slots:{default:[ja]},$$scope:{ctx:v}}}),kt=new ie({props:{title:"MT5ForQuestionAnswering",local:"transformers.MT5ForQuestionAnswering",headingTag:"h2"}}),Jt=new $({props:{name:"class transformers.MT5ForQuestionAnswering",anchor:"transformers.MT5ForQuestionAnswering",parameters:[{name:"config",val:": MT5Config"}],parametersDescription:[{anchor:"transformers.MT5ForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config">MT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L2234"}}),jt=new $({props:{name:"forward",anchor:"transformers.MT5ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MT5ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./t5#training">T5 Training</a>.`,name:"input_ids"},{anchor:"transformers.MT5ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MT5ForQuestionAnswering.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>T5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./t5#training">T5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.MT5ForQuestionAnswering.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MT5ForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MT5ForQuestionAnswering.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MT5ForQuestionAnswering.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MT5ForQuestionAnswering.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MT5ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.MT5ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.MT5ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MT5ForQuestionAnswering.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MT5ForQuestionAnswering.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MT5ForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MT5ForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MT5ForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mt5/modeling_mt5.py#L2279",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput"
>transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mt5#transformers.MT5Config"
>MT5Config</a>) and inputs.</p>
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
`}}),Be=new $t({props:{$$slots:{default:[Ua]},$$scope:{ctx:v}}}),Fe=new E({props:{anchor:"transformers.MT5ForQuestionAnswering.forward.example",$$slots:{default:[$a]},$$scope:{ctx:v}}}),Ut=new oa({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mt5.md"}}),{c(){n=p("meta"),u=l(),o=p("p"),s=l(),f=p("p"),f.innerHTML=t,h=l(),U=p("div"),U.innerHTML=pn,Ve=l(),g(de.$$.fragment),fn=l(),Ne=p("p"),Ne.innerHTML=is,gn=l(),qe=p("p"),qe.innerHTML=ds,Mn=l(),g(Me.$$.fragment),_n=l(),Re=p("p"),Re.innerHTML=cs,Tn=l(),g(_e.$$.fragment),bn=l(),Xe=p("p"),Xe.innerHTML=ps,yn=l(),Qe=p("p"),Qe.innerHTML=ms,wn=l(),g(Ee.$$.fragment),vn=l(),g(He.$$.fragment),kn=l(),Se=p("ul"),Se.innerHTML=hs,Jn=l(),g(Ae.$$.fragment),jn=l(),K=p("div"),g(De.$$.fragment),Dn=l(),Ct=p("p"),Ct.innerHTML=us,Yn=l(),It=p("p"),It.innerHTML=fs,Un=l(),g(Ye.$$.fragment),$n=l(),Le=p("div"),g(Pe.$$.fragment),Cn=l(),Oe=p("p"),Oe.innerHTML=gs,In=l(),g(Ke.$$.fragment),xn=l(),et=p("div"),g(tt.$$.fragment),zn=l(),nt=p("p"),nt.innerHTML=Ms,Zn=l(),g(ot.$$.fragment),Gn=l(),C=p("div"),g(st.$$.fragment),Ln=l(),xt=p("p"),xt.textContent=_s,Pn=l(),zt=p("p"),zt.innerHTML=Ts,On=l(),Zt=p("p"),Zt.innerHTML=bs,Kn=l(),ce=p("div"),g(at.$$.fragment),eo=l(),Gt=p("p"),Gt.textContent=ys,to=l(),g(Te.$$.fragment),no=l(),H=p("div"),g(rt.$$.fragment),oo=l(),Wt=p("p"),Wt.innerHTML=ws,so=l(),g(be.$$.fragment),ao=l(),g(ye.$$.fragment),ro=l(),S=p("div"),g(lt.$$.fragment),lo=l(),Bt=p("p"),Bt.textContent=vs,io=l(),Ft=p("p"),Ft.textContent=ks,co=l(),g(we.$$.fragment),Wn=l(),g(it.$$.fragment),Bn=l(),I=p("div"),g(dt.$$.fragment),po=l(),Vt=p("p"),Vt.innerHTML=Js,mo=l(),Nt=p("p"),Nt.innerHTML=js,ho=l(),qt=p("p"),qt.innerHTML=Us,uo=l(),pe=p("div"),g(ct.$$.fragment),fo=l(),Rt=p("p"),Rt.textContent=$s,go=l(),g(ve.$$.fragment),Mo=l(),A=p("div"),g(pt.$$.fragment),_o=l(),Xt=p("p"),Xt.innerHTML=Cs,To=l(),g(ke.$$.fragment),bo=l(),g(Je.$$.fragment),yo=l(),D=p("div"),g(mt.$$.fragment),wo=l(),Qt=p("p"),Qt.textContent=Is,vo=l(),Et=p("p"),Et.textContent=xs,ko=l(),g(je.$$.fragment),Fn=l(),g(ht.$$.fragment),Vn=l(),x=p("div"),g(ut.$$.fragment),Jo=l(),Ht=p("p"),Ht.textContent=zs,jo=l(),St=p("p"),St.innerHTML=Zs,Uo=l(),At=p("p"),At.innerHTML=Gs,$o=l(),me=p("div"),g(ft.$$.fragment),Co=l(),Dt=p("p"),Dt.textContent=Ws,Io=l(),g(Ue.$$.fragment),xo=l(),Y=p("div"),g(gt.$$.fragment),zo=l(),Yt=p("p"),Yt.innerHTML=Bs,Zo=l(),g($e.$$.fragment),Go=l(),g(Ce.$$.fragment),Wo=l(),L=p("div"),g(Mt.$$.fragment),Bo=l(),Lt=p("p"),Lt.textContent=Fs,Fo=l(),Pt=p("p"),Pt.textContent=Vs,Vo=l(),g(Ie.$$.fragment),Nn=l(),g(_t.$$.fragment),qn=l(),B=p("div"),g(Tt.$$.fragment),No=l(),Ot=p("p"),Ot.textContent=Ns,qo=l(),Kt=p("p"),Kt.innerHTML=qs,Ro=l(),en=p("p"),en.innerHTML=Rs,Xo=l(),N=p("div"),g(bt.$$.fragment),Qo=l(),tn=p("p"),tn.innerHTML=Xs,Eo=l(),g(xe.$$.fragment),Ho=l(),g(ze.$$.fragment),So=l(),g(Ze.$$.fragment),Rn=l(),g(yt.$$.fragment),Xn=l(),F=p("div"),g(wt.$$.fragment),Ao=l(),nn=p("p"),nn.textContent=Qs,Do=l(),on=p("p"),on.innerHTML=Es,Yo=l(),sn=p("p"),sn.innerHTML=Hs,Lo=l(),P=p("div"),g(vt.$$.fragment),Po=l(),an=p("p"),an.innerHTML=Ss,Oo=l(),g(Ge.$$.fragment),Ko=l(),g(We.$$.fragment),Qn=l(),g(kt.$$.fragment),En=l(),V=p("div"),g(Jt.$$.fragment),es=l(),rn=p("p"),rn.innerHTML=As,ts=l(),ln=p("p"),ln.innerHTML=Ds,ns=l(),dn=p("p"),dn.innerHTML=Ys,os=l(),O=p("div"),g(jt.$$.fragment),ss=l(),cn=p("p"),cn.innerHTML=Ls,as=l(),g(Be.$$.fragment),rs=l(),g(Fe.$$.fragment),Hn=l(),g(Ut.$$.fragment),Sn=l(),mn=p("p"),this.h()},l(e){const r=ta("svelte-u9bgzb",document.head);n=m(r,"META",{name:!0,content:!0}),r.forEach(a),u=i(e),o=m(e,"P",{}),k(o).forEach(a),s=i(e),f=m(e,"P",{"data-svelte-h":!0}),w(f)!=="svelte-9014ff"&&(f.innerHTML=t),h=i(e),U=m(e,"DIV",{style:!0,"data-svelte-h":!0}),w(U)!=="svelte-wa5t4p"&&(U.innerHTML=pn),Ve=i(e),M(de.$$.fragment,e),fn=i(e),Ne=m(e,"P",{"data-svelte-h":!0}),w(Ne)!=="svelte-53baz4"&&(Ne.innerHTML=is),gn=i(e),qe=m(e,"P",{"data-svelte-h":!0}),w(qe)!=="svelte-tx4u5x"&&(qe.innerHTML=ds),Mn=i(e),M(Me.$$.fragment,e),_n=i(e),Re=m(e,"P",{"data-svelte-h":!0}),w(Re)!=="svelte-1q65a0t"&&(Re.innerHTML=cs),Tn=i(e),M(_e.$$.fragment,e),bn=i(e),Xe=m(e,"P",{"data-svelte-h":!0}),w(Xe)!=="svelte-nf5ooi"&&(Xe.innerHTML=ps),yn=i(e),Qe=m(e,"P",{"data-svelte-h":!0}),w(Qe)!=="svelte-11sw8fc"&&(Qe.innerHTML=ms),wn=i(e),M(Ee.$$.fragment,e),vn=i(e),M(He.$$.fragment,e),kn=i(e),Se=m(e,"UL",{"data-svelte-h":!0}),w(Se)!=="svelte-1tee7q6"&&(Se.innerHTML=hs),Jn=i(e),M(Ae.$$.fragment,e),jn=i(e),K=m(e,"DIV",{class:!0});var he=k(K);M(De.$$.fragment,he),Dn=i(he),Ct=m(he,"P",{"data-svelte-h":!0}),w(Ct)!=="svelte-imk6pp"&&(Ct.innerHTML=us),Yn=i(he),It=m(he,"P",{"data-svelte-h":!0}),w(It)!=="svelte-1ek1ss9"&&(It.innerHTML=fs),he.forEach(a),Un=i(e),M(Ye.$$.fragment,e),$n=i(e),Le=m(e,"DIV",{class:!0});var hn=k(Le);M(Pe.$$.fragment,hn),hn.forEach(a),Cn=i(e),Oe=m(e,"P",{"data-svelte-h":!0}),w(Oe)!=="svelte-6pkyex"&&(Oe.innerHTML=gs),In=i(e),M(Ke.$$.fragment,e),xn=i(e),et=m(e,"DIV",{class:!0});var un=k(et);M(tt.$$.fragment,un),un.forEach(a),zn=i(e),nt=m(e,"P",{"data-svelte-h":!0}),w(nt)!=="svelte-1nqabrh"&&(nt.innerHTML=Ms),Zn=i(e),M(ot.$$.fragment,e),Gn=i(e),C=m(e,"DIV",{class:!0});var z=k(C);M(st.$$.fragment,z),Ln=i(z),xt=m(z,"P",{"data-svelte-h":!0}),w(xt)!=="svelte-1a28rsm"&&(xt.textContent=_s),Pn=i(z),zt=m(z,"P",{"data-svelte-h":!0}),w(zt)!=="svelte-q52n56"&&(zt.innerHTML=Ts),On=i(z),Zt=m(z,"P",{"data-svelte-h":!0}),w(Zt)!=="svelte-hswkmf"&&(Zt.innerHTML=bs),Kn=i(z),ce=m(z,"DIV",{class:!0});var ue=k(ce);M(at.$$.fragment,ue),eo=i(ue),Gt=m(ue,"P",{"data-svelte-h":!0}),w(Gt)!=="svelte-ewr91v"&&(Gt.textContent=ys),to=i(ue),M(Te.$$.fragment,ue),ue.forEach(a),no=i(z),H=m(z,"DIV",{class:!0});var ee=k(H);M(rt.$$.fragment,ee),oo=i(ee),Wt=m(ee,"P",{"data-svelte-h":!0}),w(Wt)!=="svelte-185h0t9"&&(Wt.innerHTML=ws),so=i(ee),M(be.$$.fragment,ee),ao=i(ee),M(ye.$$.fragment,ee),ee.forEach(a),ro=i(z),S=m(z,"DIV",{class:!0});var te=k(S);M(lt.$$.fragment,te),lo=i(te),Bt=m(te,"P",{"data-svelte-h":!0}),w(Bt)!=="svelte-1wtkcqk"&&(Bt.textContent=vs),io=i(te),Ft=m(te,"P",{"data-svelte-h":!0}),w(Ft)!=="svelte-16fuwd4"&&(Ft.textContent=ks),co=i(te),M(we.$$.fragment,te),te.forEach(a),z.forEach(a),Wn=i(e),M(it.$$.fragment,e),Bn=i(e),I=m(e,"DIV",{class:!0});var Z=k(I);M(dt.$$.fragment,Z),po=i(Z),Vt=m(Z,"P",{"data-svelte-h":!0}),w(Vt)!=="svelte-1mzt99y"&&(Vt.innerHTML=Js),mo=i(Z),Nt=m(Z,"P",{"data-svelte-h":!0}),w(Nt)!=="svelte-q52n56"&&(Nt.innerHTML=js),ho=i(Z),qt=m(Z,"P",{"data-svelte-h":!0}),w(qt)!=="svelte-hswkmf"&&(qt.innerHTML=Us),uo=i(Z),pe=m(Z,"DIV",{class:!0});var fe=k(pe);M(ct.$$.fragment,fe),fo=i(fe),Rt=m(fe,"P",{"data-svelte-h":!0}),w(Rt)!=="svelte-ewr91v"&&(Rt.textContent=$s),go=i(fe),M(ve.$$.fragment,fe),fe.forEach(a),Mo=i(Z),A=m(Z,"DIV",{class:!0});var ne=k(A);M(pt.$$.fragment,ne),_o=i(ne),Xt=m(ne,"P",{"data-svelte-h":!0}),w(Xt)!=="svelte-gbioj3"&&(Xt.innerHTML=Cs),To=i(ne),M(ke.$$.fragment,ne),bo=i(ne),M(Je.$$.fragment,ne),ne.forEach(a),yo=i(Z),D=m(Z,"DIV",{class:!0});var oe=k(D);M(mt.$$.fragment,oe),wo=i(oe),Qt=m(oe,"P",{"data-svelte-h":!0}),w(Qt)!=="svelte-1wtkcqk"&&(Qt.textContent=Is),vo=i(oe),Et=m(oe,"P",{"data-svelte-h":!0}),w(Et)!=="svelte-16fuwd4"&&(Et.textContent=xs),ko=i(oe),M(je.$$.fragment,oe),oe.forEach(a),Z.forEach(a),Fn=i(e),M(ht.$$.fragment,e),Vn=i(e),x=m(e,"DIV",{class:!0});var G=k(x);M(ut.$$.fragment,G),Jo=i(G),Ht=m(G,"P",{"data-svelte-h":!0}),w(Ht)!=="svelte-1a28rsm"&&(Ht.textContent=zs),jo=i(G),St=m(G,"P",{"data-svelte-h":!0}),w(St)!=="svelte-q52n56"&&(St.innerHTML=Zs),Uo=i(G),At=m(G,"P",{"data-svelte-h":!0}),w(At)!=="svelte-hswkmf"&&(At.innerHTML=Gs),$o=i(G),me=m(G,"DIV",{class:!0});var ge=k(me);M(ft.$$.fragment,ge),Co=i(ge),Dt=m(ge,"P",{"data-svelte-h":!0}),w(Dt)!=="svelte-ewr91v"&&(Dt.textContent=Ws),Io=i(ge),M(Ue.$$.fragment,ge),ge.forEach(a),xo=i(G),Y=m(G,"DIV",{class:!0});var se=k(Y);M(gt.$$.fragment,se),zo=i(se),Yt=m(se,"P",{"data-svelte-h":!0}),w(Yt)!=="svelte-155gbab"&&(Yt.innerHTML=Bs),Zo=i(se),M($e.$$.fragment,se),Go=i(se),M(Ce.$$.fragment,se),se.forEach(a),Wo=i(G),L=m(G,"DIV",{class:!0});var ae=k(L);M(Mt.$$.fragment,ae),Bo=i(ae),Lt=m(ae,"P",{"data-svelte-h":!0}),w(Lt)!=="svelte-1wtkcqk"&&(Lt.textContent=Fs),Fo=i(ae),Pt=m(ae,"P",{"data-svelte-h":!0}),w(Pt)!=="svelte-16fuwd4"&&(Pt.textContent=Vs),Vo=i(ae),M(Ie.$$.fragment,ae),ae.forEach(a),G.forEach(a),Nn=i(e),M(_t.$$.fragment,e),qn=i(e),B=m(e,"DIV",{class:!0});var q=k(B);M(Tt.$$.fragment,q),No=i(q),Ot=m(q,"P",{"data-svelte-h":!0}),w(Ot)!=="svelte-176h55s"&&(Ot.textContent=Ns),qo=i(q),Kt=m(q,"P",{"data-svelte-h":!0}),w(Kt)!=="svelte-q52n56"&&(Kt.innerHTML=qs),Ro=i(q),en=m(q,"P",{"data-svelte-h":!0}),w(en)!=="svelte-hswkmf"&&(en.innerHTML=Rs),Xo=i(q),N=m(q,"DIV",{class:!0});var R=k(N);M(bt.$$.fragment,R),Qo=i(R),tn=m(R,"P",{"data-svelte-h":!0}),w(tn)!=="svelte-100qiqf"&&(tn.innerHTML=Xs),Eo=i(R),M(xe.$$.fragment,R),Ho=i(R),M(ze.$$.fragment,R),So=i(R),M(Ze.$$.fragment,R),R.forEach(a),q.forEach(a),Rn=i(e),M(yt.$$.fragment,e),Xn=i(e),F=m(e,"DIV",{class:!0});var X=k(F);M(wt.$$.fragment,X),Ao=i(X),nn=m(X,"P",{"data-svelte-h":!0}),w(nn)!=="svelte-1dr4kvi"&&(nn.textContent=Qs),Do=i(X),on=m(X,"P",{"data-svelte-h":!0}),w(on)!=="svelte-q52n56"&&(on.innerHTML=Es),Yo=i(X),sn=m(X,"P",{"data-svelte-h":!0}),w(sn)!=="svelte-hswkmf"&&(sn.innerHTML=Hs),Lo=i(X),P=m(X,"DIV",{class:!0});var re=k(P);M(vt.$$.fragment,re),Po=i(re),an=m(re,"P",{"data-svelte-h":!0}),w(an)!=="svelte-15f6r4b"&&(an.innerHTML=Ss),Oo=i(re),M(Ge.$$.fragment,re),Ko=i(re),M(We.$$.fragment,re),re.forEach(a),X.forEach(a),Qn=i(e),M(kt.$$.fragment,e),En=i(e),V=m(e,"DIV",{class:!0});var Q=k(V);M(Jt.$$.fragment,Q),es=i(Q),rn=m(Q,"P",{"data-svelte-h":!0}),w(rn)!=="svelte-1czz49"&&(rn.innerHTML=As),ts=i(Q),ln=m(Q,"P",{"data-svelte-h":!0}),w(ln)!=="svelte-q52n56"&&(ln.innerHTML=Ds),ns=i(Q),dn=m(Q,"P",{"data-svelte-h":!0}),w(dn)!=="svelte-hswkmf"&&(dn.innerHTML=Ys),os=i(Q),O=m(Q,"DIV",{class:!0});var le=k(O);M(jt.$$.fragment,le),ss=i(le),cn=m(le,"P",{"data-svelte-h":!0}),w(cn)!=="svelte-1ipnkyn"&&(cn.innerHTML=Ls),as=i(le),M(Be.$$.fragment,le),rs=i(le),M(Fe.$$.fragment,le),le.forEach(a),Q.forEach(a),Hn=i(e),M(Ut.$$.fragment,e),Sn=i(e),mn=m(e,"P",{}),k(mn).forEach(a),this.h()},h(){J(n,"name","hf:doc:metadata"),J(n,"content",Ia),na(U,"float","right"),J(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(et,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,r){d(document.head,n),c(e,u,r),c(e,o,r),c(e,s,r),c(e,f,r),c(e,h,r),c(e,U,r),c(e,Ve,r),_(de,e,r),c(e,fn,r),c(e,Ne,r),c(e,gn,r),c(e,qe,r),c(e,Mn,r),_(Me,e,r),c(e,_n,r),c(e,Re,r),c(e,Tn,r),_(_e,e,r),c(e,bn,r),c(e,Xe,r),c(e,yn,r),c(e,Qe,r),c(e,wn,r),_(Ee,e,r),c(e,vn,r),_(He,e,r),c(e,kn,r),c(e,Se,r),c(e,Jn,r),_(Ae,e,r),c(e,jn,r),c(e,K,r),_(De,K,null),d(K,Dn),d(K,Ct),d(K,Yn),d(K,It),c(e,Un,r),_(Ye,e,r),c(e,$n,r),c(e,Le,r),_(Pe,Le,null),c(e,Cn,r),c(e,Oe,r),c(e,In,r),_(Ke,e,r),c(e,xn,r),c(e,et,r),_(tt,et,null),c(e,zn,r),c(e,nt,r),c(e,Zn,r),_(ot,e,r),c(e,Gn,r),c(e,C,r),_(st,C,null),d(C,Ln),d(C,xt),d(C,Pn),d(C,zt),d(C,On),d(C,Zt),d(C,Kn),d(C,ce),_(at,ce,null),d(ce,eo),d(ce,Gt),d(ce,to),_(Te,ce,null),d(C,no),d(C,H),_(rt,H,null),d(H,oo),d(H,Wt),d(H,so),_(be,H,null),d(H,ao),_(ye,H,null),d(C,ro),d(C,S),_(lt,S,null),d(S,lo),d(S,Bt),d(S,io),d(S,Ft),d(S,co),_(we,S,null),c(e,Wn,r),_(it,e,r),c(e,Bn,r),c(e,I,r),_(dt,I,null),d(I,po),d(I,Vt),d(I,mo),d(I,Nt),d(I,ho),d(I,qt),d(I,uo),d(I,pe),_(ct,pe,null),d(pe,fo),d(pe,Rt),d(pe,go),_(ve,pe,null),d(I,Mo),d(I,A),_(pt,A,null),d(A,_o),d(A,Xt),d(A,To),_(ke,A,null),d(A,bo),_(Je,A,null),d(I,yo),d(I,D),_(mt,D,null),d(D,wo),d(D,Qt),d(D,vo),d(D,Et),d(D,ko),_(je,D,null),c(e,Fn,r),_(ht,e,r),c(e,Vn,r),c(e,x,r),_(ut,x,null),d(x,Jo),d(x,Ht),d(x,jo),d(x,St),d(x,Uo),d(x,At),d(x,$o),d(x,me),_(ft,me,null),d(me,Co),d(me,Dt),d(me,Io),_(Ue,me,null),d(x,xo),d(x,Y),_(gt,Y,null),d(Y,zo),d(Y,Yt),d(Y,Zo),_($e,Y,null),d(Y,Go),_(Ce,Y,null),d(x,Wo),d(x,L),_(Mt,L,null),d(L,Bo),d(L,Lt),d(L,Fo),d(L,Pt),d(L,Vo),_(Ie,L,null),c(e,Nn,r),_(_t,e,r),c(e,qn,r),c(e,B,r),_(Tt,B,null),d(B,No),d(B,Ot),d(B,qo),d(B,Kt),d(B,Ro),d(B,en),d(B,Xo),d(B,N),_(bt,N,null),d(N,Qo),d(N,tn),d(N,Eo),_(xe,N,null),d(N,Ho),_(ze,N,null),d(N,So),_(Ze,N,null),c(e,Rn,r),_(yt,e,r),c(e,Xn,r),c(e,F,r),_(wt,F,null),d(F,Ao),d(F,nn),d(F,Do),d(F,on),d(F,Yo),d(F,sn),d(F,Lo),d(F,P),_(vt,P,null),d(P,Po),d(P,an),d(P,Oo),_(Ge,P,null),d(P,Ko),_(We,P,null),c(e,Qn,r),_(kt,e,r),c(e,En,r),c(e,V,r),_(Jt,V,null),d(V,es),d(V,rn),d(V,ts),d(V,ln),d(V,ns),d(V,dn),d(V,os),d(V,O),_(jt,O,null),d(O,ss),d(O,cn),d(O,as),_(Be,O,null),d(O,rs),_(Fe,O,null),c(e,Hn,r),_(Ut,e,r),c(e,Sn,r),c(e,mn,r),An=!0},p(e,[r]){const he={};r&2&&(he.$$scope={dirty:r,ctx:e}),Me.$set(he);const hn={};r&2&&(hn.$$scope={dirty:r,ctx:e}),_e.$set(hn);const un={};r&2&&(un.$$scope={dirty:r,ctx:e}),Te.$set(un);const z={};r&2&&(z.$$scope={dirty:r,ctx:e}),be.$set(z);const ue={};r&2&&(ue.$$scope={dirty:r,ctx:e}),ye.$set(ue);const ee={};r&2&&(ee.$$scope={dirty:r,ctx:e}),we.$set(ee);const te={};r&2&&(te.$$scope={dirty:r,ctx:e}),ve.$set(te);const Z={};r&2&&(Z.$$scope={dirty:r,ctx:e}),ke.$set(Z);const fe={};r&2&&(fe.$$scope={dirty:r,ctx:e}),Je.$set(fe);const ne={};r&2&&(ne.$$scope={dirty:r,ctx:e}),je.$set(ne);const oe={};r&2&&(oe.$$scope={dirty:r,ctx:e}),Ue.$set(oe);const G={};r&2&&(G.$$scope={dirty:r,ctx:e}),$e.$set(G);const ge={};r&2&&(ge.$$scope={dirty:r,ctx:e}),Ce.$set(ge);const se={};r&2&&(se.$$scope={dirty:r,ctx:e}),Ie.$set(se);const ae={};r&2&&(ae.$$scope={dirty:r,ctx:e}),xe.$set(ae);const q={};r&2&&(q.$$scope={dirty:r,ctx:e}),ze.$set(q);const R={};r&2&&(R.$$scope={dirty:r,ctx:e}),Ze.$set(R);const X={};r&2&&(X.$$scope={dirty:r,ctx:e}),Ge.$set(X);const re={};r&2&&(re.$$scope={dirty:r,ctx:e}),We.$set(re);const Q={};r&2&&(Q.$$scope={dirty:r,ctx:e}),Be.$set(Q);const le={};r&2&&(le.$$scope={dirty:r,ctx:e}),Fe.$set(le)},i(e){An||(T(de.$$.fragment,e),T(Me.$$.fragment,e),T(_e.$$.fragment,e),T(Ee.$$.fragment,e),T(He.$$.fragment,e),T(Ae.$$.fragment,e),T(De.$$.fragment,e),T(Ye.$$.fragment,e),T(Pe.$$.fragment,e),T(Ke.$$.fragment,e),T(tt.$$.fragment,e),T(ot.$$.fragment,e),T(st.$$.fragment,e),T(at.$$.fragment,e),T(Te.$$.fragment,e),T(rt.$$.fragment,e),T(be.$$.fragment,e),T(ye.$$.fragment,e),T(lt.$$.fragment,e),T(we.$$.fragment,e),T(it.$$.fragment,e),T(dt.$$.fragment,e),T(ct.$$.fragment,e),T(ve.$$.fragment,e),T(pt.$$.fragment,e),T(ke.$$.fragment,e),T(Je.$$.fragment,e),T(mt.$$.fragment,e),T(je.$$.fragment,e),T(ht.$$.fragment,e),T(ut.$$.fragment,e),T(ft.$$.fragment,e),T(Ue.$$.fragment,e),T(gt.$$.fragment,e),T($e.$$.fragment,e),T(Ce.$$.fragment,e),T(Mt.$$.fragment,e),T(Ie.$$.fragment,e),T(_t.$$.fragment,e),T(Tt.$$.fragment,e),T(bt.$$.fragment,e),T(xe.$$.fragment,e),T(ze.$$.fragment,e),T(Ze.$$.fragment,e),T(yt.$$.fragment,e),T(wt.$$.fragment,e),T(vt.$$.fragment,e),T(Ge.$$.fragment,e),T(We.$$.fragment,e),T(kt.$$.fragment,e),T(Jt.$$.fragment,e),T(jt.$$.fragment,e),T(Be.$$.fragment,e),T(Fe.$$.fragment,e),T(Ut.$$.fragment,e),An=!0)},o(e){b(de.$$.fragment,e),b(Me.$$.fragment,e),b(_e.$$.fragment,e),b(Ee.$$.fragment,e),b(He.$$.fragment,e),b(Ae.$$.fragment,e),b(De.$$.fragment,e),b(Ye.$$.fragment,e),b(Pe.$$.fragment,e),b(Ke.$$.fragment,e),b(tt.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b(Te.$$.fragment,e),b(rt.$$.fragment,e),b(be.$$.fragment,e),b(ye.$$.fragment,e),b(lt.$$.fragment,e),b(we.$$.fragment,e),b(it.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(ve.$$.fragment,e),b(pt.$$.fragment,e),b(ke.$$.fragment,e),b(Je.$$.fragment,e),b(mt.$$.fragment,e),b(je.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(Ue.$$.fragment,e),b(gt.$$.fragment,e),b($e.$$.fragment,e),b(Ce.$$.fragment,e),b(Mt.$$.fragment,e),b(Ie.$$.fragment,e),b(_t.$$.fragment,e),b(Tt.$$.fragment,e),b(bt.$$.fragment,e),b(xe.$$.fragment,e),b(ze.$$.fragment,e),b(Ze.$$.fragment,e),b(yt.$$.fragment,e),b(wt.$$.fragment,e),b(vt.$$.fragment,e),b(Ge.$$.fragment,e),b(We.$$.fragment,e),b(kt.$$.fragment,e),b(Jt.$$.fragment,e),b(jt.$$.fragment,e),b(Be.$$.fragment,e),b(Fe.$$.fragment,e),b(Ut.$$.fragment,e),An=!1},d(e){e&&(a(u),a(o),a(s),a(f),a(h),a(U),a(Ve),a(fn),a(Ne),a(gn),a(qe),a(Mn),a(_n),a(Re),a(Tn),a(bn),a(Xe),a(yn),a(Qe),a(wn),a(vn),a(kn),a(Se),a(Jn),a(jn),a(K),a(Un),a($n),a(Le),a(Cn),a(Oe),a(In),a(xn),a(et),a(zn),a(nt),a(Zn),a(Gn),a(C),a(Wn),a(Bn),a(I),a(Fn),a(Vn),a(x),a(Nn),a(qn),a(B),a(Rn),a(Xn),a(F),a(Qn),a(En),a(V),a(Hn),a(Sn),a(mn)),a(n),y(de,e),y(Me,e),y(_e,e),y(Ee,e),y(He,e),y(Ae,e),y(De),y(Ye,e),y(Pe),y(Ke,e),y(tt),y(ot,e),y(st),y(at),y(Te),y(rt),y(be),y(ye),y(lt),y(we),y(it,e),y(dt),y(ct),y(ve),y(pt),y(ke),y(Je),y(mt),y(je),y(ht,e),y(ut),y(ft),y(Ue),y(gt),y($e),y(Ce),y(Mt),y(Ie),y(_t,e),y(Tt),y(bt),y(xe),y(ze),y(Ze),y(yt,e),y(wt),y(vt),y(Ge),y(We),y(kt,e),y(Jt),y(jt),y(Be),y(Fe),y(Ut,e)}}}const Ia='{"title":"mT5","local":"mt5","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"MT5Config","local":"transformers.MT5Config","sections":[],"depth":2},{"title":"MT5Tokenizer","local":"transformers.MT5Tokenizer","sections":[],"depth":2},{"title":"MT5TokenizerFast","local":"transformers.MT5TokenizerFast","sections":[],"depth":2},{"title":"MT5Model","local":"transformers.MT5Model","sections":[],"depth":2},{"title":"MT5ForConditionalGeneration","local":"transformers.MT5ForConditionalGeneration","sections":[],"depth":2},{"title":"MT5EncoderModel","local":"transformers.MT5EncoderModel","sections":[],"depth":2},{"title":"MT5ForSequenceClassification","local":"transformers.MT5ForSequenceClassification","sections":[],"depth":2},{"title":"MT5ForTokenClassification","local":"transformers.MT5ForTokenClassification","sections":[],"depth":2},{"title":"MT5ForQuestionAnswering","local":"transformers.MT5ForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function xa(v){return Os(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class qa extends Ks{constructor(n){super(),ea(this,n,xa,Ca,Ps,{})}}export{qa as component};
