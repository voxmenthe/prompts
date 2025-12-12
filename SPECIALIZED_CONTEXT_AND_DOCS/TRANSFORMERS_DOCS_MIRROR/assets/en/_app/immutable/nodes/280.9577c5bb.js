import{s as qr,o as Ir,n as j}from"../chunks/scheduler.18a86fab.js";import{S as Zr,i as Wr,g as l,s,r as f,m as zr,A as Rr,h as d,f as i,c as a,j as w,x as u,u as g,n as jr,k as $,l as Vr,y as o,a as m,v as _,d as b,t as y,w as M}from"../chunks/index.98837b22.js";import{T as Kn}from"../chunks/Tip.77304350.js";import{D as J}from"../chunks/Docstring.a1ef7999.js";import{C as X}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as se}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as K,E as Xr}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Nr,a as Fr}from"../chunks/HfOption.6641485e.js";function Gr(v){let n,k="Click on the mBART models in the right sidebar for more examples of applying mBART to different language tasks.";return{c(){n=l("p"),n.textContent=k},l(r){n=d(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-1yiggqa"&&(n.textContent=k)},m(r,c){m(r,n,c)},p:j,d(r){r&&i(n)}}}function Qr(v){let n,k;return n=new X({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0cmFuc2xhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyZmFjZWJvb2slMkZtYmFydC1sYXJnZS01MC1tYW55LXRvLW1hbnktbW10JTIyJTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMHNyY19sYW5nJTNEJTIyZW5fWFglMjIlMkMlMEElMjAlMjAlMjAlMjB0Z3RfbGFuZyUzRCUyMmZyX1hYJTIyJTJDJTBBKSUwQXByaW50KHBpcGVsaW5lKCUyMlVOJTIwQ2hpZWYlMjBTYXlzJTIwVGhlcmUlMjBJcyUyME5vJTIwTWlsaXRhcnklMjBTb2x1dGlvbiUyMGluJTIwU3lyaWElMjIpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;translation&quot;</span>,
    model=<span class="hljs-string">&quot;facebook/mbart-large-50-many-to-many-mmt&quot;</span>,
    device=<span class="hljs-number">0</span>,
    dtype=torch.float16,
    src_lang=<span class="hljs-string">&quot;en_XX&quot;</span>,
    tgt_lang=<span class="hljs-string">&quot;fr_XX&quot;</span>,
)
<span class="hljs-built_in">print</span>(pipeline(<span class="hljs-string">&quot;UN Chief Says There Is No Military Solution in Syria&quot;</span>))`,wrap:!1}}),{c(){f(n.$$.fragment)},l(r){g(n.$$.fragment,r)},m(r,c){_(n,r,c),k=!0},p:j,i(r){k||(b(n.$$.fragment,r),k=!0)},o(r){y(n.$$.fragment,r),k=!1},d(r){M(n,r)}}}function Sr(v){let n,k;return n=new X({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQWFydGljbGVfZW4lMjAlM0QlMjAlMjJVTiUyMENoaWVmJTIwU2F5cyUyMFRoZXJlJTIwSXMlMjBObyUyME1pbGl0YXJ5JTIwU29sdXRpb24lMjBpbiUyMFN5cmlhJTIyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXEyU2VxTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtNTAtbWFueS10by1tYW55LW1tdCUyMiUyQyUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUyQyUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtNTAtbWFueS10by1tYW55LW1tdCUyMiklMEElMEF0b2tlbml6ZXIuc3JjX2xhbmclMjAlM0QlMjAlMjJlbl9YWCUyMiUwQWVuY29kZWRfaGklMjAlM0QlMjB0b2tlbml6ZXIoYXJ0aWNsZV9lbiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEFnZW5lcmF0ZWRfdG9rZW5zJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKiplbmNvZGVkX2hpJTJDJTIwZm9yY2VkX2Jvc190b2tlbl9pZCUzRHRva2VuaXplci5sYW5nX2NvZGVfdG9faWQlNUIlMjJmcl9YWCUyMiU1RCUyQyUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyKSUwQXByaW50KHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

article_en = <span class="hljs-string">&quot;UN Chief Says There Is No Military Solution in Syria&quot;</span>

model = AutoModelForSeq2SeqLM.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-50-many-to-many-mmt&quot;</span>, dtype=torch.bfloat16, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-50-many-to-many-mmt&quot;</span>)

tokenizer.src_lang = <span class="hljs-string">&quot;en_XX&quot;</span>
encoded_hi = tokenizer(article_en, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id[<span class="hljs-string">&quot;fr_XX&quot;</span>], cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.batch_decode(generated_tokens, skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){f(n.$$.fragment)},l(r){g(n.$$.fragment,r)},m(r,c){_(n,r,c),k=!0},p:j,i(r){k||(b(n.$$.fragment,r),k=!0)},o(r){y(n.$$.fragment,r),k=!1},d(r){M(n,r)}}}function Lr(v){let n,k,r,c;return n=new Fr({props:{id:"usage",option:"Pipeline",$$slots:{default:[Qr]},$$scope:{ctx:v}}}),r=new Fr({props:{id:"usage",option:"AutoModel",$$slots:{default:[Sr]},$$scope:{ctx:v}}}),{c(){f(n.$$.fragment),k=s(),f(r.$$.fragment)},l(h){g(n.$$.fragment,h),k=a(h),g(r.$$.fragment,h)},m(h,t){_(n,h,t),m(h,k,t),_(r,h,t),c=!0},p(h,t){const T={};t&2&&(T.$$scope={dirty:t,ctx:h}),n.$set(T);const ee={};t&2&&(ee.$$scope={dirty:t,ctx:h}),r.$set(ee)},i(h){c||(b(n.$$.fragment,h),b(r.$$.fragment,h),c=!0)},o(h){y(n.$$.fragment,h),y(r.$$.fragment,h),c=!1},d(h){h&&i(k),M(n,h),M(r,h)}}}function Er(v){let n,k="Example:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1CYXJ0Q29uZmlnJTJDJTIwTUJhcnRNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBNQkFSVCUyMGZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtY2MyNSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBNQmFydENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBmYWNlYm9vayUyRm1iYXJ0LWxhcmdlLWNjMjUlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyME1CYXJ0TW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MBartConfig, MBartModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a MBART facebook/mbart-large-cc25 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MBartConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the facebook/mbart-large-cc25 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-11lpom8"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function Yr(v){let n,k="Examples:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1CYXJ0VG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwTUJhcnRUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtZW4tcm8lMjIlMkMlMjBzcmNfbGFuZyUzRCUyMmVuX1hYJTIyJTJDJTIwdGd0X2xhbmclM0QlMjJyb19STyUyMiklMEFleGFtcGxlX2VuZ2xpc2hfcGhyYXNlJTIwJTNEJTIwJTIyJTIwVU4lMjBDaGllZiUyMFNheXMlMjBUaGVyZSUyMElzJTIwTm8lMjBNaWxpdGFyeSUyMFNvbHV0aW9uJTIwaW4lMjBTeXJpYSUyMiUwQWV4cGVjdGVkX3RyYW5zbGF0aW9uX3JvbWFuaWFuJTIwJTNEJTIwJTIyJUM1JTlFZWZ1bCUyME9OVSUyMGRlY2xhciVDNCU4MyUyMGMlQzQlODMlMjBudSUyMGV4aXN0JUM0JTgzJTIwbyUyMHNvbHUlQzUlQTNpZSUyMG1pbGl0YXIlQzQlODMlMjAlQzMlQUVuJTIwU2lyaWElMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoZXhhbXBsZV9lbmdsaXNoX3BocmFzZSUyQyUyMHRleHRfdGFyZ2V0JTNEZXhwZWN0ZWRfdHJhbnNsYXRpb25fcm9tYW5pYW4lMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MBartTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MBartTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-en-ro&quot;</span>, src_lang=<span class="hljs-string">&quot;en_XX&quot;</span>, tgt_lang=<span class="hljs-string">&quot;ro_RO&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example_english_phrase = <span class="hljs-string">&quot; UN Chief Says There Is No Military Solution in Syria&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_translation_romanian = <span class="hljs-string">&quot;Şeful ONU declară că nu există o soluţie militară în Siria&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-kvfsh7"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function Hr(v){let n,k="Examples:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1CYXJ0VG9rZW5pemVyRmFzdCUwQSUwQXRva2VuaXplciUyMCUzRCUyME1CYXJ0VG9rZW5pemVyRmFzdC5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZmFjZWJvb2slMkZtYmFydC1sYXJnZS1lbi1ybyUyMiUyQyUyMHNyY19sYW5nJTNEJTIyZW5fWFglMjIlMkMlMjB0Z3RfbGFuZyUzRCUyMnJvX1JPJTIyJTBBKSUwQWV4YW1wbGVfZW5nbGlzaF9waHJhc2UlMjAlM0QlMjAlMjIlMjBVTiUyMENoaWVmJTIwU2F5cyUyMFRoZXJlJTIwSXMlMjBObyUyME1pbGl0YXJ5JTIwU29sdXRpb24lMjBpbiUyMFN5cmlhJTIyJTBBZXhwZWN0ZWRfdHJhbnNsYXRpb25fcm9tYW5pYW4lMjAlM0QlMjAlMjIlQzUlOUVlZnVsJTIwT05VJTIwZGVjbGFyJUM0JTgzJTIwYyVDNCU4MyUyMG51JTIwZXhpc3QlQzQlODMlMjBvJTIwc29sdSVDNSVBM2llJTIwbWlsaXRhciVDNCU4MyUyMCVDMyVBRW4lMjBTaXJpYSUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihleGFtcGxlX2VuZ2xpc2hfcGhyYXNlJTJDJTIwdGV4dF90YXJnZXQlM0RleHBlY3RlZF90cmFuc2xhdGlvbl9yb21hbmlhbiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MBartTokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MBartTokenizerFast.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/mbart-large-en-ro&quot;</span>, src_lang=<span class="hljs-string">&quot;en_XX&quot;</span>, tgt_lang=<span class="hljs-string">&quot;ro_RO&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example_english_phrase = <span class="hljs-string">&quot; UN Chief Says There Is No Military Solution in Syria&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_translation_romanian = <span class="hljs-string">&quot;Şeful ONU declară că nu există o soluţie militară în Siria&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-kvfsh7"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function Dr(v){let n,k="Examples:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1CYXJ0NTBUb2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBNQmFydDUwVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm1iYXJ0LWxhcmdlLTUwJTIyJTJDJTIwc3JjX2xhbmclM0QlMjJlbl9YWCUyMiUyQyUyMHRndF9sYW5nJTNEJTIycm9fUk8lMjIpJTBBc3JjX3RleHQlMjAlM0QlMjAlMjIlMjBVTiUyMENoaWVmJTIwU2F5cyUyMFRoZXJlJTIwSXMlMjBObyUyME1pbGl0YXJ5JTIwU29sdXRpb24lMjBpbiUyMFN5cmlhJTIyJTBBdGd0X3RleHQlMjAlM0QlMjAlMjIlQzUlOUVlZnVsJTIwT05VJTIwZGVjbGFyJUM0JTgzJTIwYyVDNCU4MyUyMG51JTIwZXhpc3QlQzQlODMlMjBvJTIwc29sdSVDNSVBM2llJTIwbWlsaXRhciVDNCU4MyUyMCVDMyVBRW4lMjBTaXJpYSUyMiUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplcihzcmNfdGV4dCUyQyUyMHRleHRfdGFyZ2V0JTNEdGd0X3RleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUyMyUyMG1vZGVsKCoqbW9kZWxfaW5wdXRzKSUyMHNob3VsZCUyMHdvcms=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MBart50Tokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MBart50Tokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-50&quot;</span>, src_lang=<span class="hljs-string">&quot;en_XX&quot;</span>, tgt_lang=<span class="hljs-string">&quot;ro_RO&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>src_text = <span class="hljs-string">&quot; UN Chief Says There Is No Military Solution in Syria&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>tgt_text = <span class="hljs-string">&quot;Şeful ONU declară că nu există o soluţie militară în Siria&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># model(**model_inputs) should work</span>`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-kvfsh7"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function Ar(v){let n,k="Examples:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1CYXJ0NTBUb2tlbml6ZXJGYXN0JTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwTUJhcnQ1MFRva2VuaXplckZhc3QuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtNTAlMjIlMkMlMjBzcmNfbGFuZyUzRCUyMmVuX1hYJTIyJTJDJTIwdGd0X2xhbmclM0QlMjJyb19STyUyMiklMEFzcmNfdGV4dCUyMCUzRCUyMCUyMiUyMFVOJTIwQ2hpZWYlMjBTYXlzJTIwVGhlcmUlMjBJcyUyME5vJTIwTWlsaXRhcnklMjBTb2x1dGlvbiUyMGluJTIwU3lyaWElMjIlMEF0Z3RfdGV4dCUyMCUzRCUyMCUyMiVDNSU5RWVmdWwlMjBPTlUlMjBkZWNsYXIlQzQlODMlMjBjJUM0JTgzJTIwbnUlMjBleGlzdCVDNCU4MyUyMG8lMjBzb2x1JUM1JUEzaWUlMjBtaWxpdGFyJUM0JTgzJTIwJUMzJUFFbiUyMFNpcmlhJTIyJTBBbW9kZWxfaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHNyY190ZXh0JTJDJTIwdGV4dF90YXJnZXQlM0R0Z3RfdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTIzJTIwbW9kZWwoKiptb2RlbF9pbnB1dHMpJTIwc2hvdWxkJTIwd29yaw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MBart50TokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = MBart50TokenizerFast.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-50&quot;</span>, src_lang=<span class="hljs-string">&quot;en_XX&quot;</span>, tgt_lang=<span class="hljs-string">&quot;ro_RO&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>src_text = <span class="hljs-string">&quot; UN Chief Says There Is No Military Solution in Syria&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>tgt_text = <span class="hljs-string">&quot;Şeful ONU declară că nu există o soluţie militară în Siria&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># model(**model_inputs) should work</span>`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-kvfsh7"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function Or(v){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=k},l(r){n=d(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(r,c){m(r,n,c)},p:j,d(r){r&&i(n)}}}function Pr(v){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=k},l(r){n=d(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(r,c){m(r,n,c)},p:j,d(r){r&&i(n)}}}function Kr(v){let n,k="Example Translation:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNQmFydEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQW1vZGVsJTIwJTNEJTIwTUJhcnRGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtZW4tcm8lMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYmFydC1sYXJnZS1lbi1ybyUyMiklMEElMEFleGFtcGxlX2VuZ2xpc2hfcGhyYXNlJTIwJTNEJTIwJTIyNDIlMjBpcyUyMHRoZSUyMGFuc3dlciUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihleGFtcGxlX2VuZ2xpc2hfcGhyYXNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBUcmFuc2xhdGUlMEFnZW5lcmF0ZWRfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBudW1fYmVhbXMlM0Q0JTJDJTIwbWF4X2xlbmd0aCUzRDUpJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZWRfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MBartForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-en-ro&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-en-ro&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>example_english_phrase = <span class="hljs-string">&quot;42 is the answer&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(example_english_phrase, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Translate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(**inputs, num_beams=<span class="hljs-number">4</span>, max_length=<span class="hljs-number">5</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&#x27;42 este răspuns&#x27;</span>`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-hvxwgb"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function ei(v){let n,k="Mask filling example:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNQmFydEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQW1vZGVsJTIwJTNEJTIwTUJhcnRGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtY2MyNSUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm1iYXJ0LWxhcmdlLWNjMjUlMjIpJTBBJTBBJTIzJTIwZGVfREUlMjBpcyUyMHRoZSUyMGxhbmd1YWdlJTIwc3ltYm9sJTIwaWQlMjAlM0NMSUQlM0UlMjBmb3IlMjBHZXJtYW4lMEFUWFQlMjAlM0QlMjAlMjIlM0MlMkZzJTNFJTIwTWVpbmUlMjBGcmV1bmRlJTIwc2luZCUyMCUzQ21hc2slM0UlMjBuZXR0JTIwYWJlciUyMHNpZSUyMGVzc2VuJTIwenUlMjB2aWVsJTIwS3VjaGVuLiUyMCUzQyUyRnMlM0UlMjBkZV9ERSUyMiUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglNUJUWFQlNUQlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBbG9naXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzKS5sb2dpdHMlMEElMEFtYXNrZWRfaW5kZXglMjAlM0QlMjAoaW5wdXRfaWRzJTVCMCU1RCUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKS5ub256ZXJvKCkuaXRlbSgpJTBBcHJvYnMlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza2VkX2luZGV4JTVELnNvZnRtYXgoZGltJTNEMCklMEF2YWx1ZXMlMkMlMjBwcmVkaWN0aW9ucyUyMCUzRCUyMHByb2JzLnRvcGsoNSklMEElMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3Rpb25zKS5zcGxpdCgp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MBartForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># de_DE is the language symbol id &lt;LID&gt; for German</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>TXT = <span class="hljs-string">&quot;&lt;/s&gt; Meine Freunde sind &lt;mask&gt; nett aber sie essen zu viel Kuchen. &lt;/s&gt; de_DE&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer([TXT], add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(input_ids).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>masked_index = (input_ids[<span class="hljs-number">0</span>] == tokenizer.mask_token_id).nonzero().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>probs = logits[<span class="hljs-number">0</span>, masked_index].softmax(dim=<span class="hljs-number">0</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>values, predictions = probs.topk(<span class="hljs-number">5</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predictions).split()
[<span class="hljs-string">&#x27;nett&#x27;</span>, <span class="hljs-string">&#x27;sehr&#x27;</span>, <span class="hljs-string">&#x27;ganz&#x27;</span>, <span class="hljs-string">&#x27;nicht&#x27;</span>, <span class="hljs-string">&#x27;so&#x27;</span>]`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-1p9uukt"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function ti(v){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=k},l(r){n=d(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(r,c){m(r,n,c)},p:j,d(r){r&&i(n)}}}function ni(v){let n,k="Example:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNQmFydEZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm1iYXJ0LWxhcmdlLWNjMjUlMjIpJTBBbW9kZWwlMjAlM0QlMjBNQmFydEZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm1iYXJ0LWxhcmdlLWNjMjUlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MBartForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>)

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
...`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-11lpom8"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function oi(v){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=k},l(r){n=d(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(r,c){m(r,n,c)},p:j,d(r){r&&i(n)}}}function si(v){let n,k="Example of single-label classification:",r,c,h;return c=new X({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1CYXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtY2MyNSUyMiklMEFtb2RlbCUyMCUzRCUyME1CYXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYmFydC1sYXJnZS1jYzI1JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBNQmFydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtY2MyNSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MBartForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-ykxpe4"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function ai(v){let n,k="Example of multi-label classification:",r,c,h;return c=new X({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1CYXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtY2MyNSUyMiklMEFtb2RlbCUyMCUzRCUyME1CYXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYmFydC1sYXJnZS1jYzI1JTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBNQmFydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtY2MyNSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MBartForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-1l8e32d"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function ri(v){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=k},l(r){n=d(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(r,c){m(r,n,c)},p:j,d(r){r&&i(n)}}}function ii(v){let n,k="Example:",r,c,h;return c=new X({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNQmFydEZvckNhdXNhbExNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYmFydC1sYXJnZS1jYzI1JTIyKSUwQW1vZGVsJTIwJTNEJTIwTUJhcnRGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYmFydC1sYXJnZS1jYzI1JTIyJTJDJTIwYWRkX2Nyb3NzX2F0dGVudGlvbiUzREZhbHNlKSUwQWFzc2VydCUyMG1vZGVsLmNvbmZpZy5pc19kZWNvZGVyJTJDJTIwZiUyMiU3Qm1vZGVsLl9fY2xhc3NfXyU3RCUyMGhhcyUyMHRvJTIwYmUlMjBjb25maWd1cmVkJTIwYXMlMjBhJTIwZGVjb2Rlci4lMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBZXhwZWN0ZWRfc2hhcGUlMjAlM0QlMjAlNUIxJTJDJTIwaW5wdXRzLmlucHV0X2lkcy5zaGFwZSU1Qi0xJTVEJTJDJTIwbW9kZWwuY29uZmlnLnZvY2FiX3NpemUlNUQlMEFsaXN0KGxvZ2l0cy5zaGFwZSklMjAlM0QlM0QlMjBleHBlY3RlZF9zaGFwZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MBartForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MBartForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-cc25&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_shape = [<span class="hljs-number">1</span>, inputs.input_ids.shape[-<span class="hljs-number">1</span>], model.config.vocab_size]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape) == expected_shape
<span class="hljs-literal">True</span>`,wrap:!1}}),{c(){n=l("p"),n.textContent=k,r=s(),f(c.$$.fragment)},l(t){n=d(t,"P",{"data-svelte-h":!0}),u(n)!=="svelte-11lpom8"&&(n.textContent=k),r=a(t),g(c.$$.fragment,t)},m(t,T){m(t,n,T),m(t,r,T),_(c,t,T),h=!0},p:j,i(t){h||(b(c.$$.fragment,t),h=!0)},o(t){y(c.$$.fragment,t),h=!1},d(t){t&&(i(n),i(r)),M(c,t)}}}function li(v){let n,k,r,c,h,t="<em>This model was released on 2020-01-22 and added to Hugging Face Transformers on 2020-11-16.</em>",T,ee,Ta='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',to,Qe,no,Se,va='<a href="https://huggingface.co/papers/2001.08210" rel="nofollow">mBART</a> is a multilingual machine translation model that pretrains the entire translation model (encoder-decoder) unlike previous methods that only focused on parts of the model. The model is trained on a denoising objective which reconstructs the corrupted text. This allows mBART to handle the source language and the target text to translate to.',oo,Le,wa='<a href="https://huggingface.co/paper/2008.00401" rel="nofollow">mBART-50</a> is pretrained on an additional 25 languages.',so,Ee,$a='You can find all the original mBART checkpoints under the <a href="https://huggingface.co/facebook?search_models=mbart" rel="nofollow">AI at Meta</a> organization.',ao,ue,ro,Ye,Ja=`<p>[!NOTE]
The <code>head_mask</code> argument is ignored when using all attention implementation other than “eager”. If you have a <code>head_mask</code> and want it to have effect, load the model with <code>XXXModel.from_pretrained(model_id, attn_implementation=&quot;eager&quot;)</code></p>`,io,He,xa='The example below demonstrates how to translate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',lo,fe,co,De,po,N,Ht,Ua="<p>You can check the full list of language codes via <code>tokenizer.lang_code_to_id.keys()</code>.</p>",Ro,Dt,Ba="<p>mBART requires a special language id token in the source and target text during training. The source text format is <code>X [eos, src_lang_code]</code> where <code>X</code> is the source text. The target text format is <code>[tgt_lang_code] X [eos]</code>. The <code>bos</code> token is never used. The <code>~PreTrainedTokenizerBase._call_</code> encodes the source text format passed as the first argument or with the <code>text</code> keyword. The target text format is passed with the <code>text_label</code> keyword.</p>",Vo,Ae,At,Ca="Set the <code>decoder_start_token_id</code> to the target language id for mBART.",Xo,Oe,No,Ot,za="<p>mBART-50 has a different text format. The language id token is used as the prefix for the source and target text. The text format is <code>[lang_code] X [eos]</code> where <code>lang_code</code> is the source language id for the source text and target language id for the target text. <code>X</code> is the source or target text respectively.</p>",Go,Pe,Pt,ja='Set the <code>eos_token_id</code> as the <code>decoder_start_token_id</code> for mBART-50. The target language id is used as the first generated token by passing <code>forced_bos_token_id</code> to <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate">generate()</a>.',Qo,Ke,mo,et,ho,E,tt,So,Kt,Fa=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartModel">MBartModel</a>. It is used to instantiate an MBART
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the MBART
<a href="https://huggingface.co/facebook/mbart-large-cc25" rel="nofollow">facebook/mbart-large-cc25</a> architecture.`,Lo,en,qa=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Eo,ge,uo,nt,fo,F,ot,Yo,tn,Ia="Construct an MBART tokenizer.",Ho,nn,Za=`Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.`,Do,on,Wa="The tokenization method is <code>&lt;tokens&gt; &lt;eos&gt; &lt;language code&gt;</code> for source language documents, and `&lt;language code&gt;",Ao,_e,Oo,D,st,Po,sn,Ra=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An MBART sequence has the following format, where <code>X</code> represents the sequence:`,Ko,an,Va="<li><code>input_ids</code> (for encoder) <code>X [eos, src_lang_code]</code></li> <li><code>decoder_input_ids</code>: (for decoder) <code>X [eos, tgt_lang_code]</code></li>",es,rn,Xa=`BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.`,go,at,_o,x,rt,ts,ln,Na=`Construct a “fast” MBART tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models" rel="nofollow">BPE</a>.`,ns,dn,Ga=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,os,cn,Qa="The tokenization method is <code>&lt;tokens&gt; &lt;eos&gt; &lt;language code&gt;</code> for source language documents, and `&lt;language code&gt;",ss,be,as,G,it,rs,pn,Sa=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. The special tokens depend on calling set_lang.`,is,mn,La="An MBART sequence has the following format, where <code>X</code> represents the sequence:",ls,hn,Ea="<li><code>input_ids</code> (for encoder) <code>X [eos, src_lang_code]</code></li> <li><code>decoder_input_ids</code>: (for decoder) <code>X [eos, tgt_lang_code]</code></li>",ds,un,Ya=`BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.`,cs,ye,lt,ps,fn,Ha=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. mBART does not
make use of token type ids, therefore a list of zeros is returned.`,ms,Me,dt,hs,gn,Da="Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code].",us,ke,ct,fs,_n,Aa="Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code].",bo,pt,yo,U,mt,gs,bn,Oa='Construct a MBart50 tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',_s,yn,Pa=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,bs,Te,ys,A,ht,Ms,Mn,Ka=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An MBART-50 sequence has the following format, where <code>X</code> represents the sequence:`,ks,kn,er="<li><code>input_ids</code> (for encoder) <code>[src_lang_code] X [eos]</code></li> <li><code>labels</code>: (for decoder) <code>[tgt_lang_code] X [eos]</code></li>",Ts,Tn,tr=`BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.`,vs,ve,ut,ws,vn,nr="Converts a sequence of tokens (string) in a single string.",$s,we,ft,Js,wn,or=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,xs,$e,gt,Us,$n,sr="Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos].",Bs,Je,_t,Cs,Jn,ar="Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos].",Mo,bt,ko,z,yt,zs,xn,rr=`Construct a “fast” MBART tokenizer for mBART-50 (backed by HuggingFace’s <em>tokenizers</em> library). Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models" rel="nofollow">BPE</a>.`,js,Un,ir=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Fs,xe,qs,Q,Mt,Is,Bn,lr=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. The special tokens depend on calling set_lang.`,Zs,Cn,dr="An MBART-50 sequence has the following format, where <code>X</code> represents the sequence:",Ws,zn,cr="<li><code>input_ids</code> (for encoder) <code>[src_lang_code] X [eos]</code></li> <li><code>labels</code>: (for decoder) <code>[tgt_lang_code] X [eos]</code></li>",Rs,jn,pr=`BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.`,Vs,Ue,kt,Xs,Fn,mr="Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos].",Ns,Be,Tt,Gs,qn,hr="Reset the special tokens to the target language setting. prefix=[src_lang_code] and suffix=[eos].",To,vt,vo,I,wt,Qs,In,ur="The bare Mbart Model outputting raw hidden-states without any specific head on top.",Ss,Zn,fr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ls,Wn,gr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Es,ae,$t,Ys,Rn,_r='The <a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartModel">MBartModel</a> forward method, overrides the <code>__call__</code> special method.',Hs,Ce,wo,Jt,$o,Z,xt,Ds,Vn,br="The MBART Model with a language modeling head. Can be used for summarization, after fine-tuning the pretrained models.",As,Xn,yr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Os,Nn,Mr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ps,S,Ut,Ks,Gn,kr='The <a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForConditionalGeneration">MBartForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',ea,ze,ta,je,na,Fe,Jo,Bt,xo,W,Ct,oa,Qn,Tr=`The Mbart transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,sa,Sn,vr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,aa,Ln,wr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ra,O,zt,ia,En,$r='The <a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForQuestionAnswering">MBartForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',la,qe,da,Ie,Uo,jt,Bo,R,Ft,ca,Yn,Jr=`MBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.`,pa,Hn,xr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ma,Dn,Ur=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ha,L,qt,ua,An,Br='The <a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForSequenceClassification">MBartForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',fa,Ze,ga,We,_a,Re,Co,It,zo,he,Zt,ba,P,Wt,ya,On,Cr='The <a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForCausalLM">MBartForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ma,Ve,ka,Xe,jo,Rt,Fo,eo,qo;return Qe=new K({props:{title:"mBART",local:"mbart",headingTag:"h1"}}),ue=new Kn({props:{warning:!1,$$slots:{default:[Gr]},$$scope:{ctx:v}}}),fe=new Nr({props:{id:"usage",options:["Pipeline","AutoModel"],$$slots:{default:[Lr]},$$scope:{ctx:v}}}),De=new K({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Oe=new X({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm1iYXJ0LWxhcmdlLWVuLXJvJTIyJTJDJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwTUJhcnRUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWJhcnQtbGFyZ2UtZW4tcm8lMjIlMkMlMjBzcmNfbGFuZyUzRCUyMmVuX1hYJTIyKSUwQSUwQWFydGljbGUlMjAlM0QlMjAlMjJVTiUyMENoaWVmJTIwU2F5cyUyMFRoZXJlJTIwSXMlMjBObyUyME1pbGl0YXJ5JTIwU29sdXRpb24lMjBpbiUyMFN5cmlhJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKGFydGljbGUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXRyYW5zbGF0ZWRfdG9rZW5zJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBkZWNvZGVyX3N0YXJ0X3Rva2VuX2lkJTNEdG9rZW5pemVyLmxhbmdfY29kZV90b19pZCU1QiUyMnJvX1JPJTIyJTVEKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUodHJhbnNsYXRlZF90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklNUIwJTVE",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-en-ro&quot;</span>, dtype=torch.bfloat16, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>)
tokenizer = MBartTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-en-ro&quot;</span>, src_lang=<span class="hljs-string">&quot;en_XX&quot;</span>)

article = <span class="hljs-string">&quot;UN Chief Says There Is No Military Solution in Syria&quot;</span>
inputs = tokenizer(article, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id[<span class="hljs-string">&quot;ro_RO&quot;</span>])
tokenizer.batch_decode(translated_tokens, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]`,wrap:!1}}),Ke=new X({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm1iYXJ0LWxhcmdlLTUwLW1hbnktdG8tbWFueS1tbXQlMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMkMlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBNQmFydFRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYmFydC1sYXJnZS01MC1tYW55LXRvLW1hbnktbW10JTIyKSUwQSUwQWFydGljbGVfYXIlMjAlM0QlMjAlMjIlRDglQTclRDklODQlRDglQTMlRDklODUlRDklOEElRDklODYlMjAlRDglQTclRDklODQlRDglQjklRDglQTclRDklODUlMjAlRDklODQlRDklODQlRDglQTMlRDklODUlRDklODUlMjAlRDglQTclRDklODQlRDklODUlRDglQUElRDglQUQlRDglQUYlRDglQTklMjAlRDklOEElRDklODIlRDklODglRDklODQlMjAlRDglQTUlRDklODYlRDklODclMjAlRDklODQlRDglQTclMjAlRDklOEElRDklODglRDglQUMlRDglQUYlMjAlRDglQUQlRDklODQlMjAlRDglQjklRDglQjMlRDklODMlRDglQjElRDklOEElMjAlRDklODElRDklOEElMjAlRDglQjMlRDklODglRDglQjElRDklOEElRDglQTcuJTIyJTBBdG9rZW5pemVyLnNyY19sYW5nJTIwJTNEJTIwJTIyYXJfQVIlMjIlMEElMEFlbmNvZGVkX2FyJTIwJTNEJTIwdG9rZW5pemVyKGFydGljbGVfYXIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWdlbmVyYXRlZF90b2tlbnMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmVuY29kZWRfYXIlMkMlMjBmb3JjZWRfYm9zX3Rva2VuX2lkJTNEdG9rZW5pemVyLmxhbmdfY29kZV90b19pZCU1QiUyMmVuX1hYJTIyJTVEKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-50-many-to-many-mmt&quot;</span>, dtype=torch.bfloat16, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>)
tokenizer = MBartTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/mbart-large-50-many-to-many-mmt&quot;</span>)

article_ar = <span class="hljs-string">&quot;الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا.&quot;</span>
tokenizer.src_lang = <span class="hljs-string">&quot;ar_AR&quot;</span>

encoded_ar = tokenizer(article_ar, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id[<span class="hljs-string">&quot;en_XX&quot;</span>])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),et=new K({props:{title:"MBartConfig",local:"transformers.MBartConfig",headingTag:"h2"}}),tt=new J({props:{name:"class transformers.MBartConfig",anchor:"transformers.MBartConfig",parameters:[{name:"vocab_size",val:" = 50265"},{name:"max_position_embeddings",val:" = 1024"},{name:"encoder_layers",val:" = 12"},{name:"encoder_ffn_dim",val:" = 4096"},{name:"encoder_attention_heads",val:" = 16"},{name:"decoder_layers",val:" = 12"},{name:"decoder_ffn_dim",val:" = 4096"},{name:"decoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 1024"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"classifier_dropout",val:" = 0.0"},{name:"scale_embedding",val:" = False"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"forced_eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MBartConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the MBART model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartModel">MBartModel</a> or <code>TFMBartModel</code>.`,name:"vocab_size"},{anchor:"transformers.MBartConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.MBartConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.MBartConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.MBartConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.MBartConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.MBartConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.MBartConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.MBartConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.MBartConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.MBartConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.MBartConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.MBartConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.MBartConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.MBartConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.MBartConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.MBartConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.MBartConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.MBartConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.MBartConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/configuration_mbart.py#L31"}}),ge=new se({props:{anchor:"transformers.MBartConfig.example",$$slots:{default:[Er]},$$scope:{ctx:v}}}),nt=new K({props:{title:"MBartTokenizer",local:"transformers.MBartTokenizer",headingTag:"h2"}}),ot=new J({props:{name:"class transformers.MBartTokenizer",anchor:"transformers.MBartTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"tokenizer_file",val:" = None"},{name:"src_lang",val:" = None"},{name:"tgt_lang",val:" = None"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"additional_special_tokens",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/tokenization_mbart.py#L38"}}),_e=new se({props:{anchor:"transformers.MBartTokenizer.example",$$slots:{default:[Yr]},$$scope:{ctx:v}}}),st=new J({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MBartTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MBartTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MBartTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/tokenization_mbart.py#L202",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),at=new K({props:{title:"MBartTokenizerFast",local:"transformers.MBartTokenizerFast",headingTag:"h2"}}),rt=new J({props:{name:"class transformers.MBartTokenizerFast",anchor:"transformers.MBartTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"src_lang",val:" = None"},{name:"tgt_lang",val:" = None"},{name:"additional_special_tokens",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/tokenization_mbart_fast.py#L42"}}),be=new se({props:{anchor:"transformers.MBartTokenizerFast.example",$$slots:{default:[Hr]},$$scope:{ctx:v}}}),it=new J({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MBartTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MBartTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MBartTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/tokenization_mbart_fast.py#L135",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),lt=new J({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.MBartTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MBartTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.MBartTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/tokenization_mbart_fast.py#L164",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),dt=new J({props:{name:"set_src_lang_special_tokens",anchor:"transformers.MBartTokenizerFast.set_src_lang_special_tokens",parameters:[{name:"src_lang",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/tokenization_mbart_fast.py#L219"}}),ct=new J({props:{name:"set_tgt_lang_special_tokens",anchor:"transformers.MBartTokenizerFast.set_tgt_lang_special_tokens",parameters:[{name:"lang",val:": str"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/tokenization_mbart_fast.py#L234"}}),pt=new K({props:{title:"MBart50Tokenizer",local:"transformers.MBart50Tokenizer",headingTag:"h2"}}),mt=new J({props:{name:"class transformers.MBart50Tokenizer",anchor:"transformers.MBart50Tokenizer",parameters:[{name:"vocab_file",val:""},{name:"src_lang",val:" = None"},{name:"tgt_lang",val:" = None"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MBart50Tokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.MBart50Tokenizer.src_lang",description:`<strong>src_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the source language.`,name:"src_lang"},{anchor:"transformers.MBart50Tokenizer.tgt_lang",description:`<strong>tgt_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the target language.`,name:"tgt_lang"},{anchor:"transformers.MBart50Tokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.MBart50Tokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MBart50Tokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MBart50Tokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MBart50Tokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MBart50Tokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MBart50Tokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50.py#L38"}}),Te=new se({props:{anchor:"transformers.MBart50Tokenizer.example",$$slots:{default:[Dr]},$$scope:{ctx:v}}}),ht=new J({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MBart50Tokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MBart50Tokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MBart50Tokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50.py#L289",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ut=new J({props:{name:"convert_tokens_to_string",anchor:"transformers.MBart50Tokenizer.convert_tokens_to_string",parameters:[{name:"tokens",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50.py#L223"}}),ft=new J({props:{name:"get_special_tokens_mask",anchor:"transformers.MBart50Tokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MBart50Tokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.MBart50Tokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.MBart50Tokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50.py#L259",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),gt=new J({props:{name:"set_src_lang_special_tokens",anchor:"transformers.MBart50Tokenizer.set_src_lang_special_tokens",parameters:[{name:"src_lang",val:": str"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50.py#L346"}}),_t=new J({props:{name:"set_tgt_lang_special_tokens",anchor:"transformers.MBart50Tokenizer.set_tgt_lang_special_tokens",parameters:[{name:"tgt_lang",val:": str"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50.py#L352"}}),bt=new K({props:{title:"MBart50TokenizerFast",local:"transformers.MBart50TokenizerFast",headingTag:"h2"}}),yt=new J({props:{name:"class transformers.MBart50TokenizerFast",anchor:"transformers.MBart50TokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"src_lang",val:" = None"},{name:"tgt_lang",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MBart50TokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.MBart50TokenizerFast.src_lang",description:`<strong>src_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the source language.`,name:"src_lang"},{anchor:"transformers.MBart50TokenizerFast.tgt_lang",description:`<strong>tgt_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the target language.`,name:"tgt_lang"},{anchor:"transformers.MBart50TokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.MBart50TokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MBart50TokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MBart50TokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MBart50TokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MBart50TokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50_fast.py#L41"}}),xe=new se({props:{anchor:"transformers.MBart50TokenizerFast.example",$$slots:{default:[Ar]},$$scope:{ctx:v}}}),Mt=new J({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MBart50TokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MBart50TokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MBart50TokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50_fast.py#L149",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),kt=new J({props:{name:"set_src_lang_special_tokens",anchor:"transformers.MBart50TokenizerFast.set_src_lang_special_tokens",parameters:[{name:"src_lang",val:": str"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50_fast.py#L196"}}),Tt=new J({props:{name:"set_tgt_lang_special_tokens",anchor:"transformers.MBart50TokenizerFast.set_tgt_lang_special_tokens",parameters:[{name:"tgt_lang",val:": str"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart50/tokenization_mbart50_fast.py#L211"}}),vt=new K({props:{title:"MBartModel",local:"transformers.MBartModel",headingTag:"h2"}}),wt=new J({props:{name:"class transformers.MBartModel",anchor:"transformers.MBartModel",parameters:[{name:"config",val:": MBartConfig"}],parametersDescription:[{anchor:"transformers.MBartModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig">MBartConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1157"}}),$t=new J({props:{name:"forward",anchor:"transformers.MBartModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MBartModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MBartModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MBartModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>MBart uses a specific language id token as the starting token for <code>decoder_input_ids</code> generation that
varies according to source and target language, <em>e.g.</em> 25004 for <em>en_XX</em>, and 25003 for <em>de_DE</em>. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.MBartModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MBartModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MBartModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MBartModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MBartModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MBartModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MBartModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MBartModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MBartModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MBartModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MBartModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MBartModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MBartModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1189",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig"
>MBartConfig</a>) and inputs.</p>
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
`}}),Ce=new Kn({props:{$$slots:{default:[Or]},$$scope:{ctx:v}}}),Jt=new K({props:{title:"MBartForConditionalGeneration",local:"transformers.MBartForConditionalGeneration",headingTag:"h2"}}),xt=new J({props:{name:"class transformers.MBartForConditionalGeneration",anchor:"transformers.MBartForConditionalGeneration",parameters:[{name:"config",val:": MBartConfig"}],parametersDescription:[{anchor:"transformers.MBartForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig">MBartConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1303"}}),Ut=new J({props:{name:"forward",anchor:"transformers.MBartForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MBartForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MBartForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MBartForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>MBart uses a specific language id token as the starting token for <code>decoder_input_ids</code> generation that
varies according to source and target language, <em>e.g.</em> 25004 for <em>en_XX</em>, and 25003 for <em>de_DE</em>. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.MBartForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MBartForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MBartForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MBartForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MBartForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MBartForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MBartForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MBartForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MBartForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MBartForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MBartForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MBartForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MBartForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MBartForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1339",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig"
>MBartConfig</a>) and inputs.</p>
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
`}}),ze=new Kn({props:{$$slots:{default:[Pr]},$$scope:{ctx:v}}}),je=new se({props:{anchor:"transformers.MBartForConditionalGeneration.forward.example",$$slots:{default:[Kr]},$$scope:{ctx:v}}}),Fe=new se({props:{anchor:"transformers.MBartForConditionalGeneration.forward.example-2",$$slots:{default:[ei]},$$scope:{ctx:v}}}),Bt=new K({props:{title:"MBartForQuestionAnswering",local:"transformers.MBartForQuestionAnswering",headingTag:"h2"}}),Ct=new J({props:{name:"class transformers.MBartForQuestionAnswering",anchor:"transformers.MBartForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MBartForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartForQuestionAnswering">MBartForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1637"}}),zt=new J({props:{name:"forward",anchor:"transformers.MBartForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.MBartForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MBartForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MBartForQuestionAnswering.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Bart uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.MBartForQuestionAnswering.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read <code>modeling_bart._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.MBartForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MBartForQuestionAnswering.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MBartForQuestionAnswering.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MBartForQuestionAnswering.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MBartForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.MBartForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.MBartForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MBartForQuestionAnswering.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MBartForQuestionAnswering.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MBartForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MBartForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MBartForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MBartForQuestionAnswering.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1652",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput"
>transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig"
>MBartConfig</a>) and inputs.</p>
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
`}}),qe=new Kn({props:{$$slots:{default:[ti]},$$scope:{ctx:v}}}),Ie=new se({props:{anchor:"transformers.MBartForQuestionAnswering.forward.example",$$slots:{default:[ni]},$$scope:{ctx:v}}}),jt=new K({props:{title:"MBartForSequenceClassification",local:"transformers.MBartForSequenceClassification",headingTag:"h2"}}),Ft=new J({props:{name:"class transformers.MBartForSequenceClassification",anchor:"transformers.MBartForSequenceClassification",parameters:[{name:"config",val:": MBartConfig"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MBartForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig">MBartConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1490"}}),qt=new J({props:{name:"forward",anchor:"transformers.MBartForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.MBartForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MBartForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MBartForSequenceClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Bart uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.MBartForSequenceClassification.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read <code>modeling_bart._prepare_decoder_attention_mask</code>
and modify to your needs. See diagram 1 in <a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more
information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.MBartForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MBartForSequenceClassification.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.MBartForSequenceClassification.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MBartForSequenceClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MBartForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MBartForSequenceClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MBartForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.MBartForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MBartForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MBartForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MBartForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MBartForSequenceClassification.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1506",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig"
>MBartConfig</a>) and inputs.</p>
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
`}}),Ze=new Kn({props:{$$slots:{default:[oi]},$$scope:{ctx:v}}}),We=new se({props:{anchor:"transformers.MBartForSequenceClassification.forward.example",$$slots:{default:[si]},$$scope:{ctx:v}}}),Re=new se({props:{anchor:"transformers.MBartForSequenceClassification.forward.example-2",$$slots:{default:[ai]},$$scope:{ctx:v}}}),It=new K({props:{title:"MBartForCausalLM",local:"transformers.MBartForCausalLM",headingTag:"h2"}}),Zt=new J({props:{name:"class transformers.MBartForCausalLM",anchor:"transformers.MBartForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1786"}}),Wt=new J({props:{name:"forward",anchor:"transformers.MBartForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.MBartForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MBartForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MBartForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MBartForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.MBartForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MBartForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MBartForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MBartForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MBartForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MBartForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MBartForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MBartForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MBartForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MBartForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mbart/modeling_mbart.py#L1812",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mbart#transformers.MBartConfig"
>MBartConfig</a>) and inputs.</p>
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
`}}),Ve=new Kn({props:{$$slots:{default:[ri]},$$scope:{ctx:v}}}),Xe=new se({props:{anchor:"transformers.MBartForCausalLM.forward.example",$$slots:{default:[ii]},$$scope:{ctx:v}}}),Rt=new Xr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mbart.md"}}),{c(){n=l("meta"),k=s(),r=l("p"),c=s(),h=l("p"),h.innerHTML=t,T=s(),ee=l("div"),ee.innerHTML=Ta,to=s(),f(Qe.$$.fragment),no=s(),Se=l("p"),Se.innerHTML=va,oo=s(),Le=l("p"),Le.innerHTML=wa,so=s(),Ee=l("p"),Ee.innerHTML=$a,ao=s(),f(ue.$$.fragment),ro=s(),Ye=l("blockquote"),Ye.innerHTML=Ja,io=s(),He=l("p"),He.innerHTML=xa,lo=s(),f(fe.$$.fragment),co=s(),f(De.$$.fragment),po=s(),N=l("ul"),Ht=l("li"),Ht.innerHTML=Ua,Ro=s(),Dt=l("li"),Dt.innerHTML=Ba,Vo=s(),Ae=l("li"),At=l("p"),At.innerHTML=Ca,Xo=s(),f(Oe.$$.fragment),No=s(),Ot=l("li"),Ot.innerHTML=za,Go=s(),Pe=l("li"),Pt=l("p"),Pt.innerHTML=ja,Qo=s(),f(Ke.$$.fragment),mo=s(),f(et.$$.fragment),ho=s(),E=l("div"),f(tt.$$.fragment),So=s(),Kt=l("p"),Kt.innerHTML=Fa,Lo=s(),en=l("p"),en.innerHTML=qa,Eo=s(),f(ge.$$.fragment),uo=s(),f(nt.$$.fragment),fo=s(),F=l("div"),f(ot.$$.fragment),Yo=s(),tn=l("p"),tn.textContent=Ia,Ho=s(),nn=l("p"),nn.innerHTML=Za,Do=s(),on=l("p"),on.innerHTML=Wa,Ao=zr(`
<tokens> <eos>\` for target language documents.
`),f(_e.$$.fragment),Oo=s(),D=l("div"),f(st.$$.fragment),Po=s(),sn=l("p"),sn.innerHTML=Ra,Ko=s(),an=l("ul"),an.innerHTML=Va,es=s(),rn=l("p"),rn.textContent=Xa,go=s(),f(at.$$.fragment),_o=s(),x=l("div"),f(rt.$$.fragment),ts=s(),ln=l("p"),ln.innerHTML=Na,ns=s(),dn=l("p"),dn.innerHTML=Ga,os=s(),cn=l("p"),cn.innerHTML=Qa,ss=zr(`
<tokens> <eos>\` for target language documents.
`),f(be.$$.fragment),as=s(),G=l("div"),f(it.$$.fragment),rs=s(),pn=l("p"),pn.textContent=Sa,is=s(),mn=l("p"),mn.innerHTML=La,ls=s(),hn=l("ul"),hn.innerHTML=Ea,ds=s(),un=l("p"),un.textContent=Ya,cs=s(),ye=l("div"),f(lt.$$.fragment),ps=s(),fn=l("p"),fn.textContent=Ha,ms=s(),Me=l("div"),f(dt.$$.fragment),hs=s(),gn=l("p"),gn.textContent=Da,us=s(),ke=l("div"),f(ct.$$.fragment),fs=s(),_n=l("p"),_n.textContent=Aa,bo=s(),f(pt.$$.fragment),yo=s(),U=l("div"),f(mt.$$.fragment),gs=s(),bn=l("p"),bn.innerHTML=Oa,_s=s(),yn=l("p"),yn.innerHTML=Pa,bs=s(),f(Te.$$.fragment),ys=s(),A=l("div"),f(ht.$$.fragment),Ms=s(),Mn=l("p"),Mn.innerHTML=Ka,ks=s(),kn=l("ul"),kn.innerHTML=er,Ts=s(),Tn=l("p"),Tn.textContent=tr,vs=s(),ve=l("div"),f(ut.$$.fragment),ws=s(),vn=l("p"),vn.textContent=nr,$s=s(),we=l("div"),f(ft.$$.fragment),Js=s(),wn=l("p"),wn.innerHTML=or,xs=s(),$e=l("div"),f(gt.$$.fragment),Us=s(),$n=l("p"),$n.textContent=sr,Bs=s(),Je=l("div"),f(_t.$$.fragment),Cs=s(),Jn=l("p"),Jn.textContent=ar,Mo=s(),f(bt.$$.fragment),ko=s(),z=l("div"),f(yt.$$.fragment),zs=s(),xn=l("p"),xn.innerHTML=rr,js=s(),Un=l("p"),Un.innerHTML=ir,Fs=s(),f(xe.$$.fragment),qs=s(),Q=l("div"),f(Mt.$$.fragment),Is=s(),Bn=l("p"),Bn.textContent=lr,Zs=s(),Cn=l("p"),Cn.innerHTML=dr,Ws=s(),zn=l("ul"),zn.innerHTML=cr,Rs=s(),jn=l("p"),jn.textContent=pr,Vs=s(),Ue=l("div"),f(kt.$$.fragment),Xs=s(),Fn=l("p"),Fn.textContent=mr,Ns=s(),Be=l("div"),f(Tt.$$.fragment),Gs=s(),qn=l("p"),qn.textContent=hr,To=s(),f(vt.$$.fragment),vo=s(),I=l("div"),f(wt.$$.fragment),Qs=s(),In=l("p"),In.textContent=ur,Ss=s(),Zn=l("p"),Zn.innerHTML=fr,Ls=s(),Wn=l("p"),Wn.innerHTML=gr,Es=s(),ae=l("div"),f($t.$$.fragment),Ys=s(),Rn=l("p"),Rn.innerHTML=_r,Hs=s(),f(Ce.$$.fragment),wo=s(),f(Jt.$$.fragment),$o=s(),Z=l("div"),f(xt.$$.fragment),Ds=s(),Vn=l("p"),Vn.textContent=br,As=s(),Xn=l("p"),Xn.innerHTML=yr,Os=s(),Nn=l("p"),Nn.innerHTML=Mr,Ps=s(),S=l("div"),f(Ut.$$.fragment),Ks=s(),Gn=l("p"),Gn.innerHTML=kr,ea=s(),f(ze.$$.fragment),ta=s(),f(je.$$.fragment),na=s(),f(Fe.$$.fragment),Jo=s(),f(Bt.$$.fragment),xo=s(),W=l("div"),f(Ct.$$.fragment),oa=s(),Qn=l("p"),Qn.innerHTML=Tr,sa=s(),Sn=l("p"),Sn.innerHTML=vr,aa=s(),Ln=l("p"),Ln.innerHTML=wr,ra=s(),O=l("div"),f(zt.$$.fragment),ia=s(),En=l("p"),En.innerHTML=$r,la=s(),f(qe.$$.fragment),da=s(),f(Ie.$$.fragment),Uo=s(),f(jt.$$.fragment),Bo=s(),R=l("div"),f(Ft.$$.fragment),ca=s(),Yn=l("p"),Yn.textContent=Jr,pa=s(),Hn=l("p"),Hn.innerHTML=xr,ma=s(),Dn=l("p"),Dn.innerHTML=Ur,ha=s(),L=l("div"),f(qt.$$.fragment),ua=s(),An=l("p"),An.innerHTML=Br,fa=s(),f(Ze.$$.fragment),ga=s(),f(We.$$.fragment),_a=s(),f(Re.$$.fragment),Co=s(),f(It.$$.fragment),zo=s(),he=l("div"),f(Zt.$$.fragment),ba=s(),P=l("div"),f(Wt.$$.fragment),ya=s(),On=l("p"),On.innerHTML=Cr,Ma=s(),f(Ve.$$.fragment),ka=s(),f(Xe.$$.fragment),jo=s(),f(Rt.$$.fragment),Fo=s(),eo=l("p"),this.h()},l(e){const p=Rr("svelte-u9bgzb",document.head);n=d(p,"META",{name:!0,content:!0}),p.forEach(i),k=a(e),r=d(e,"P",{}),w(r).forEach(i),c=a(e),h=d(e,"P",{"data-svelte-h":!0}),u(h)!=="svelte-9do264"&&(h.innerHTML=t),T=a(e),ee=d(e,"DIV",{style:!0,"data-svelte-h":!0}),u(ee)!=="svelte-1lhmk4n"&&(ee.innerHTML=Ta),to=a(e),g(Qe.$$.fragment,e),no=a(e),Se=d(e,"P",{"data-svelte-h":!0}),u(Se)!=="svelte-1vysjab"&&(Se.innerHTML=va),oo=a(e),Le=d(e,"P",{"data-svelte-h":!0}),u(Le)!=="svelte-wwro76"&&(Le.innerHTML=wa),so=a(e),Ee=d(e,"P",{"data-svelte-h":!0}),u(Ee)!=="svelte-1xhm8em"&&(Ee.innerHTML=$a),ao=a(e),g(ue.$$.fragment,e),ro=a(e),Ye=d(e,"BLOCKQUOTE",{"data-svelte-h":!0}),u(Ye)!=="svelte-1fwzni2"&&(Ye.innerHTML=Ja),io=a(e),He=d(e,"P",{"data-svelte-h":!0}),u(He)!=="svelte-1s7eaah"&&(He.innerHTML=xa),lo=a(e),g(fe.$$.fragment,e),co=a(e),g(De.$$.fragment,e),po=a(e),N=d(e,"UL",{});var Y=w(N);Ht=d(Y,"LI",{"data-svelte-h":!0}),u(Ht)!=="svelte-788e1g"&&(Ht.innerHTML=Ua),Ro=a(Y),Dt=d(Y,"LI",{"data-svelte-h":!0}),u(Dt)!=="svelte-o85afb"&&(Dt.innerHTML=Ba),Vo=a(Y),Ae=d(Y,"LI",{});var Vt=w(Ae);At=d(Vt,"P",{"data-svelte-h":!0}),u(At)!=="svelte-1lrkeju"&&(At.innerHTML=Ca),Xo=a(Vt),g(Oe.$$.fragment,Vt),Vt.forEach(i),No=a(Y),Ot=d(Y,"LI",{"data-svelte-h":!0}),u(Ot)!=="svelte-1kaf5qs"&&(Ot.innerHTML=za),Go=a(Y),Pe=d(Y,"LI",{});var Xt=w(Pe);Pt=d(Xt,"P",{"data-svelte-h":!0}),u(Pt)!=="svelte-130d63b"&&(Pt.innerHTML=ja),Qo=a(Xt),g(Ke.$$.fragment,Xt),Xt.forEach(i),Y.forEach(i),mo=a(e),g(et.$$.fragment,e),ho=a(e),E=d(e,"DIV",{class:!0});var te=w(E);g(tt.$$.fragment,te),So=a(te),Kt=d(te,"P",{"data-svelte-h":!0}),u(Kt)!=="svelte-l8s8h7"&&(Kt.innerHTML=Fa),Lo=a(te),en=d(te,"P",{"data-svelte-h":!0}),u(en)!=="svelte-1ek1ss9"&&(en.innerHTML=qa),Eo=a(te),g(ge.$$.fragment,te),te.forEach(i),uo=a(e),g(nt.$$.fragment,e),fo=a(e),F=d(e,"DIV",{class:!0});var V=w(F);g(ot.$$.fragment,V),Yo=a(V),tn=d(V,"P",{"data-svelte-h":!0}),u(tn)!=="svelte-bx5v1x"&&(tn.textContent=Ia),Ho=a(V),nn=d(V,"P",{"data-svelte-h":!0}),u(nn)!=="svelte-19vr0qz"&&(nn.innerHTML=Za),Do=a(V),on=d(V,"P",{"data-svelte-h":!0}),u(on)!=="svelte-1i8rh37"&&(on.innerHTML=Wa),Ao=jr(V,`
<tokens> <eos>\` for target language documents.
`),g(_e.$$.fragment,V),Oo=a(V),D=d(V,"DIV",{class:!0});var ne=w(D);g(st.$$.fragment,ne),Po=a(ne),sn=d(ne,"P",{"data-svelte-h":!0}),u(sn)!=="svelte-1homupa"&&(sn.innerHTML=Ra),Ko=a(ne),an=d(ne,"UL",{"data-svelte-h":!0}),u(an)!=="svelte-mlrsks"&&(an.innerHTML=Va),es=a(ne),rn=d(ne,"P",{"data-svelte-h":!0}),u(rn)!=="svelte-46aam0"&&(rn.textContent=Xa),ne.forEach(i),V.forEach(i),go=a(e),g(at.$$.fragment,e),_o=a(e),x=d(e,"DIV",{class:!0});var B=w(x);g(rt.$$.fragment,B),ts=a(B),ln=d(B,"P",{"data-svelte-h":!0}),u(ln)!=="svelte-15e1szj"&&(ln.innerHTML=Na),ns=a(B),dn=d(B,"P",{"data-svelte-h":!0}),u(dn)!=="svelte-gxzj9w"&&(dn.innerHTML=Ga),os=a(B),cn=d(B,"P",{"data-svelte-h":!0}),u(cn)!=="svelte-1i8rh37"&&(cn.innerHTML=Qa),ss=jr(B,`
<tokens> <eos>\` for target language documents.
`),g(be.$$.fragment,B),as=a(B),G=d(B,"DIV",{class:!0});var H=w(G);g(it.$$.fragment,H),rs=a(H),pn=d(H,"P",{"data-svelte-h":!0}),u(pn)!=="svelte-1vll0v2"&&(pn.textContent=Sa),is=a(H),mn=d(H,"P",{"data-svelte-h":!0}),u(mn)!=="svelte-93oclo"&&(mn.innerHTML=La),ls=a(H),hn=d(H,"UL",{"data-svelte-h":!0}),u(hn)!=="svelte-mlrsks"&&(hn.innerHTML=Ea),ds=a(H),un=d(H,"P",{"data-svelte-h":!0}),u(un)!=="svelte-46aam0"&&(un.textContent=Ya),H.forEach(i),cs=a(B),ye=d(B,"DIV",{class:!0});var Nt=w(ye);g(lt.$$.fragment,Nt),ps=a(Nt),fn=d(Nt,"P",{"data-svelte-h":!0}),u(fn)!=="svelte-1b92gql"&&(fn.textContent=Ha),Nt.forEach(i),ms=a(B),Me=d(B,"DIV",{class:!0});var Gt=w(Me);g(dt.$$.fragment,Gt),hs=a(Gt),gn=d(Gt,"P",{"data-svelte-h":!0}),u(gn)!=="svelte-1q9gpxy"&&(gn.textContent=Da),Gt.forEach(i),us=a(B),ke=d(B,"DIV",{class:!0});var Qt=w(ke);g(ct.$$.fragment,Qt),fs=a(Qt),_n=d(Qt,"P",{"data-svelte-h":!0}),u(_n)!=="svelte-e4r809"&&(_n.textContent=Aa),Qt.forEach(i),B.forEach(i),bo=a(e),g(pt.$$.fragment,e),yo=a(e),U=d(e,"DIV",{class:!0});var C=w(U);g(mt.$$.fragment,C),gs=a(C),bn=d(C,"P",{"data-svelte-h":!0}),u(bn)!=="svelte-n0bh6j"&&(bn.innerHTML=Oa),_s=a(C),yn=d(C,"P",{"data-svelte-h":!0}),u(yn)!=="svelte-ntrhio"&&(yn.innerHTML=Pa),bs=a(C),g(Te.$$.fragment,C),ys=a(C),A=d(C,"DIV",{class:!0});var oe=w(A);g(ht.$$.fragment,oe),Ms=a(oe),Mn=d(oe,"P",{"data-svelte-h":!0}),u(Mn)!=="svelte-1xfoywu"&&(Mn.innerHTML=Ka),ks=a(oe),kn=d(oe,"UL",{"data-svelte-h":!0}),u(kn)!=="svelte-yv4lcp"&&(kn.innerHTML=er),Ts=a(oe),Tn=d(oe,"P",{"data-svelte-h":!0}),u(Tn)!=="svelte-46aam0"&&(Tn.textContent=tr),oe.forEach(i),vs=a(C),ve=d(C,"DIV",{class:!0});var St=w(ve);g(ut.$$.fragment,St),ws=a(St),vn=d(St,"P",{"data-svelte-h":!0}),u(vn)!=="svelte-b3k2yi"&&(vn.textContent=nr),St.forEach(i),$s=a(C),we=d(C,"DIV",{class:!0});var Lt=w(we);g(ft.$$.fragment,Lt),Js=a(Lt),wn=d(Lt,"P",{"data-svelte-h":!0}),u(wn)!=="svelte-1f4f5kp"&&(wn.innerHTML=or),Lt.forEach(i),xs=a(C),$e=d(C,"DIV",{class:!0});var Et=w($e);g(gt.$$.fragment,Et),Us=a(Et),$n=d(Et,"P",{"data-svelte-h":!0}),u($n)!=="svelte-f2psqy"&&($n.textContent=sr),Et.forEach(i),Bs=a(C),Je=d(C,"DIV",{class:!0});var Yt=w(Je);g(_t.$$.fragment,Yt),Cs=a(Yt),Jn=d(Yt,"P",{"data-svelte-h":!0}),u(Jn)!=="svelte-cp14dz"&&(Jn.textContent=ar),Yt.forEach(i),C.forEach(i),Mo=a(e),g(bt.$$.fragment,e),ko=a(e),z=d(e,"DIV",{class:!0});var q=w(z);g(yt.$$.fragment,q),zs=a(q),xn=d(q,"P",{"data-svelte-h":!0}),u(xn)!=="svelte-1khs09e"&&(xn.innerHTML=rr),js=a(q),Un=d(q,"P",{"data-svelte-h":!0}),u(Un)!=="svelte-gxzj9w"&&(Un.innerHTML=ir),Fs=a(q),g(xe.$$.fragment,q),qs=a(q),Q=d(q,"DIV",{class:!0});var re=w(Q);g(Mt.$$.fragment,re),Is=a(re),Bn=d(re,"P",{"data-svelte-h":!0}),u(Bn)!=="svelte-1vll0v2"&&(Bn.textContent=lr),Zs=a(re),Cn=d(re,"P",{"data-svelte-h":!0}),u(Cn)!=="svelte-23ncsc"&&(Cn.innerHTML=dr),Ws=a(re),zn=d(re,"UL",{"data-svelte-h":!0}),u(zn)!=="svelte-yv4lcp"&&(zn.innerHTML=cr),Rs=a(re),jn=d(re,"P",{"data-svelte-h":!0}),u(jn)!=="svelte-46aam0"&&(jn.textContent=pr),re.forEach(i),Vs=a(q),Ue=d(q,"DIV",{class:!0});var Io=w(Ue);g(kt.$$.fragment,Io),Xs=a(Io),Fn=d(Io,"P",{"data-svelte-h":!0}),u(Fn)!=="svelte-f2psqy"&&(Fn.textContent=mr),Io.forEach(i),Ns=a(q),Be=d(q,"DIV",{class:!0});var Zo=w(Be);g(Tt.$$.fragment,Zo),Gs=a(Zo),qn=d(Zo,"P",{"data-svelte-h":!0}),u(qn)!=="svelte-vhljfs"&&(qn.textContent=hr),Zo.forEach(i),q.forEach(i),To=a(e),g(vt.$$.fragment,e),vo=a(e),I=d(e,"DIV",{class:!0});var ie=w(I);g(wt.$$.fragment,ie),Qs=a(ie),In=d(ie,"P",{"data-svelte-h":!0}),u(In)!=="svelte-1b6cory"&&(In.textContent=ur),Ss=a(ie),Zn=d(ie,"P",{"data-svelte-h":!0}),u(Zn)!=="svelte-q52n56"&&(Zn.innerHTML=fr),Ls=a(ie),Wn=d(ie,"P",{"data-svelte-h":!0}),u(Wn)!=="svelte-hswkmf"&&(Wn.innerHTML=gr),Es=a(ie),ae=d(ie,"DIV",{class:!0});var Pn=w(ae);g($t.$$.fragment,Pn),Ys=a(Pn),Rn=d(Pn,"P",{"data-svelte-h":!0}),u(Rn)!=="svelte-uyvhgd"&&(Rn.innerHTML=_r),Hs=a(Pn),g(Ce.$$.fragment,Pn),Pn.forEach(i),ie.forEach(i),wo=a(e),g(Jt.$$.fragment,e),$o=a(e),Z=d(e,"DIV",{class:!0});var le=w(Z);g(xt.$$.fragment,le),Ds=a(le),Vn=d(le,"P",{"data-svelte-h":!0}),u(Vn)!=="svelte-1d2hjhr"&&(Vn.textContent=br),As=a(le),Xn=d(le,"P",{"data-svelte-h":!0}),u(Xn)!=="svelte-q52n56"&&(Xn.innerHTML=yr),Os=a(le),Nn=d(le,"P",{"data-svelte-h":!0}),u(Nn)!=="svelte-hswkmf"&&(Nn.innerHTML=Mr),Ps=a(le),S=d(le,"DIV",{class:!0});var de=w(S);g(Ut.$$.fragment,de),Ks=a(de),Gn=d(de,"P",{"data-svelte-h":!0}),u(Gn)!=="svelte-k0ma8n"&&(Gn.innerHTML=kr),ea=a(de),g(ze.$$.fragment,de),ta=a(de),g(je.$$.fragment,de),na=a(de),g(Fe.$$.fragment,de),de.forEach(i),le.forEach(i),Jo=a(e),g(Bt.$$.fragment,e),xo=a(e),W=d(e,"DIV",{class:!0});var ce=w(W);g(Ct.$$.fragment,ce),oa=a(ce),Qn=d(ce,"P",{"data-svelte-h":!0}),u(Qn)!=="svelte-uij1zd"&&(Qn.innerHTML=Tr),sa=a(ce),Sn=d(ce,"P",{"data-svelte-h":!0}),u(Sn)!=="svelte-q52n56"&&(Sn.innerHTML=vr),aa=a(ce),Ln=d(ce,"P",{"data-svelte-h":!0}),u(Ln)!=="svelte-hswkmf"&&(Ln.innerHTML=wr),ra=a(ce),O=d(ce,"DIV",{class:!0});var Ne=w(O);g(zt.$$.fragment,Ne),ia=a(Ne),En=d(Ne,"P",{"data-svelte-h":!0}),u(En)!=="svelte-146pkw7"&&(En.innerHTML=$r),la=a(Ne),g(qe.$$.fragment,Ne),da=a(Ne),g(Ie.$$.fragment,Ne),Ne.forEach(i),ce.forEach(i),Uo=a(e),g(jt.$$.fragment,e),Bo=a(e),R=d(e,"DIV",{class:!0});var pe=w(R);g(Ft.$$.fragment,pe),ca=a(pe),Yn=d(pe,"P",{"data-svelte-h":!0}),u(Yn)!=="svelte-1a55pns"&&(Yn.textContent=Jr),pa=a(pe),Hn=d(pe,"P",{"data-svelte-h":!0}),u(Hn)!=="svelte-q52n56"&&(Hn.innerHTML=xr),ma=a(pe),Dn=d(pe,"P",{"data-svelte-h":!0}),u(Dn)!=="svelte-hswkmf"&&(Dn.innerHTML=Ur),ha=a(pe),L=d(pe,"DIV",{class:!0});var me=w(L);g(qt.$$.fragment,me),ua=a(me),An=d(me,"P",{"data-svelte-h":!0}),u(An)!=="svelte-1mdpu9r"&&(An.innerHTML=Br),fa=a(me),g(Ze.$$.fragment,me),ga=a(me),g(We.$$.fragment,me),_a=a(me),g(Re.$$.fragment,me),me.forEach(i),pe.forEach(i),Co=a(e),g(It.$$.fragment,e),zo=a(e),he=d(e,"DIV",{class:!0});var Wo=w(he);g(Zt.$$.fragment,Wo),ba=a(Wo),P=d(Wo,"DIV",{class:!0});var Ge=w(P);g(Wt.$$.fragment,Ge),ya=a(Ge),On=d(Ge,"P",{"data-svelte-h":!0}),u(On)!=="svelte-j8gezt"&&(On.innerHTML=Cr),Ma=a(Ge),g(Ve.$$.fragment,Ge),ka=a(Ge),g(Xe.$$.fragment,Ge),Ge.forEach(i),Wo.forEach(i),jo=a(e),g(Rt.$$.fragment,e),Fo=a(e),eo=d(e,"P",{}),w(eo).forEach(i),this.h()},h(){$(n,"name","hf:doc:metadata"),$(n,"content",di),Vr(ee,"float","right"),$(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(we,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$($e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,p){o(document.head,n),m(e,k,p),m(e,r,p),m(e,c,p),m(e,h,p),m(e,T,p),m(e,ee,p),m(e,to,p),_(Qe,e,p),m(e,no,p),m(e,Se,p),m(e,oo,p),m(e,Le,p),m(e,so,p),m(e,Ee,p),m(e,ao,p),_(ue,e,p),m(e,ro,p),m(e,Ye,p),m(e,io,p),m(e,He,p),m(e,lo,p),_(fe,e,p),m(e,co,p),_(De,e,p),m(e,po,p),m(e,N,p),o(N,Ht),o(N,Ro),o(N,Dt),o(N,Vo),o(N,Ae),o(Ae,At),o(Ae,Xo),_(Oe,Ae,null),o(N,No),o(N,Ot),o(N,Go),o(N,Pe),o(Pe,Pt),o(Pe,Qo),_(Ke,Pe,null),m(e,mo,p),_(et,e,p),m(e,ho,p),m(e,E,p),_(tt,E,null),o(E,So),o(E,Kt),o(E,Lo),o(E,en),o(E,Eo),_(ge,E,null),m(e,uo,p),_(nt,e,p),m(e,fo,p),m(e,F,p),_(ot,F,null),o(F,Yo),o(F,tn),o(F,Ho),o(F,nn),o(F,Do),o(F,on),o(F,Ao),_(_e,F,null),o(F,Oo),o(F,D),_(st,D,null),o(D,Po),o(D,sn),o(D,Ko),o(D,an),o(D,es),o(D,rn),m(e,go,p),_(at,e,p),m(e,_o,p),m(e,x,p),_(rt,x,null),o(x,ts),o(x,ln),o(x,ns),o(x,dn),o(x,os),o(x,cn),o(x,ss),_(be,x,null),o(x,as),o(x,G),_(it,G,null),o(G,rs),o(G,pn),o(G,is),o(G,mn),o(G,ls),o(G,hn),o(G,ds),o(G,un),o(x,cs),o(x,ye),_(lt,ye,null),o(ye,ps),o(ye,fn),o(x,ms),o(x,Me),_(dt,Me,null),o(Me,hs),o(Me,gn),o(x,us),o(x,ke),_(ct,ke,null),o(ke,fs),o(ke,_n),m(e,bo,p),_(pt,e,p),m(e,yo,p),m(e,U,p),_(mt,U,null),o(U,gs),o(U,bn),o(U,_s),o(U,yn),o(U,bs),_(Te,U,null),o(U,ys),o(U,A),_(ht,A,null),o(A,Ms),o(A,Mn),o(A,ks),o(A,kn),o(A,Ts),o(A,Tn),o(U,vs),o(U,ve),_(ut,ve,null),o(ve,ws),o(ve,vn),o(U,$s),o(U,we),_(ft,we,null),o(we,Js),o(we,wn),o(U,xs),o(U,$e),_(gt,$e,null),o($e,Us),o($e,$n),o(U,Bs),o(U,Je),_(_t,Je,null),o(Je,Cs),o(Je,Jn),m(e,Mo,p),_(bt,e,p),m(e,ko,p),m(e,z,p),_(yt,z,null),o(z,zs),o(z,xn),o(z,js),o(z,Un),o(z,Fs),_(xe,z,null),o(z,qs),o(z,Q),_(Mt,Q,null),o(Q,Is),o(Q,Bn),o(Q,Zs),o(Q,Cn),o(Q,Ws),o(Q,zn),o(Q,Rs),o(Q,jn),o(z,Vs),o(z,Ue),_(kt,Ue,null),o(Ue,Xs),o(Ue,Fn),o(z,Ns),o(z,Be),_(Tt,Be,null),o(Be,Gs),o(Be,qn),m(e,To,p),_(vt,e,p),m(e,vo,p),m(e,I,p),_(wt,I,null),o(I,Qs),o(I,In),o(I,Ss),o(I,Zn),o(I,Ls),o(I,Wn),o(I,Es),o(I,ae),_($t,ae,null),o(ae,Ys),o(ae,Rn),o(ae,Hs),_(Ce,ae,null),m(e,wo,p),_(Jt,e,p),m(e,$o,p),m(e,Z,p),_(xt,Z,null),o(Z,Ds),o(Z,Vn),o(Z,As),o(Z,Xn),o(Z,Os),o(Z,Nn),o(Z,Ps),o(Z,S),_(Ut,S,null),o(S,Ks),o(S,Gn),o(S,ea),_(ze,S,null),o(S,ta),_(je,S,null),o(S,na),_(Fe,S,null),m(e,Jo,p),_(Bt,e,p),m(e,xo,p),m(e,W,p),_(Ct,W,null),o(W,oa),o(W,Qn),o(W,sa),o(W,Sn),o(W,aa),o(W,Ln),o(W,ra),o(W,O),_(zt,O,null),o(O,ia),o(O,En),o(O,la),_(qe,O,null),o(O,da),_(Ie,O,null),m(e,Uo,p),_(jt,e,p),m(e,Bo,p),m(e,R,p),_(Ft,R,null),o(R,ca),o(R,Yn),o(R,pa),o(R,Hn),o(R,ma),o(R,Dn),o(R,ha),o(R,L),_(qt,L,null),o(L,ua),o(L,An),o(L,fa),_(Ze,L,null),o(L,ga),_(We,L,null),o(L,_a),_(Re,L,null),m(e,Co,p),_(It,e,p),m(e,zo,p),m(e,he,p),_(Zt,he,null),o(he,ba),o(he,P),_(Wt,P,null),o(P,ya),o(P,On),o(P,Ma),_(Ve,P,null),o(P,ka),_(Xe,P,null),m(e,jo,p),_(Rt,e,p),m(e,Fo,p),m(e,eo,p),qo=!0},p(e,[p]){const Y={};p&2&&(Y.$$scope={dirty:p,ctx:e}),ue.$set(Y);const Vt={};p&2&&(Vt.$$scope={dirty:p,ctx:e}),fe.$set(Vt);const Xt={};p&2&&(Xt.$$scope={dirty:p,ctx:e}),ge.$set(Xt);const te={};p&2&&(te.$$scope={dirty:p,ctx:e}),_e.$set(te);const V={};p&2&&(V.$$scope={dirty:p,ctx:e}),be.$set(V);const ne={};p&2&&(ne.$$scope={dirty:p,ctx:e}),Te.$set(ne);const B={};p&2&&(B.$$scope={dirty:p,ctx:e}),xe.$set(B);const H={};p&2&&(H.$$scope={dirty:p,ctx:e}),Ce.$set(H);const Nt={};p&2&&(Nt.$$scope={dirty:p,ctx:e}),ze.$set(Nt);const Gt={};p&2&&(Gt.$$scope={dirty:p,ctx:e}),je.$set(Gt);const Qt={};p&2&&(Qt.$$scope={dirty:p,ctx:e}),Fe.$set(Qt);const C={};p&2&&(C.$$scope={dirty:p,ctx:e}),qe.$set(C);const oe={};p&2&&(oe.$$scope={dirty:p,ctx:e}),Ie.$set(oe);const St={};p&2&&(St.$$scope={dirty:p,ctx:e}),Ze.$set(St);const Lt={};p&2&&(Lt.$$scope={dirty:p,ctx:e}),We.$set(Lt);const Et={};p&2&&(Et.$$scope={dirty:p,ctx:e}),Re.$set(Et);const Yt={};p&2&&(Yt.$$scope={dirty:p,ctx:e}),Ve.$set(Yt);const q={};p&2&&(q.$$scope={dirty:p,ctx:e}),Xe.$set(q)},i(e){qo||(b(Qe.$$.fragment,e),b(ue.$$.fragment,e),b(fe.$$.fragment,e),b(De.$$.fragment,e),b(Oe.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(ge.$$.fragment,e),b(nt.$$.fragment,e),b(ot.$$.fragment,e),b(_e.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b(rt.$$.fragment,e),b(be.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(Te.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(gt.$$.fragment,e),b(_t.$$.fragment,e),b(bt.$$.fragment,e),b(yt.$$.fragment,e),b(xe.$$.fragment,e),b(Mt.$$.fragment,e),b(kt.$$.fragment,e),b(Tt.$$.fragment,e),b(vt.$$.fragment,e),b(wt.$$.fragment,e),b($t.$$.fragment,e),b(Ce.$$.fragment,e),b(Jt.$$.fragment,e),b(xt.$$.fragment,e),b(Ut.$$.fragment,e),b(ze.$$.fragment,e),b(je.$$.fragment,e),b(Fe.$$.fragment,e),b(Bt.$$.fragment,e),b(Ct.$$.fragment,e),b(zt.$$.fragment,e),b(qe.$$.fragment,e),b(Ie.$$.fragment,e),b(jt.$$.fragment,e),b(Ft.$$.fragment,e),b(qt.$$.fragment,e),b(Ze.$$.fragment,e),b(We.$$.fragment,e),b(Re.$$.fragment,e),b(It.$$.fragment,e),b(Zt.$$.fragment,e),b(Wt.$$.fragment,e),b(Ve.$$.fragment,e),b(Xe.$$.fragment,e),b(Rt.$$.fragment,e),qo=!0)},o(e){y(Qe.$$.fragment,e),y(ue.$$.fragment,e),y(fe.$$.fragment,e),y(De.$$.fragment,e),y(Oe.$$.fragment,e),y(Ke.$$.fragment,e),y(et.$$.fragment,e),y(tt.$$.fragment,e),y(ge.$$.fragment,e),y(nt.$$.fragment,e),y(ot.$$.fragment,e),y(_e.$$.fragment,e),y(st.$$.fragment,e),y(at.$$.fragment,e),y(rt.$$.fragment,e),y(be.$$.fragment,e),y(it.$$.fragment,e),y(lt.$$.fragment,e),y(dt.$$.fragment,e),y(ct.$$.fragment,e),y(pt.$$.fragment,e),y(mt.$$.fragment,e),y(Te.$$.fragment,e),y(ht.$$.fragment,e),y(ut.$$.fragment,e),y(ft.$$.fragment,e),y(gt.$$.fragment,e),y(_t.$$.fragment,e),y(bt.$$.fragment,e),y(yt.$$.fragment,e),y(xe.$$.fragment,e),y(Mt.$$.fragment,e),y(kt.$$.fragment,e),y(Tt.$$.fragment,e),y(vt.$$.fragment,e),y(wt.$$.fragment,e),y($t.$$.fragment,e),y(Ce.$$.fragment,e),y(Jt.$$.fragment,e),y(xt.$$.fragment,e),y(Ut.$$.fragment,e),y(ze.$$.fragment,e),y(je.$$.fragment,e),y(Fe.$$.fragment,e),y(Bt.$$.fragment,e),y(Ct.$$.fragment,e),y(zt.$$.fragment,e),y(qe.$$.fragment,e),y(Ie.$$.fragment,e),y(jt.$$.fragment,e),y(Ft.$$.fragment,e),y(qt.$$.fragment,e),y(Ze.$$.fragment,e),y(We.$$.fragment,e),y(Re.$$.fragment,e),y(It.$$.fragment,e),y(Zt.$$.fragment,e),y(Wt.$$.fragment,e),y(Ve.$$.fragment,e),y(Xe.$$.fragment,e),y(Rt.$$.fragment,e),qo=!1},d(e){e&&(i(k),i(r),i(c),i(h),i(T),i(ee),i(to),i(no),i(Se),i(oo),i(Le),i(so),i(Ee),i(ao),i(ro),i(Ye),i(io),i(He),i(lo),i(co),i(po),i(N),i(mo),i(ho),i(E),i(uo),i(fo),i(F),i(go),i(_o),i(x),i(bo),i(yo),i(U),i(Mo),i(ko),i(z),i(To),i(vo),i(I),i(wo),i($o),i(Z),i(Jo),i(xo),i(W),i(Uo),i(Bo),i(R),i(Co),i(zo),i(he),i(jo),i(Fo),i(eo)),i(n),M(Qe,e),M(ue,e),M(fe,e),M(De,e),M(Oe),M(Ke),M(et,e),M(tt),M(ge),M(nt,e),M(ot),M(_e),M(st),M(at,e),M(rt),M(be),M(it),M(lt),M(dt),M(ct),M(pt,e),M(mt),M(Te),M(ht),M(ut),M(ft),M(gt),M(_t),M(bt,e),M(yt),M(xe),M(Mt),M(kt),M(Tt),M(vt,e),M(wt),M($t),M(Ce),M(Jt,e),M(xt),M(Ut),M(ze),M(je),M(Fe),M(Bt,e),M(Ct),M(zt),M(qe),M(Ie),M(jt,e),M(Ft),M(qt),M(Ze),M(We),M(Re),M(It,e),M(Zt),M(Wt),M(Ve),M(Xe),M(Rt,e)}}}const di='{"title":"mBART","local":"mbart","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"MBartConfig","local":"transformers.MBartConfig","sections":[],"depth":2},{"title":"MBartTokenizer","local":"transformers.MBartTokenizer","sections":[],"depth":2},{"title":"MBartTokenizerFast","local":"transformers.MBartTokenizerFast","sections":[],"depth":2},{"title":"MBart50Tokenizer","local":"transformers.MBart50Tokenizer","sections":[],"depth":2},{"title":"MBart50TokenizerFast","local":"transformers.MBart50TokenizerFast","sections":[],"depth":2},{"title":"MBartModel","local":"transformers.MBartModel","sections":[],"depth":2},{"title":"MBartForConditionalGeneration","local":"transformers.MBartForConditionalGeneration","sections":[],"depth":2},{"title":"MBartForQuestionAnswering","local":"transformers.MBartForQuestionAnswering","sections":[],"depth":2},{"title":"MBartForSequenceClassification","local":"transformers.MBartForSequenceClassification","sections":[],"depth":2},{"title":"MBartForCausalLM","local":"transformers.MBartForCausalLM","sections":[],"depth":2}],"depth":1}';function ci(v){return Ir(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class yi extends Zr{constructor(n){super(),Wr(this,n,ci,li,qr,{})}}export{yi as component};
