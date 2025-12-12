import{s as Eo,o as Ao,n as G}from"../chunks/scheduler.18a86fab.js";import{S as Oo,i as Do,g as h,s as i,r as f,A as Ko,h as m,f as a,c as d,j as F,x as T,u as _,k as V,l as en,y as u,a as l,v as y,d as b,t as M,w}from"../chunks/index.98837b22.js";import{T as it}from"../chunks/Tip.77304350.js";import{D as S}from"../chunks/Docstring.a1ef7999.js";import{C as E}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as dt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as D,E as tn}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as on,a as Mo}from"../chunks/HfOption.6641485e.js";function nn(B){let t,p='This model was contributed by <a href="https://huggingface.co/vasudevgupta" rel="nofollow">vasudevgupta</a>.',o,r,g="Click on the BigBirdPegasus models in the right sidebar for more examples of how to apply BigBirdPegasus to different language tasks.";return{c(){t=h("p"),t.innerHTML=p,o=i(),r=h("p"),r.textContent=g},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-ndy5wp"&&(t.innerHTML=p),o=d(n),r=m(n,"P",{"data-svelte-h":!0}),T(r)!=="svelte-tn582l"&&(r.textContent=g)},m(n,c){l(n,t,c),l(n,o,c),l(n,r,c)},p:G,d(n){n&&(a(t),a(o),a(r))}}}function sn(B){let t,p;return t=new E({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJzdW1tYXJpemF0aW9uJTIyJTJDJTBBJTIwJTIwJTIwJTIwbW9kZWwlM0QlMjJnb29nbGUlMkZiaWdiaXJkLXBlZ2FzdXMtbGFyZ2UtYXJ4aXYlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MzIlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUyMiUyMiUyMlBsYW50cyUyMGFyZSUyMGFtb25nJTIwdGhlJTIwbW9zdCUyMHJlbWFya2FibGUlMjBhbmQlMjBlc3NlbnRpYWwlMjBsaWZlJTIwZm9ybXMlMjBvbiUyMEVhcnRoJTJDJTIwcG9zc2Vzc2luZyUyMGElMjB1bmlxdWUlMjBhYmlsaXR5JTIwdG8lMjBwcm9kdWNlJTIwdGhlaXIlMjBvd24lMjBmb29kJTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMHBob3Rvc3ludGhlc2lzLiUyMFRoaXMlMjBjb21wbGV4JTIwYmlvY2hlbWljYWwlMjBwcm9jZXNzJTIwaXMlMjBmdW5kYW1lbnRhbCUyMG5vdCUyMG9ubHklMjB0byUyMHBsYW50JTIwbGlmZSUyMGJ1dCUyMHRvJTIwdmlydHVhbGx5JTIwYWxsJTIwbGlmZSUyMG9uJTIwdGhlJTIwcGxhbmV0LiUwQVRocm91Z2glMjBwaG90b3N5bnRoZXNpcyUyQyUyMHBsYW50cyUyMGNhcHR1cmUlMjBlbmVyZ3klMjBmcm9tJTIwc3VubGlnaHQlMjB1c2luZyUyMGElMjBncmVlbiUyMHBpZ21lbnQlMjBjYWxsZWQlMjBjaGxvcm9waHlsbCUyQyUyMHdoaWNoJTIwaXMlMjBsb2NhdGVkJTIwaW4lMjBzcGVjaWFsaXplZCUyMGNlbGwlMjBzdHJ1Y3R1cmVzJTIwY2FsbGVkJTIwY2hsb3JvcGxhc3RzLiUyMEluJTIwdGhlJTIwcHJlc2VuY2UlMjBvZiUyMGxpZ2h0JTJDJTIwcGxhbnRzJTIwYWJzb3JiJTIwY2FyYm9uJTIwZGlveGlkZSUyMGZyb20lMjB0aGUlMjBhdG1vc3BoZXJlJTIwdGhyb3VnaCUyMHNtYWxsJTIwcG9yZXMlMjBpbiUyMHRoZWlyJTIwbGVhdmVzJTIwY2FsbGVkJTIwc3RvbWF0YSUyQyUyMGFuZCUyMHRha2UlMjBpbiUyMHdhdGVyJTIwZnJvbSUyMHRoZSUyMHNvaWwlMjB0aHJvdWdoJTIwdGhlaXIlMjByb290JTIwc3lzdGVtcy4lMEFUaGVzZSUyMGluZ3JlZGllbnRzJTIwYXJlJTIwdGhlbiUyMHRyYW5zZm9ybWVkJTIwaW50byUyMGdsdWNvc2UlMkMlMjBhJTIwdHlwZSUyMG9mJTIwc3VnYXIlMjB0aGF0JTIwc2VydmVzJTIwYXMlMjBhJTIwc291cmNlJTIwb2YlMjBjaGVtaWNhbCUyMGVuZXJneSUyQyUyMGFuZCUyMG94eWdlbiUyQyUyMHdoaWNoJTIwaXMlMjByZWxlYXNlZCUyMGFzJTIwYSUyMGJ5cHJvZHVjdCUyMGludG8lMjB0aGUlMjBhdG1vc3BoZXJlLiUyMFRoZSUyMGdsdWNvc2UlMjBwcm9kdWNlZCUyMGR1cmluZyUyMHBob3Rvc3ludGhlc2lzJTIwaXMlMjBub3QlMjBqdXN0JTIwdXNlZCUyMGltbWVkaWF0ZWx5JTNCJTIwcGxhbnRzJTIwYWxzbyUyMHN0b3JlJTIwaXQlMjBhcyUyMHN0YXJjaCUyMG9yJTIwY29udmVydCUyMGl0JTIwaW50byUyMG90aGVyJTIwb3JnYW5pYyUyMGNvbXBvdW5kcyUyMGxpa2UlMjBjZWxsdWxvc2UlMkMlMjB3aGljaCUyMGlzJTIwZXNzZW50aWFsJTIwZm9yJTIwYnVpbGRpbmclMjB0aGVpciUyMGNlbGx1bGFyJTIwc3RydWN0dXJlLiUwQVRoaXMlMjBlbmVyZ3klMjByZXNlcnZlJTIwYWxsb3dzJTIwdGhlbSUyMHRvJTIwZ3JvdyUyQyUyMGRldmVsb3AlMjBsZWF2ZXMlMkMlMjBwcm9kdWNlJTIwZmxvd2VycyUyQyUyMGJlYXIlMjBmcnVpdCUyQyUyMGFuZCUyMGNhcnJ5JTIwb3V0JTIwdmFyaW91cyUyMHBoeXNpb2xvZ2ljYWwlMjBwcm9jZXNzZXMlMjB0aHJvdWdob3V0JTIwdGhlaXIlMjBsaWZlY3ljbGUuJTIyJTIyJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;summarization&quot;</span>,
    model=<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>,
    dtype=torch.float32,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;&quot;&quot;Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle.&quot;&quot;&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,r){y(t,o,r),p=!0},p:G,i(o){p||(b(t.$$.fragment,o),p=!0)},o(o){M(t.$$.fragment,o),p=!1},d(o){w(t,o)}}}function an(B){let t,p;return t=new E({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvclNlcTJTZXFMTSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRmJpZ2JpcmQtcGVnYXN1cy1sYXJnZS1hcnhpdiUyMiUwQSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvclNlcTJTZXFMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGYmlnYmlyZC1wZWdhc3VzLWxhcmdlLWFyeGl2JTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBKSUwQSUwQWlucHV0X3RleHQlMjAlM0QlMjAlMjIlMjIlMjJQbGFudHMlMjBhcmUlMjBhbW9uZyUyMHRoZSUyMG1vc3QlMjByZW1hcmthYmxlJTIwYW5kJTIwZXNzZW50aWFsJTIwbGlmZSUyMGZvcm1zJTIwb24lMjBFYXJ0aCUyQyUyMHBvc3Nlc3NpbmclMjBhJTIwdW5pcXVlJTIwYWJpbGl0eSUyMHRvJTIwcHJvZHVjZSUyMHRoZWlyJTIwb3duJTIwZm9vZCUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjBwaG90b3N5bnRoZXNpcy4lMjBUaGlzJTIwY29tcGxleCUyMGJpb2NoZW1pY2FsJTIwcHJvY2VzcyUyMGlzJTIwZnVuZGFtZW50YWwlMjBub3QlMjBvbmx5JTIwdG8lMjBwbGFudCUyMGxpZmUlMjBidXQlMjB0byUyMHZpcnR1YWxseSUyMGFsbCUyMGxpZmUlMjBvbiUyMHRoZSUyMHBsYW5ldC4lMEFUaHJvdWdoJTIwcGhvdG9zeW50aGVzaXMlMkMlMjBwbGFudHMlMjBjYXB0dXJlJTIwZW5lcmd5JTIwZnJvbSUyMHN1bmxpZ2h0JTIwdXNpbmclMjBhJTIwZ3JlZW4lMjBwaWdtZW50JTIwY2FsbGVkJTIwY2hsb3JvcGh5bGwlMkMlMjB3aGljaCUyMGlzJTIwbG9jYXRlZCUyMGluJTIwc3BlY2lhbGl6ZWQlMjBjZWxsJTIwc3RydWN0dXJlcyUyMGNhbGxlZCUyMGNobG9yb3BsYXN0cy4lMjBJbiUyMHRoZSUyMHByZXNlbmNlJTIwb2YlMjBsaWdodCUyQyUyMHBsYW50cyUyMGFic29yYiUyMGNhcmJvbiUyMGRpb3hpZGUlMjBmcm9tJTIwdGhlJTIwYXRtb3NwaGVyZSUyMHRocm91Z2glMjBzbWFsbCUyMHBvcmVzJTIwaW4lMjB0aGVpciUyMGxlYXZlcyUyMGNhbGxlZCUyMHN0b21hdGElMkMlMjBhbmQlMjB0YWtlJTIwaW4lMjB3YXRlciUyMGZyb20lMjB0aGUlMjBzb2lsJTIwdGhyb3VnaCUyMHRoZWlyJTIwcm9vdCUyMHN5c3RlbXMuJTBBVGhlc2UlMjBpbmdyZWRpZW50cyUyMGFyZSUyMHRoZW4lMjB0cmFuc2Zvcm1lZCUyMGludG8lMjBnbHVjb3NlJTJDJTIwYSUyMHR5cGUlMjBvZiUyMHN1Z2FyJTIwdGhhdCUyMHNlcnZlcyUyMGFzJTIwYSUyMHNvdXJjZSUyMG9mJTIwY2hlbWljYWwlMjBlbmVyZ3klMkMlMjBhbmQlMjBveHlnZW4lMkMlMjB3aGljaCUyMGlzJTIwcmVsZWFzZWQlMjBhcyUyMGElMjBieXByb2R1Y3QlMjBpbnRvJTIwdGhlJTIwYXRtb3NwaGVyZS4lMjBUaGUlMjBnbHVjb3NlJTIwcHJvZHVjZWQlMjBkdXJpbmclMjBwaG90b3N5bnRoZXNpcyUyMGlzJTIwbm90JTIwanVzdCUyMHVzZWQlMjBpbW1lZGlhdGVseSUzQiUyMHBsYW50cyUyMGFsc28lMjBzdG9yZSUyMGl0JTIwYXMlMjBzdGFyY2glMjBvciUyMGNvbnZlcnQlMjBpdCUyMGludG8lMjBvdGhlciUyMG9yZ2FuaWMlMjBjb21wb3VuZHMlMjBsaWtlJTIwY2VsbHVsb3NlJTJDJTIwd2hpY2glMjBpcyUyMGVzc2VudGlhbCUyMGZvciUyMGJ1aWxkaW5nJTIwdGhlaXIlMjBjZWxsdWxhciUyMHN0cnVjdHVyZS4lMEFUaGlzJTIwZW5lcmd5JTIwcmVzZXJ2ZSUyMGFsbG93cyUyMHRoZW0lMjB0byUyMGdyb3clMkMlMjBkZXZlbG9wJTIwbGVhdmVzJTJDJTIwcHJvZHVjZSUyMGZsb3dlcnMlMkMlMjBiZWFyJTIwZnJ1aXQlMkMlMjBhbmQlMjBjYXJyeSUyMG91dCUyMHZhcmlvdXMlMjBwaHlzaW9sb2dpY2FsJTIwcHJvY2Vzc2VzJTIwdGhyb3VnaG91dCUyMHRoZWlyJTIwbGlmZWN5Y2xlLiUyMiUyMiUyMiUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcihpbnB1dF90ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)

input_text = <span class="hljs-string">&quot;&quot;&quot;Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle.&quot;&quot;&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,r){y(t,o,r),p=!0},p:G,i(o){p||(b(t.$$.fragment,o),p=!0)},o(o){M(t.$$.fragment,o),p=!1},d(o){w(t,o)}}}function rn(B){let t,p;return t=new E({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwYXJlJTIwYW1vbmclMjB0aGUlMjBtb3N0JTIwcmVtYXJrYWJsZSUyMGFuZCUyMGVzc2VudGlhbCUyMGxpZmUlMjBmb3JtcyUyMG9uJTIwRWFydGglMkMlMjBwb3NzZXNzaW5nJTIwYSUyMHVuaXF1ZSUyMGFiaWxpdHklMjB0byUyMHByb2R1Y2UlMjB0aGVpciUyMG93biUyMGZvb2QlMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIwcGhvdG9zeW50aGVzaXMuJTIwVGhpcyUyMGNvbXBsZXglMjBiaW9jaGVtaWNhbCUyMHByb2Nlc3MlMjBpcyUyMGZ1bmRhbWVudGFsJTIwbm90JTIwb25seSUyMHRvJTIwcGxhbnQlMjBsaWZlJTIwYnV0JTIwdG8lMjB2aXJ0dWFsbHklMjBhbGwlMjBsaWZlJTIwb24lMjB0aGUlMjBwbGFuZXQuJTIwVGhyb3VnaCUyMHBob3Rvc3ludGhlc2lzJTJDJTIwcGxhbnRzJTIwY2FwdHVyZSUyMGVuZXJneSUyMGZyb20lMjBzdW5saWdodCUyMHVzaW5nJTIwYSUyMGdyZWVuJTIwcGlnbWVudCUyMGNhbGxlZCUyMGNobG9yb3BoeWxsJTJDJTIwd2hpY2glMjBpcyUyMGxvY2F0ZWQlMjBpbiUyMHNwZWNpYWxpemVkJTIwY2VsbCUyMHN0cnVjdHVyZXMlMjBjYWxsZWQlMjBjaGxvcm9wbGFzdHMuJTIyJTIwJTdDJTIwdHJhbnNmb3JtZXJzLWNsaSUyMHJ1biUyMC0tdGFzayUyMHN1bW1hcml6YXRpb24lMjAtLW1vZGVsJTIwZ29vZ2xlJTJGYmlnYmlyZC1wZWdhc3VzLWxhcmdlLWFyeGl2JTIwLS1kZXZpY2UlMjAw",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet. Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts.&quot;</span> | transformers-cli run --task summarization --model google/bigbird-pegasus-large-arxiv --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,r){y(t,o,r),p=!0},p:G,i(o){p||(b(t.$$.fragment,o),p=!0)},o(o){M(t.$$.fragment,o),p=!1},d(o){w(t,o)}}}function dn(B){let t,p,o,r,g,n;return t=new Mo({props:{id:"usage",option:"Pipeline",$$slots:{default:[sn]},$$scope:{ctx:B}}}),o=new Mo({props:{id:"usage",option:"AutoModel",$$slots:{default:[an]},$$scope:{ctx:B}}}),g=new Mo({props:{id:"usage",option:"transformers-cli",$$slots:{default:[rn]},$$scope:{ctx:B}}}),{c(){f(t.$$.fragment),p=i(),f(o.$$.fragment),r=i(),f(g.$$.fragment)},l(c){_(t.$$.fragment,c),p=d(c),_(o.$$.fragment,c),r=d(c),_(g.$$.fragment,c)},m(c,v){y(t,c,v),l(c,p,v),y(o,c,v),l(c,r,v),y(g,c,v),n=!0},p(c,v){const lt={};v&2&&(lt.$$scope={dirty:v,ctx:c}),t.$set(lt);const ue={};v&2&&(ue.$$scope={dirty:v,ctx:c}),o.$set(ue);const Q={};v&2&&(Q.$$scope={dirty:v,ctx:c}),g.$set(Q)},i(c){n||(b(t.$$.fragment,c),b(o.$$.fragment,c),b(g.$$.fragment,c),n=!0)},o(c){M(t.$$.fragment,c),M(o.$$.fragment,c),M(g.$$.fragment,c),n=!1},d(c){c&&(a(p),a(r)),w(t,c),w(o,c),w(g,c)}}}function ln(B){let t,p="Example:",o,r,g;return r=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJpZ0JpcmRQZWdhc3VzQ29uZmlnJTJDJTIwQmlnQmlyZFBlZ2FzdXNNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBCaWdCaXJkUGVnYXN1cyUyMGJpZ2JpcmQtcGVnYXN1cy1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEJpZ0JpcmRQZWdhc3VzQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGJpZ2JpcmQtcGVnYXN1cy1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBCaWdCaXJkUGVnYXN1c01vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BigBirdPegasusConfig, BigBirdPegasusModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a BigBirdPegasus bigbird-pegasus-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BigBirdPegasusConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the bigbird-pegasus-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdPegasusModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=h("p"),t.textContent=p,o=i(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=p),o=d(n),_(r.$$.fragment,n)},m(n,c){l(n,t,c),l(n,o,c),y(r,n,c),g=!0},p:G,i(n){g||(b(r.$$.fragment,n),g=!0)},o(n){M(r.$$.fragment,n),g=!1},d(n){n&&(a(t),a(o)),w(r,n)}}}function cn(B){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,r){l(o,t,r)},p:G,d(o){o&&a(t)}}}function pn(B){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,r){l(o,t,r)},p:G,d(o){o&&a(t)}}}function un(B){let t,p="Example summarization:",o,r,g;return r=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaWdCaXJkUGVnYXN1c0ZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQW1vZGVsJTIwJTNEJTIwQmlnQmlyZFBlZ2FzdXNGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmJpZ2JpcmQtcGVnYXN1cy1sYXJnZS1hcnhpdiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZiaWdiaXJkLXBlZ2FzdXMtbGFyZ2UtYXJ4aXYlMjIpJTBBJTBBQVJUSUNMRV9UT19TVU1NQVJJWkUlMjAlM0QlMjAoJTBBJTIwJTIwJTIwJTIwJTIyVGhlJTIwZG9taW5hbnQlMjBzZXF1ZW5jZSUyMHRyYW5zZHVjdGlvbiUyMG1vZGVscyUyMGFyZSUyMGJhc2VkJTIwb24lMjBjb21wbGV4JTIwcmVjdXJyZW50JTIwb3IlMjBjb252b2x1dGlvbmFsJTIwbmV1cmFsJTIwJTIyJTBBJTIwJTIwJTIwJTIwJTIybmV0d29ya3MlMjBpbiUyMGFuJTIwZW5jb2Rlci1kZWNvZGVyJTIwY29uZmlndXJhdGlvbi4lMjBUaGUlMjBiZXN0JTIwcGVyZm9ybWluZyUyMG1vZGVscyUyMGFsc28lMjBjb25uZWN0JTIwdGhlJTIwZW5jb2RlciUyMCUyMiUwQSUyMCUyMCUyMCUyMCUyMmFuZCUyMGRlY29kZXIlMjB0aHJvdWdoJTIwYW4lMjBhdHRlbnRpb24lMjBtZWNoYW5pc20uJTIwV2UlMjBwcm9wb3NlJTIwYSUyMG5ldyUyMHNpbXBsZSUyMG5ldHdvcmslMjBhcmNoaXRlY3R1cmUlMkMlMjB0aGUlMjBUcmFuc2Zvcm1lciUyQyUyMCUyMiUwQSUyMCUyMCUyMCUyMCUyMmJhc2VkJTIwc29sZWx5JTIwb24lMjBhdHRlbnRpb24lMjBtZWNoYW5pc21zJTJDJTIwZGlzcGVuc2luZyUyMHdpdGglMjByZWN1cnJlbmNlJTIwYW5kJTIwY29udm9sdXRpb25zJTIwZW50aXJlbHkuJTIwJTIyJTBBJTIwJTIwJTIwJTIwJTIyRXhwZXJpbWVudHMlMjBvbiUyMHR3byUyMG1hY2hpbmUlMjB0cmFuc2xhdGlvbiUyMHRhc2tzJTIwc2hvdyUyMHRoZXNlJTIwbW9kZWxzJTIwdG8lMjBiZSUyMHN1cGVyaW9yJTIwaW4lMjBxdWFsaXR5JTIwJTIyJTBBJTIwJTIwJTIwJTIwJTIyd2hpbGUlMjBiZWluZyUyMG1vcmUlMjBwYXJhbGxlbGl6YWJsZSUyMGFuZCUyMHJlcXVpcmluZyUyMHNpZ25pZmljYW50bHklMjBsZXNzJTIwdGltZSUyMHRvJTIwdHJhaW4uJTIyJTBBKSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglNUJBUlRJQ0xFX1RPX1NVTU1BUklaRSU1RCUyQyUyMG1heF9sZW5ndGglM0Q0MDk2JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHRydW5jYXRpb24lM0RUcnVlKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTIwU3VtbWFyeSUwQXN1bW1hcnlfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTJDJTIwbnVtX2JlYW1zJTNENCUyQyUyMG1heF9sZW5ndGglM0QxNSklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKHN1bW1hcnlfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdPegasusForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdPegasusForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ARTICLE_TO_SUMMARIZE = (
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;The dominant sequence transduction models are based on complex recurrent or convolutional neural &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;networks in an encoder-decoder configuration. The best performing models also connect the encoder &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Experiments on two machine translation tasks show these models to be superior in quality &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;while being more parallelizable and requiring significantly less time to train.&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=<span class="hljs-number">4096</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, truncation=<span class="hljs-literal">True</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate Summary</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>summary_ids = model.generate(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>], num_beams=<span class="hljs-number">4</span>, max_length=<span class="hljs-number">15</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(summary_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&#x27;dominant sequence models are based on recurrent or convolutional neural networks .&#x27;</span>`,wrap:!1}}),{c(){t=h("p"),t.textContent=p,o=i(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-iw9ecv"&&(t.textContent=p),o=d(n),_(r.$$.fragment,n)},m(n,c){l(n,t,c),l(n,o,c),y(r,n,c),g=!0},p:G,i(n){g||(b(r.$$.fragment,n),g=!0)},o(n){M(r.$$.fragment,n),g=!1},d(n){n&&(a(t),a(o)),w(r,n)}}}function hn(B){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,r){l(o,t,r)},p:G,d(o){o&&a(t)}}}function mn(B){let t,p="Example of single-label classification:",o,r,g;return r=new E({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJpZ0JpcmRQZWdhc3VzRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmJpZ2JpcmQtcGVnYXN1cy1sYXJnZS1hcnhpdiUyMiklMEFtb2RlbCUyMCUzRCUyMEJpZ0JpcmRQZWdhc3VzRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGYmlnYmlyZC1wZWdhc3VzLWxhcmdlLWFyeGl2JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBCaWdCaXJkUGVnYXN1c0ZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmJpZ2JpcmQtcGVnYXN1cy1sYXJnZS1hcnhpdiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdPegasusForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdPegasusForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdPegasusForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=h("p"),t.textContent=p,o=i(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-ykxpe4"&&(t.textContent=p),o=d(n),_(r.$$.fragment,n)},m(n,c){l(n,t,c),l(n,o,c),y(r,n,c),g=!0},p:G,i(n){g||(b(r.$$.fragment,n),g=!0)},o(n){M(r.$$.fragment,n),g=!1},d(n){n&&(a(t),a(o)),w(r,n)}}}function gn(B){let t,p="Example of multi-label classification:",o,r,g;return r=new E({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJpZ0JpcmRQZWdhc3VzRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmJpZ2JpcmQtcGVnYXN1cy1sYXJnZS1hcnhpdiUyMiklMEFtb2RlbCUyMCUzRCUyMEJpZ0JpcmRQZWdhc3VzRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGYmlnYmlyZC1wZWdhc3VzLWxhcmdlLWFyeGl2JTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBCaWdCaXJkUGVnYXN1c0ZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRmJpZ2JpcmQtcGVnYXN1cy1sYXJnZS1hcnhpdiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdPegasusForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdPegasusForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdPegasusForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=h("p"),t.textContent=p,o=i(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1l8e32d"&&(t.textContent=p),o=d(n),_(r.$$.fragment,n)},m(n,c){l(n,t,c),l(n,o,c),y(r,n,c),g=!0},p:G,i(n){g||(b(r.$$.fragment,n),g=!0)},o(n){M(r.$$.fragment,n),g=!1},d(n){n&&(a(t),a(o)),w(r,n)}}}function fn(B){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,r){l(o,t,r)},p:G,d(o){o&&a(t)}}}function _n(B){let t,p="Example:",o,r,g;return r=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaWdCaXJkUGVnYXN1c0ZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZiaWdiaXJkLXBlZ2FzdXMtbGFyZ2UtYXJ4aXYlMjIpJTBBbW9kZWwlMjAlM0QlMjBCaWdCaXJkUGVnYXN1c0ZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZiaWdiaXJkLXBlZ2FzdXMtbGFyZ2UtYXJ4aXYlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdPegasusForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdPegasusForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>)

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
...`,wrap:!1}}),{c(){t=h("p"),t.textContent=p,o=i(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=p),o=d(n),_(r.$$.fragment,n)},m(n,c){l(n,t,c),l(n,o,c),y(r,n,c),g=!0},p:G,i(n){g||(b(r.$$.fragment,n),g=!0)},o(n){M(r.$$.fragment,n),g=!1},d(n){n&&(a(t),a(o)),w(r,n)}}}function yn(B){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,r){l(o,t,r)},p:G,d(o){o&&a(t)}}}function bn(B){let t,p="Example:",o,r,g;return r=new E({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaWdCaXJkUGVnYXN1c0ZvckNhdXNhbExNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGYmlnYmlyZC1wZWdhc3VzLWxhcmdlLWFyeGl2JTIyKSUwQW1vZGVsJTIwJTNEJTIwQmlnQmlyZFBlZ2FzdXNGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGYmlnYmlyZC1wZWdhc3VzLWxhcmdlLWFyeGl2JTIyJTJDJTIwYWRkX2Nyb3NzX2F0dGVudGlvbiUzREZhbHNlJTBBKSUwQWFzc2VydCUyMG1vZGVsLmNvbmZpZy5pc19kZWNvZGVyJTJDJTIwZiUyMiU3Qm1vZGVsLl9fY2xhc3NfXyU3RCUyMGhhcyUyMHRvJTIwYmUlMjBjb25maWd1cmVkJTIwYXMlMjBhJTIwZGVjb2Rlci4lMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdPegasusForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdPegasusForCausalLM.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=h("p"),t.textContent=p,o=i(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=p),o=d(n),_(r.$$.fragment,n)},m(n,c){l(n,t,c),l(n,o,c),y(r,n,c),g=!0},p:G,i(n){g||(b(r.$$.fragment,n),g=!0)},o(n){M(r.$$.fragment,n),g=!1},d(n){n&&(a(t),a(o)),w(r,n)}}}function Mn(B){let t,p,o,r,g,n="<em>This model was released on 2020-07-28 and added to Hugging Face Transformers on 2021-05-07.</em>",c,v,lt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',ue,Q,pt,he,wo='<a href="https://huggingface.co/papers/2007.14062" rel="nofollow">BigBirdPegasus</a> is an encoder-decoder (sequence-to-sequence) transformer model for long-input summarization. It extends the <a href="./big_bird">BigBird</a> architecture with an additional pretraining objective borrowed from <a href="./pegasus">Pegasus</a> called gap sequence generation (GSG). Whole sentences are masked and the model has to fill in the gaps in the document. BigBirdPegasus’s ability to keep track of long contexts makes it effective at summarizing lengthy inputs, surpassing the performance of base Pegasus models.',ut,me,To='You can find all the original BigBirdPegasus checkpoints under the <a href="https://huggingface.co/google/models?search=bigbird-pegasus" rel="nofollow">Google</a> organization.',ht,K,mt,ge,Bo='The example below demonstrates how to summarize text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',gt,ee,ft,fe,vo='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',_t,_e,Jo='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to int4.',yt,ye,bt,be,Mt,Me,Uo='<li>BigBirdPegasus also uses the <a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusTokenizer">PegasusTokenizer</a>.</li> <li>Inputs should be padded on the right because BigBird uses absolute position embeddings.</li> <li>BigBirdPegasus supports <code>original_full</code> and <code>block_sparse</code> attention. If the input sequence length is less than 1024, it is recommended to use <code>original_full</code> since sparse patterns don’t offer much benefit for smaller inputs.</li> <li>The current implementation uses window size of 3 blocks and 2 global blocks, only supports the ITC-implementation, and doesn’t support <code>num_random_blocks=0</code>.</li> <li>The sequence length must be divisible by the block size.</li>',wt,we,Tt,Te,ko='Read the <a href="https://huggingface.co/blog/big-bird" rel="nofollow">Understanding BigBird’s Block Sparse Attention</a> blog post for more details about how BigBird’s attention works.',Bt,Be,vt,C,ve,Vt,Pe,jo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusModel">BigBirdPegasusModel</a>. It is used to instantiate
an BigBirdPegasus model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the BigBirdPegasus
<a href="https://huggingface.co/google/bigbird-pegasus-large-arxiv" rel="nofollow">google/bigbird-pegasus-large-arxiv</a> architecture.`,qt,Re,Go=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Xt,te,Jt,Je,Ut,J,Ue,Nt,He,Zo="The bare Bigbird Pegasus Model outputting raw hidden-states without any specific head on top.",Pt,Ye,Co=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rt,Se,Io=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ht,L,ke,Yt,Qe,Wo='The <a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusModel">BigBirdPegasusModel</a> forward method, overrides the <code>__call__</code> special method.',St,oe,kt,je,jt,U,Ge,Qt,Le,$o="The BigBirdPegasus Model with a language modeling head. Can be used for summarization.",Lt,Ee,xo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Et,Ae,zo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,At,q,Ze,Ot,Oe,Fo='The <a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForConditionalGeneration">BigBirdPegasusForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Dt,ne,Kt,se,Gt,Ce,Zt,k,Ie,eo,De,Vo=`BigBirdPegasus model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g.
for GLUE tasks.`,to,Ke,qo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,oo,et,Xo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,no,Z,We,so,tt,No='The <a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForSequenceClassification">BigBirdPegasusForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',ao,ae,ro,re,io,ie,Ct,$e,It,j,xe,lo,ot,Po=`The Bigbird Pegasus transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,co,nt,Ro=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,po,st,Ho=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,uo,X,ze,ho,at,Yo='The <a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForQuestionAnswering">BigBirdPegasusForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',mo,de,go,le,Wt,Fe,$t,A,Ve,fo,N,qe,_o,rt,So='The <a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForCausalLM">BigBirdPegasusForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',yo,ce,bo,pe,xt,Xe,zt,ct,Ft;return Q=new D({props:{title:"BigBirdPegasus",local:"bigbirdpegasus",headingTag:"h1"}}),K=new it({props:{warning:!1,$$slots:{default:[nn]},$$scope:{ctx:B}}}),ee=new on({props:{id:"usage",options:["Pipeline","AutoModel","transformers-cli"],$$slots:{default:[dn]},$$scope:{ctx:B}}}),ye=new E({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBCaXRzQW5kQnl0ZXNDb25maWcoJTBBJTIwJTIwJTIwJTIwbG9hZF9pbl80Yml0JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3F1YW50X3R5cGUlM0QlMjJuZjQlMjIlMEEpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXEyU2VxTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRmJpZ2JpcmQtcGVnYXN1cy1sYXJnZS1hcnhpdiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RxdWFudGl6YXRpb25fY29uZmlnJTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRmJpZ2JpcmQtcGVnYXN1cy1sYXJnZS1hcnhpdiUyMiUwQSklMEElMEFpbnB1dF90ZXh0JTIwJTNEJTIwJTIyJTIyJTIyUGxhbnRzJTIwYXJlJTIwYW1vbmclMjB0aGUlMjBtb3N0JTIwcmVtYXJrYWJsZSUyMGFuZCUyMGVzc2VudGlhbCUyMGxpZmUlMjBmb3JtcyUyMG9uJTIwRWFydGglMkMlMjBwb3NzZXNzaW5nJTIwYSUyMHVuaXF1ZSUyMGFiaWxpdHklMjB0byUyMHByb2R1Y2UlMjB0aGVpciUyMG93biUyMGZvb2QlMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIwcGhvdG9zeW50aGVzaXMuJTIwVGhpcyUyMGNvbXBsZXglMjBiaW9jaGVtaWNhbCUyMHByb2Nlc3MlMjBpcyUyMGZ1bmRhbWVudGFsJTIwbm90JTIwb25seSUyMHRvJTIwcGxhbnQlMjBsaWZlJTIwYnV0JTIwdG8lMjB2aXJ0dWFsbHklMjBhbGwlMjBsaWZlJTIwb24lMjB0aGUlMjBwbGFuZXQuJTBBVGhyb3VnaCUyMHBob3Rvc3ludGhlc2lzJTJDJTIwcGxhbnRzJTIwY2FwdHVyZSUyMGVuZXJneSUyMGZyb20lMjBzdW5saWdodCUyMHVzaW5nJTIwYSUyMGdyZWVuJTIwcGlnbWVudCUyMGNhbGxlZCUyMGNobG9yb3BoeWxsJTJDJTIwd2hpY2glMjBpcyUyMGxvY2F0ZWQlMjBpbiUyMHNwZWNpYWxpemVkJTIwY2VsbCUyMHN0cnVjdHVyZXMlMjBjYWxsZWQlMjBjaGxvcm9wbGFzdHMuJTIwSW4lMjB0aGUlMjBwcmVzZW5jZSUyMG9mJTIwbGlnaHQlMkMlMjBwbGFudHMlMjBhYnNvcmIlMjBjYXJib24lMjBkaW94aWRlJTIwZnJvbSUyMHRoZSUyMGF0bW9zcGhlcmUlMjB0aHJvdWdoJTIwc21hbGwlMjBwb3JlcyUyMGluJTIwdGhlaXIlMjBsZWF2ZXMlMjBjYWxsZWQlMjBzdG9tYXRhJTJDJTIwYW5kJTIwdGFrZSUyMGluJTIwd2F0ZXIlMjBmcm9tJTIwdGhlJTIwc29pbCUyMHRocm91Z2glMjB0aGVpciUyMHJvb3QlMjBzeXN0ZW1zLiUwQVRoZXNlJTIwaW5ncmVkaWVudHMlMjBhcmUlMjB0aGVuJTIwdHJhbnNmb3JtZWQlMjBpbnRvJTIwZ2x1Y29zZSUyQyUyMGElMjB0eXBlJTIwb2YlMjBzdWdhciUyMHRoYXQlMjBzZXJ2ZXMlMjBhcyUyMGElMjBzb3VyY2UlMjBvZiUyMGNoZW1pY2FsJTIwZW5lcmd5JTJDJTIwYW5kJTIwb3h5Z2VuJTJDJTIwd2hpY2glMjBpcyUyMHJlbGVhc2VkJTIwYXMlMjBhJTIwYnlwcm9kdWN0JTIwaW50byUyMHRoZSUyMGF0bW9zcGhlcmUuJTIwVGhlJTIwZ2x1Y29zZSUyMHByb2R1Y2VkJTIwZHVyaW5nJTIwcGhvdG9zeW50aGVzaXMlMjBpcyUyMG5vdCUyMGp1c3QlMjB1c2VkJTIwaW1tZWRpYXRlbHklM0IlMjBwbGFudHMlMjBhbHNvJTIwc3RvcmUlMjBpdCUyMGFzJTIwc3RhcmNoJTIwb3IlMjBjb252ZXJ0JTIwaXQlMjBpbnRvJTIwb3RoZXIlMjBvcmdhbmljJTIwY29tcG91bmRzJTIwbGlrZSUyMGNlbGx1bG9zZSUyQyUyMHdoaWNoJTIwaXMlMjBlc3NlbnRpYWwlMjBmb3IlMjBidWlsZGluZyUyMHRoZWlyJTIwY2VsbHVsYXIlMjBzdHJ1Y3R1cmUuJTBBVGhpcyUyMGVuZXJneSUyMHJlc2VydmUlMjBhbGxvd3MlMjB0aGVtJTIwdG8lMjBncm93JTJDJTIwZGV2ZWxvcCUyMGxlYXZlcyUyQyUyMHByb2R1Y2UlMjBmbG93ZXJzJTJDJTIwYmVhciUyMGZydWl0JTJDJTIwYW5kJTIwY2FycnklMjBvdXQlMjB2YXJpb3VzJTIwcGh5c2lvbG9naWNhbCUyMHByb2Nlc3NlcyUyMHRocm91Z2hvdXQlMjB0aGVpciUyMGxpZmVjeWNsZS4lMjIlMjIlMjIlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoaW5wdXRfdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google/bigbird-pegasus-large-arxiv&quot;</span>
)

input_text = <span class="hljs-string">&quot;&quot;&quot;Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle.&quot;&quot;&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),be=new D({props:{title:"Notes",local:"notes",headingTag:"h2"}}),we=new D({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Be=new D({props:{title:"BigBirdPegasusConfig",local:"transformers.BigBirdPegasusConfig",headingTag:"h2"}}),ve=new S({props:{name:"class transformers.BigBirdPegasusConfig",anchor:"transformers.BigBirdPegasusConfig",parameters:[{name:"vocab_size",val:" = 96103"},{name:"max_position_embeddings",val:" = 4096"},{name:"encoder_layers",val:" = 16"},{name:"encoder_ffn_dim",val:" = 4096"},{name:"encoder_attention_heads",val:" = 16"},{name:"decoder_layers",val:" = 16"},{name:"decoder_ffn_dim",val:" = 4096"},{name:"decoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu_new'"},{name:"d_model",val:" = 1024"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 2"},{name:"classifier_dropout",val:" = 0.0"},{name:"scale_embedding",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 2"},{name:"eos_token_id",val:" = 1"},{name:"attention_type",val:" = 'block_sparse'"},{name:"block_size",val:" = 64"},{name:"num_random_blocks",val:" = 3"},{name:"use_bias",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BigBirdPegasusConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 96103) &#x2014;
Vocabulary size of the BigBirdPegasus model. Defines the number of different tokens that can be represented
by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusModel">BigBirdPegasusModel</a>.`,name:"vocab_size"},{anchor:"transformers.BigBirdPegasusConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.BigBirdPegasusConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.BigBirdPegasusConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.BigBirdPegasusConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.BigBirdPegasusConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.BigBirdPegasusConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.BigBirdPegasusConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.BigBirdPegasusConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu_new&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.BigBirdPegasusConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.BigBirdPegasusConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.BigBirdPegasusConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.BigBirdPegasusConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.BigBirdPegasusConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 1024 or 2048 or 4096).`,name:"max_position_embeddings"},{anchor:"transformers.BigBirdPegasusConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.BigBirdPegasusConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.BigBirdPegasusConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.BigBirdPegasusConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.BigBirdPegasusConfig.attention_type",description:`<strong>attention_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;block_sparse&quot;</code>) &#x2014;
Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
layer (with n^2 complexity) in encoder. Possible values are <code>&quot;original_full&quot;</code> and <code>&quot;block_sparse&quot;</code>.`,name:"attention_type"},{anchor:"transformers.BigBirdPegasusConfig.use_bias",description:`<strong>use_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use bias in query, key, value.`,name:"use_bias"},{anchor:"transformers.BigBirdPegasusConfig.block_size",description:`<strong>block_size</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Size of each block. Useful only when <code>attention_type == &quot;block_sparse&quot;</code>.`,name:"block_size"},{anchor:"transformers.BigBirdPegasusConfig.num_random_blocks",description:`<strong>num_random_blocks</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Each query is going to attend these many number of random blocks. Useful only when <code>attention_type == &quot;block_sparse&quot;</code>.`,name:"num_random_blocks"},{anchor:"transformers.BigBirdPegasusConfig.scale_embeddings",description:`<strong>scale_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to rescale embeddings with (hidden_size ** 0.5).`,name:"scale_embeddings"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/configuration_bigbird_pegasus.py#L31"}}),te=new dt({props:{anchor:"transformers.BigBirdPegasusConfig.example",$$slots:{default:[ln]},$$scope:{ctx:B}}}),Je=new D({props:{title:"BigBirdPegasusModel",local:"transformers.BigBirdPegasusModel",headingTag:"h2"}}),Ue=new S({props:{name:"class transformers.BigBirdPegasusModel",anchor:"transformers.BigBirdPegasusModel",parameters:[{name:"config",val:": BigBirdPegasusConfig"}],parametersDescription:[{anchor:"transformers.BigBirdPegasusModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig">BigBirdPegasusConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2320"}}),ke=new S({props:{name:"forward",anchor:"transformers.BigBirdPegasusModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.BigBirdPegasusModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdPegasusModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdPegasusModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Provide for translation and summarization training. By default, the model will create this tensor by
shifting the <code>input_ids</code> to the right, following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.BigBirdPegasusModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read
<code>modeling_bigbird_pegasus._prepare_decoder_attention_mask</code> and modify to your needs. See diagram 1 in
<a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.BigBirdPegasusModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdPegasusModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BigBirdPegasusModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BigBirdPegasusModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BigBirdPegasusModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BigBirdPegasusModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdPegasusModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BigBirdPegasusModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BigBirdPegasusModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdPegasusModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdPegasusModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BigBirdPegasusModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2354",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig"
>BigBirdPegasusConfig</a>) and inputs.</p>
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
`}}),oe=new it({props:{$$slots:{default:[cn]},$$scope:{ctx:B}}}),je=new D({props:{title:"BigBirdPegasusForConditionalGeneration",local:"transformers.BigBirdPegasusForConditionalGeneration",headingTag:"h2"}}),Ge=new S({props:{name:"class transformers.BigBirdPegasusForConditionalGeneration",anchor:"transformers.BigBirdPegasusForConditionalGeneration",parameters:[{name:"config",val:": BigBirdPegasusConfig"}],parametersDescription:[{anchor:"transformers.BigBirdPegasusForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig">BigBirdPegasusConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2468"}}),Ze=new S({props:{name:"forward",anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Provide for translation and summarization training. By default, the model will create this tensor by
shifting the <code>input_ids</code> to the right, following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read
<code>modeling_bigbird_pegasus._prepare_decoder_attention_mask</code> and modify to your needs. See diagram 1 in
<a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2509",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig"
>BigBirdPegasusConfig</a>) and inputs.</p>
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
`}}),ne=new it({props:{$$slots:{default:[pn]},$$scope:{ctx:B}}}),se=new dt({props:{anchor:"transformers.BigBirdPegasusForConditionalGeneration.forward.example",$$slots:{default:[un]},$$scope:{ctx:B}}}),Ce=new D({props:{title:"BigBirdPegasusForSequenceClassification",local:"transformers.BigBirdPegasusForSequenceClassification",headingTag:"h2"}}),Ie=new S({props:{name:"class transformers.BigBirdPegasusForSequenceClassification",anchor:"transformers.BigBirdPegasusForSequenceClassification",parameters:[{name:"config",val:": BigBirdPegasusConfig"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BigBirdPegasusForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig">BigBirdPegasusConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2641"}}),We=new S({props:{name:"forward",anchor:"transformers.BigBirdPegasusForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Provide for translation and summarization training. By default, the model will create this tensor by
shifting the <code>input_ids</code> to the right, following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read
<code>modeling_bigbird_pegasus._prepare_decoder_attention_mask</code> and modify to your needs. See diagram 1 in
<a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2657",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig"
>BigBirdPegasusConfig</a>) and inputs.</p>
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
`}}),ae=new it({props:{$$slots:{default:[hn]},$$scope:{ctx:B}}}),re=new dt({props:{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.example",$$slots:{default:[mn]},$$scope:{ctx:B}}}),ie=new dt({props:{anchor:"transformers.BigBirdPegasusForSequenceClassification.forward.example-2",$$slots:{default:[gn]},$$scope:{ctx:B}}}),$e=new D({props:{title:"BigBirdPegasusForQuestionAnswering",local:"transformers.BigBirdPegasusForQuestionAnswering",headingTag:"h2"}}),xe=new S({props:{name:"class transformers.BigBirdPegasusForQuestionAnswering",anchor:"transformers.BigBirdPegasusForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BigBirdPegasusForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForQuestionAnswering">BigBirdPegasusForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2775"}}),ze=new S({props:{name:"forward",anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Provide for translation and summarization training. By default, the model will create this tensor by
shifting the <code>input_ids</code> to the right, following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.</p>
<p>If you want to change padding behavior, you should read
<code>modeling_bigbird_pegasus._prepare_decoder_attention_mask</code> and modify to your needs. See diagram 1 in
<a href="https://huggingface.co/papers/1910.13461" rel="nofollow">the paper</a> for more information on the default strategy.`,name:"decoder_attention_mask"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2790",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput"
>transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig"
>BigBirdPegasusConfig</a>) and inputs.</p>
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
`}}),de=new it({props:{$$slots:{default:[fn]},$$scope:{ctx:B}}}),le=new dt({props:{anchor:"transformers.BigBirdPegasusForQuestionAnswering.forward.example",$$slots:{default:[_n]},$$scope:{ctx:B}}}),Fe=new D({props:{title:"BigBirdPegasusForCausalLM",local:"transformers.BigBirdPegasusForCausalLM",headingTag:"h2"}}),Ve=new S({props:{name:"class transformers.BigBirdPegasusForCausalLM",anchor:"transformers.BigBirdPegasusForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2910"}}),qe=new S({props:{name:"forward",anchor:"transformers.BigBirdPegasusForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.BigBirdPegasusForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BigBirdPegasusForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2936",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig"
>BigBirdPegasusConfig</a>) and inputs.</p>
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
`}}),ce=new it({props:{$$slots:{default:[yn]},$$scope:{ctx:B}}}),pe=new dt({props:{anchor:"transformers.BigBirdPegasusForCausalLM.forward.example",$$slots:{default:[bn]},$$scope:{ctx:B}}}),Xe=new tn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bigbird_pegasus.md"}}),{c(){t=h("meta"),p=i(),o=h("p"),r=i(),g=h("p"),g.innerHTML=n,c=i(),v=h("div"),v.innerHTML=lt,ue=i(),f(Q.$$.fragment),pt=i(),he=h("p"),he.innerHTML=wo,ut=i(),me=h("p"),me.innerHTML=To,ht=i(),f(K.$$.fragment),mt=i(),ge=h("p"),ge.innerHTML=Bo,gt=i(),f(ee.$$.fragment),ft=i(),fe=h("p"),fe.innerHTML=vo,_t=i(),_e=h("p"),_e.innerHTML=Jo,yt=i(),f(ye.$$.fragment),bt=i(),f(be.$$.fragment),Mt=i(),Me=h("ul"),Me.innerHTML=Uo,wt=i(),f(we.$$.fragment),Tt=i(),Te=h("p"),Te.innerHTML=ko,Bt=i(),f(Be.$$.fragment),vt=i(),C=h("div"),f(ve.$$.fragment),Vt=i(),Pe=h("p"),Pe.innerHTML=jo,qt=i(),Re=h("p"),Re.innerHTML=Go,Xt=i(),f(te.$$.fragment),Jt=i(),f(Je.$$.fragment),Ut=i(),J=h("div"),f(Ue.$$.fragment),Nt=i(),He=h("p"),He.textContent=Zo,Pt=i(),Ye=h("p"),Ye.innerHTML=Co,Rt=i(),Se=h("p"),Se.innerHTML=Io,Ht=i(),L=h("div"),f(ke.$$.fragment),Yt=i(),Qe=h("p"),Qe.innerHTML=Wo,St=i(),f(oe.$$.fragment),kt=i(),f(je.$$.fragment),jt=i(),U=h("div"),f(Ge.$$.fragment),Qt=i(),Le=h("p"),Le.textContent=$o,Lt=i(),Ee=h("p"),Ee.innerHTML=xo,Et=i(),Ae=h("p"),Ae.innerHTML=zo,At=i(),q=h("div"),f(Ze.$$.fragment),Ot=i(),Oe=h("p"),Oe.innerHTML=Fo,Dt=i(),f(ne.$$.fragment),Kt=i(),f(se.$$.fragment),Gt=i(),f(Ce.$$.fragment),Zt=i(),k=h("div"),f(Ie.$$.fragment),eo=i(),De=h("p"),De.textContent=Vo,to=i(),Ke=h("p"),Ke.innerHTML=qo,oo=i(),et=h("p"),et.innerHTML=Xo,no=i(),Z=h("div"),f(We.$$.fragment),so=i(),tt=h("p"),tt.innerHTML=No,ao=i(),f(ae.$$.fragment),ro=i(),f(re.$$.fragment),io=i(),f(ie.$$.fragment),Ct=i(),f($e.$$.fragment),It=i(),j=h("div"),f(xe.$$.fragment),lo=i(),ot=h("p"),ot.innerHTML=Po,co=i(),nt=h("p"),nt.innerHTML=Ro,po=i(),st=h("p"),st.innerHTML=Ho,uo=i(),X=h("div"),f(ze.$$.fragment),ho=i(),at=h("p"),at.innerHTML=Yo,mo=i(),f(de.$$.fragment),go=i(),f(le.$$.fragment),Wt=i(),f(Fe.$$.fragment),$t=i(),A=h("div"),f(Ve.$$.fragment),fo=i(),N=h("div"),f(qe.$$.fragment),_o=i(),rt=h("p"),rt.innerHTML=So,yo=i(),f(ce.$$.fragment),bo=i(),f(pe.$$.fragment),xt=i(),f(Xe.$$.fragment),zt=i(),ct=h("p"),this.h()},l(e){const s=Ko("svelte-u9bgzb",document.head);t=m(s,"META",{name:!0,content:!0}),s.forEach(a),p=d(e),o=m(e,"P",{}),F(o).forEach(a),r=d(e),g=m(e,"P",{"data-svelte-h":!0}),T(g)!=="svelte-j72416"&&(g.innerHTML=n),c=d(e),v=m(e,"DIV",{style:!0,"data-svelte-h":!0}),T(v)!=="svelte-383xsf"&&(v.innerHTML=lt),ue=d(e),_(Q.$$.fragment,e),pt=d(e),he=m(e,"P",{"data-svelte-h":!0}),T(he)!=="svelte-1cznzu0"&&(he.innerHTML=wo),ut=d(e),me=m(e,"P",{"data-svelte-h":!0}),T(me)!=="svelte-1awiq7d"&&(me.innerHTML=To),ht=d(e),_(K.$$.fragment,e),mt=d(e),ge=m(e,"P",{"data-svelte-h":!0}),T(ge)!=="svelte-1q65a0t"&&(ge.innerHTML=Bo),gt=d(e),_(ee.$$.fragment,e),ft=d(e),fe=m(e,"P",{"data-svelte-h":!0}),T(fe)!=="svelte-nf5ooi"&&(fe.innerHTML=vo),_t=d(e),_e=m(e,"P",{"data-svelte-h":!0}),T(_e)!=="svelte-11sw8fc"&&(_e.innerHTML=Jo),yt=d(e),_(ye.$$.fragment,e),bt=d(e),_(be.$$.fragment,e),Mt=d(e),Me=m(e,"UL",{"data-svelte-h":!0}),T(Me)!=="svelte-13eyhcs"&&(Me.innerHTML=Uo),wt=d(e),_(we.$$.fragment,e),Tt=d(e),Te=m(e,"P",{"data-svelte-h":!0}),T(Te)!=="svelte-1d2g2fv"&&(Te.innerHTML=ko),Bt=d(e),_(Be.$$.fragment,e),vt=d(e),C=m(e,"DIV",{class:!0});var P=F(C);_(ve.$$.fragment,P),Vt=d(P),Pe=m(P,"P",{"data-svelte-h":!0}),T(Pe)!=="svelte-s5ges1"&&(Pe.innerHTML=jo),qt=d(P),Re=m(P,"P",{"data-svelte-h":!0}),T(Re)!=="svelte-1ek1ss9"&&(Re.innerHTML=Go),Xt=d(P),_(te.$$.fragment,P),P.forEach(a),Jt=d(e),_(Je.$$.fragment,e),Ut=d(e),J=m(e,"DIV",{class:!0});var I=F(J);_(Ue.$$.fragment,I),Nt=d(I),He=m(I,"P",{"data-svelte-h":!0}),T(He)!=="svelte-qf9sfh"&&(He.textContent=Zo),Pt=d(I),Ye=m(I,"P",{"data-svelte-h":!0}),T(Ye)!=="svelte-q52n56"&&(Ye.innerHTML=Co),Rt=d(I),Se=m(I,"P",{"data-svelte-h":!0}),T(Se)!=="svelte-hswkmf"&&(Se.innerHTML=Io),Ht=d(I),L=m(I,"DIV",{class:!0});var O=F(L);_(ke.$$.fragment,O),Yt=d(O),Qe=m(O,"P",{"data-svelte-h":!0}),T(Qe)!=="svelte-6zdlb1"&&(Qe.innerHTML=Wo),St=d(O),_(oe.$$.fragment,O),O.forEach(a),I.forEach(a),kt=d(e),_(je.$$.fragment,e),jt=d(e),U=m(e,"DIV",{class:!0});var W=F(U);_(Ge.$$.fragment,W),Qt=d(W),Le=m(W,"P",{"data-svelte-h":!0}),T(Le)!=="svelte-1ceza4p"&&(Le.textContent=$o),Lt=d(W),Ee=m(W,"P",{"data-svelte-h":!0}),T(Ee)!=="svelte-q52n56"&&(Ee.innerHTML=xo),Et=d(W),Ae=m(W,"P",{"data-svelte-h":!0}),T(Ae)!=="svelte-hswkmf"&&(Ae.innerHTML=zo),At=d(W),q=m(W,"DIV",{class:!0});var R=F(q);_(Ze.$$.fragment,R),Ot=d(R),Oe=m(R,"P",{"data-svelte-h":!0}),T(Oe)!=="svelte-utzn3h"&&(Oe.innerHTML=Fo),Dt=d(R),_(ne.$$.fragment,R),Kt=d(R),_(se.$$.fragment,R),R.forEach(a),W.forEach(a),Gt=d(e),_(Ce.$$.fragment,e),Zt=d(e),k=m(e,"DIV",{class:!0});var $=F(k);_(Ie.$$.fragment,$),eo=d($),De=m($,"P",{"data-svelte-h":!0}),T(De)!=="svelte-1czdklt"&&(De.textContent=Vo),to=d($),Ke=m($,"P",{"data-svelte-h":!0}),T(Ke)!=="svelte-q52n56"&&(Ke.innerHTML=qo),oo=d($),et=m($,"P",{"data-svelte-h":!0}),T(et)!=="svelte-hswkmf"&&(et.innerHTML=Xo),no=d($),Z=m($,"DIV",{class:!0});var x=F(Z);_(We.$$.fragment,x),so=d(x),tt=m(x,"P",{"data-svelte-h":!0}),T(tt)!=="svelte-1gg07mp"&&(tt.innerHTML=No),ao=d(x),_(ae.$$.fragment,x),ro=d(x),_(re.$$.fragment,x),io=d(x),_(ie.$$.fragment,x),x.forEach(a),$.forEach(a),Ct=d(e),_($e.$$.fragment,e),It=d(e),j=m(e,"DIV",{class:!0});var z=F(j);_(xe.$$.fragment,z),lo=d(z),ot=m(z,"P",{"data-svelte-h":!0}),T(ot)!=="svelte-1qjy740"&&(ot.innerHTML=Po),co=d(z),nt=m(z,"P",{"data-svelte-h":!0}),T(nt)!=="svelte-q52n56"&&(nt.innerHTML=Ro),po=d(z),st=m(z,"P",{"data-svelte-h":!0}),T(st)!=="svelte-hswkmf"&&(st.innerHTML=Ho),uo=d(z),X=m(z,"DIV",{class:!0});var H=F(X);_(ze.$$.fragment,H),ho=d(H),at=m(H,"P",{"data-svelte-h":!0}),T(at)!=="svelte-1lwcor1"&&(at.innerHTML=Yo),mo=d(H),_(de.$$.fragment,H),go=d(H),_(le.$$.fragment,H),H.forEach(a),z.forEach(a),Wt=d(e),_(Fe.$$.fragment,e),$t=d(e),A=m(e,"DIV",{class:!0});var Ne=F(A);_(Ve.$$.fragment,Ne),fo=d(Ne),N=m(Ne,"DIV",{class:!0});var Y=F(N);_(qe.$$.fragment,Y),_o=d(Y),rt=m(Y,"P",{"data-svelte-h":!0}),T(rt)!=="svelte-vqpj5p"&&(rt.innerHTML=So),yo=d(Y),_(ce.$$.fragment,Y),bo=d(Y),_(pe.$$.fragment,Y),Y.forEach(a),Ne.forEach(a),xt=d(e),_(Xe.$$.fragment,e),zt=d(e),ct=m(e,"P",{}),F(ct).forEach(a),this.h()},h(){V(t,"name","hf:doc:metadata"),V(t,"content",wn),en(v,"float","right"),V(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){u(document.head,t),l(e,p,s),l(e,o,s),l(e,r,s),l(e,g,s),l(e,c,s),l(e,v,s),l(e,ue,s),y(Q,e,s),l(e,pt,s),l(e,he,s),l(e,ut,s),l(e,me,s),l(e,ht,s),y(K,e,s),l(e,mt,s),l(e,ge,s),l(e,gt,s),y(ee,e,s),l(e,ft,s),l(e,fe,s),l(e,_t,s),l(e,_e,s),l(e,yt,s),y(ye,e,s),l(e,bt,s),y(be,e,s),l(e,Mt,s),l(e,Me,s),l(e,wt,s),y(we,e,s),l(e,Tt,s),l(e,Te,s),l(e,Bt,s),y(Be,e,s),l(e,vt,s),l(e,C,s),y(ve,C,null),u(C,Vt),u(C,Pe),u(C,qt),u(C,Re),u(C,Xt),y(te,C,null),l(e,Jt,s),y(Je,e,s),l(e,Ut,s),l(e,J,s),y(Ue,J,null),u(J,Nt),u(J,He),u(J,Pt),u(J,Ye),u(J,Rt),u(J,Se),u(J,Ht),u(J,L),y(ke,L,null),u(L,Yt),u(L,Qe),u(L,St),y(oe,L,null),l(e,kt,s),y(je,e,s),l(e,jt,s),l(e,U,s),y(Ge,U,null),u(U,Qt),u(U,Le),u(U,Lt),u(U,Ee),u(U,Et),u(U,Ae),u(U,At),u(U,q),y(Ze,q,null),u(q,Ot),u(q,Oe),u(q,Dt),y(ne,q,null),u(q,Kt),y(se,q,null),l(e,Gt,s),y(Ce,e,s),l(e,Zt,s),l(e,k,s),y(Ie,k,null),u(k,eo),u(k,De),u(k,to),u(k,Ke),u(k,oo),u(k,et),u(k,no),u(k,Z),y(We,Z,null),u(Z,so),u(Z,tt),u(Z,ao),y(ae,Z,null),u(Z,ro),y(re,Z,null),u(Z,io),y(ie,Z,null),l(e,Ct,s),y($e,e,s),l(e,It,s),l(e,j,s),y(xe,j,null),u(j,lo),u(j,ot),u(j,co),u(j,nt),u(j,po),u(j,st),u(j,uo),u(j,X),y(ze,X,null),u(X,ho),u(X,at),u(X,mo),y(de,X,null),u(X,go),y(le,X,null),l(e,Wt,s),y(Fe,e,s),l(e,$t,s),l(e,A,s),y(Ve,A,null),u(A,fo),u(A,N),y(qe,N,null),u(N,_o),u(N,rt),u(N,yo),y(ce,N,null),u(N,bo),y(pe,N,null),l(e,xt,s),y(Xe,e,s),l(e,zt,s),l(e,ct,s),Ft=!0},p(e,[s]){const P={};s&2&&(P.$$scope={dirty:s,ctx:e}),K.$set(P);const I={};s&2&&(I.$$scope={dirty:s,ctx:e}),ee.$set(I);const O={};s&2&&(O.$$scope={dirty:s,ctx:e}),te.$set(O);const W={};s&2&&(W.$$scope={dirty:s,ctx:e}),oe.$set(W);const R={};s&2&&(R.$$scope={dirty:s,ctx:e}),ne.$set(R);const $={};s&2&&($.$$scope={dirty:s,ctx:e}),se.$set($);const x={};s&2&&(x.$$scope={dirty:s,ctx:e}),ae.$set(x);const z={};s&2&&(z.$$scope={dirty:s,ctx:e}),re.$set(z);const H={};s&2&&(H.$$scope={dirty:s,ctx:e}),ie.$set(H);const Ne={};s&2&&(Ne.$$scope={dirty:s,ctx:e}),de.$set(Ne);const Y={};s&2&&(Y.$$scope={dirty:s,ctx:e}),le.$set(Y);const Qo={};s&2&&(Qo.$$scope={dirty:s,ctx:e}),ce.$set(Qo);const Lo={};s&2&&(Lo.$$scope={dirty:s,ctx:e}),pe.$set(Lo)},i(e){Ft||(b(Q.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(ye.$$.fragment,e),b(be.$$.fragment,e),b(we.$$.fragment,e),b(Be.$$.fragment,e),b(ve.$$.fragment,e),b(te.$$.fragment,e),b(Je.$$.fragment,e),b(Ue.$$.fragment,e),b(ke.$$.fragment,e),b(oe.$$.fragment,e),b(je.$$.fragment,e),b(Ge.$$.fragment,e),b(Ze.$$.fragment,e),b(ne.$$.fragment,e),b(se.$$.fragment,e),b(Ce.$$.fragment,e),b(Ie.$$.fragment,e),b(We.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(ie.$$.fragment,e),b($e.$$.fragment,e),b(xe.$$.fragment,e),b(ze.$$.fragment,e),b(de.$$.fragment,e),b(le.$$.fragment,e),b(Fe.$$.fragment,e),b(Ve.$$.fragment,e),b(qe.$$.fragment,e),b(ce.$$.fragment,e),b(pe.$$.fragment,e),b(Xe.$$.fragment,e),Ft=!0)},o(e){M(Q.$$.fragment,e),M(K.$$.fragment,e),M(ee.$$.fragment,e),M(ye.$$.fragment,e),M(be.$$.fragment,e),M(we.$$.fragment,e),M(Be.$$.fragment,e),M(ve.$$.fragment,e),M(te.$$.fragment,e),M(Je.$$.fragment,e),M(Ue.$$.fragment,e),M(ke.$$.fragment,e),M(oe.$$.fragment,e),M(je.$$.fragment,e),M(Ge.$$.fragment,e),M(Ze.$$.fragment,e),M(ne.$$.fragment,e),M(se.$$.fragment,e),M(Ce.$$.fragment,e),M(Ie.$$.fragment,e),M(We.$$.fragment,e),M(ae.$$.fragment,e),M(re.$$.fragment,e),M(ie.$$.fragment,e),M($e.$$.fragment,e),M(xe.$$.fragment,e),M(ze.$$.fragment,e),M(de.$$.fragment,e),M(le.$$.fragment,e),M(Fe.$$.fragment,e),M(Ve.$$.fragment,e),M(qe.$$.fragment,e),M(ce.$$.fragment,e),M(pe.$$.fragment,e),M(Xe.$$.fragment,e),Ft=!1},d(e){e&&(a(p),a(o),a(r),a(g),a(c),a(v),a(ue),a(pt),a(he),a(ut),a(me),a(ht),a(mt),a(ge),a(gt),a(ft),a(fe),a(_t),a(_e),a(yt),a(bt),a(Mt),a(Me),a(wt),a(Tt),a(Te),a(Bt),a(vt),a(C),a(Jt),a(Ut),a(J),a(kt),a(jt),a(U),a(Gt),a(Zt),a(k),a(Ct),a(It),a(j),a(Wt),a($t),a(A),a(xt),a(zt),a(ct)),a(t),w(Q,e),w(K,e),w(ee,e),w(ye,e),w(be,e),w(we,e),w(Be,e),w(ve),w(te),w(Je,e),w(Ue),w(ke),w(oe),w(je,e),w(Ge),w(Ze),w(ne),w(se),w(Ce,e),w(Ie),w(We),w(ae),w(re),w(ie),w($e,e),w(xe),w(ze),w(de),w(le),w(Fe,e),w(Ve),w(qe),w(ce),w(pe),w(Xe,e)}}}const wn='{"title":"BigBirdPegasus","local":"bigbirdpegasus","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"BigBirdPegasusConfig","local":"transformers.BigBirdPegasusConfig","sections":[],"depth":2},{"title":"BigBirdPegasusModel","local":"transformers.BigBirdPegasusModel","sections":[],"depth":2},{"title":"BigBirdPegasusForConditionalGeneration","local":"transformers.BigBirdPegasusForConditionalGeneration","sections":[],"depth":2},{"title":"BigBirdPegasusForSequenceClassification","local":"transformers.BigBirdPegasusForSequenceClassification","sections":[],"depth":2},{"title":"BigBirdPegasusForQuestionAnswering","local":"transformers.BigBirdPegasusForQuestionAnswering","sections":[],"depth":2},{"title":"BigBirdPegasusForCausalLM","local":"transformers.BigBirdPegasusForCausalLM","sections":[],"depth":2}],"depth":1}';function Tn(B){return Ao(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Cn extends Oo{constructor(t){super(),Do(this,t,Tn,Mn,Eo,{})}}export{Cn as component};
