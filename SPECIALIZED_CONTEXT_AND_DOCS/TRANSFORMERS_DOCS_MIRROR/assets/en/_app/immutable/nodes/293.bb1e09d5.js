import{s as ba,o as ya,n as P}from"../chunks/scheduler.18a86fab.js";import{S as wa,i as va,g as c,s as a,r as h,A as ka,h as d,f as l,c as s,j as $,x as T,u,k as x,y as n,a as p,v as g,d as f,t as _,w as M}from"../chunks/index.98837b22.js";import{T as gt}from"../chunks/Tip.77304350.js";import{D as I}from"../chunks/Docstring.a1ef7999.js";import{C as re}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ft}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as V,E as xa}from"../chunks/getInferenceSnippets.06c2775f.js";function $a(v){let t,b="Mllama has an extra token used as a placeholder for image positions in the text. It means that input ids and an input embedding layer will have an extra token. But since the weights for input and output embeddings are not tied, the <code>lm_head</code> layer has one less token and will fail if you want to calculate loss on image tokens or apply some logit processors. In case you are training, make sure to mask out special <code>&quot;&lt;|image|&gt;&quot;</code> tokens in the <code>labels</code> as the model should not be trained on predicting them.",i,m,y="Otherwise if you see CUDA-side index errors when generating, use the below code to expand the <code>lm_head</code> by one more token.",r,w,H;return w=new re({props:{code:"b2xkX2VtYmVkZGluZ3MlMjAlM0QlMjBtb2RlbC5nZXRfb3V0cHV0X2VtYmVkZGluZ3MoKSUwQSUwQW51bV90b2tlbnMlMjAlM0QlMjBtb2RlbC52b2NhYl9zaXplJTIwJTJCJTIwMSUwQXJlc2l6ZWRfZW1iZWRkaW5ncyUyMCUzRCUyMG1vZGVsLl9nZXRfcmVzaXplZF9sbV9oZWFkKG9sZF9lbWJlZGRpbmdzJTJDJTIwbmV3X251bV90b2tlbnMlM0RudW1fdG9rZW5zJTJDJTIwbWVhbl9yZXNpemluZyUzRFRydWUpJTBBcmVzaXplZF9lbWJlZGRpbmdzLnJlcXVpcmVzX2dyYWRfKG9sZF9lbWJlZGRpbmdzLndlaWdodC5yZXF1aXJlc19ncmFkKSUwQW1vZGVsLnNldF9vdXRwdXRfZW1iZWRkaW5ncyhyZXNpemVkX2VtYmVkZGluZ3Mp",highlighted:`old_embeddings = model.get_output_embeddings()

num_tokens = model.vocab_size + <span class="hljs-number">1</span>
resized_embeddings = model._get_resized_lm_head(old_embeddings, new_num_tokens=num_tokens, mean_resizing=<span class="hljs-literal">True</span>)
resized_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
model.set_output_embeddings(resized_embeddings)`,wrap:!1}}),{c(){t=c("p"),t.innerHTML=b,i=a(),m=c("p"),m.innerHTML=y,r=a(),h(w.$$.fragment)},l(k){t=d(k,"P",{"data-svelte-h":!0}),T(t)!=="svelte-k28zha"&&(t.innerHTML=b),i=s(k),m=d(k,"P",{"data-svelte-h":!0}),T(m)!=="svelte-10oofbh"&&(m.innerHTML=y),r=s(k),u(w.$$.fragment,k)},m(k,Z){p(k,t,Z),p(k,i,Z),p(k,m,Z),p(k,r,Z),g(w,k,Z),H=!0},p:P,i(k){H||(f(w.$$.fragment,k),H=!0)},o(k){_(w.$$.fragment,k),H=!1},d(k){k&&(l(t),l(i),l(m),l(r)),M(w,k)}}}function Ia(v){let t,b="Example:",i,m,y;return m=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1sbGFtYUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyME1sbGFtYUNvbmZpZyUyQyUyME1sbGFtYVZpc2lvbkNvbmZpZyUyQyUyME1sbGFtYVRleHRDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwQ0xJUC12aXNpb24lMjBjb25maWclMEF2aXNpb25fY29uZmlnJTIwJTNEJTIwTWxsYW1hVmlzaW9uQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTGxhbWElMjBjb25maWclMEF0ZXh0X2NvbmZpZyUyMCUzRCUyME1sbGFtYVRleHRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtbGxhbWEtMTFiJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME1sbGFtYUNvbmZpZyh2aXNpb25fY29uZmlnJTJDJTIwdGV4dF9jb25maWcpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMG1sbGFtYS0xMWIlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyME1sbGFtYUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbihjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MllamaForConditionalGeneration, MllamaConfig, MllamaVisionConfig, MllamaTextConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a CLIP-vision config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>vision_config = MllamaVisionConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Llama config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>text_config = MllamaTextConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a mllama-11b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MllamaConfig(vision_config, text_config)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the mllama-11b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MllamaForConditionalGeneration(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=a(),h(m.$$.fragment)},l(r){t=d(r,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=b),i=s(r),u(m.$$.fragment,r)},m(r,w){p(r,t,w),p(r,i,w),g(m,r,w),y=!0},p:P,i(r){y||(f(m.$$.fragment,r),y=!0)},o(r){_(m.$$.fragment,r),y=!1},d(r){r&&(l(t),l(i)),M(m,r)}}}function Ca(v){let t,b;return t=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1sbGFtYVByb2Nlc3NvciUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQSUwQXByb2Nlc3NvciUyMCUzRCUyME1sbGFtYVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1sbGFtYSUyRkxsYW1hLTMuMi0xMUItVmlzaW9uJTIyKSUwQSUwQXByb2Nlc3NvciglMEElMjAlMjAlMjAlMjBpbWFnZXMlM0R5b3VyX3BpbF9pbWFnZSUyQyUwQSUyMCUyMCUyMCUyMHRleHQlM0QlNUIlMjIlM0MlN0NpbWFnZSU3QyUzRUlmJTIwSSUyMGhhZCUyMHRvJTIwd3JpdGUlMjBhJTIwaGFpa3UlMjBmb3IlMjB0aGlzJTIwb25lJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwaW1hZ2VzX2t3YXJncyUyMCUzRCUyMCU3QiUyMnNpemUlMjIlM0ElMjAlN0IlMjJoZWlnaHQlMjIlM0ElMjA0NDglMkMlMjAlMjJ3aWR0aCUyMiUzQSUyMDQ0OCU3RCU3RCUyQyUwQSUyMCUyMCUyMCUyMHRleHRfa3dhcmdzJTIwJTNEJTIwJTdCJTIycGFkZGluZyUyMiUzQSUyMCUyMnJpZ2h0JTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwY29tbW9uX2t3YXJncyUyMCUzRCUyMCU3QiUyMnJldHVybl90ZW5zb3JzJTIyJTNBJTIwJTIycHQlMjIlN0QlMkMlMEEp",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MllamaProcessor
<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

processor = MllamaProcessor.from_pretrained(<span class="hljs-string">&quot;meta-llama/Llama-3.2-11B-Vision&quot;</span>)

processor(
    images=your_pil_image,
    text=[<span class="hljs-string">&quot;&lt;|image|&gt;If I had to write a haiku for this one&quot;</span>],
    images_kwargs = {<span class="hljs-string">&quot;size&quot;</span>: {<span class="hljs-string">&quot;height&quot;</span>: <span class="hljs-number">448</span>, <span class="hljs-string">&quot;width&quot;</span>: <span class="hljs-number">448</span>}},
    text_kwargs = {<span class="hljs-string">&quot;padding&quot;</span>: <span class="hljs-string">&quot;right&quot;</span>},
    common_kwargs = {<span class="hljs-string">&quot;return_tensors&quot;</span>: <span class="hljs-string">&quot;pt&quot;</span>},
)`,wrap:!1}}),{c(){h(t.$$.fragment)},l(i){u(t.$$.fragment,i)},m(i,m){g(t,i,m),b=!0},p:P,i(i){b||(f(t.$$.fragment,i),b=!0)},o(i){_(t.$$.fragment,i),b=!1},d(i){M(t,i)}}}function Ja(v){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=d(i,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,m){p(i,t,m)},p:P,d(i){i&&l(t)}}}function Ua(v){let t,b="Example:",i,m,y;return m=new re({props:{code:"ZnJvbSUyMFBJTCUyMGltcG9ydCUyMEltYWdlJTBBaW1wb3J0JTIwcmVxdWVzdHMlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyME1sbGFtYUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQWNoZWNrcG9pbnQlMjAlM0QlMjAlMjJtZXRhLWxsYW1hJTJGTGxhbWEtMy4yLTExQi1WaXNpb24lMjIlMEFtb2RlbCUyMCUzRCUyME1sbGFtYUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoY2hlY2twb2ludCklMEFwcm9jZXNzb3IlMjAlM0QlMjBBdXRvUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZChjaGVja3BvaW50KSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMiUzQyU3Q2ltYWdlJTdDJTNFSWYlMjBJJTIwaGFkJTIwdG8lMjB3cml0ZSUyMGElMjBoYWlrdSUyMGZvciUyMHRoaXMlMjBvbmUlMjIlMEF1cmwlMjAlM0QlMjAlMjJodHRwcyUzQSUyRiUyRnd3dy5pbGFua2VsbWFuLm9yZyUyRnN0b3BzaWducyUyRmF1c3RyYWxpYS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEElMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IodGV4dCUzRHByb21wdCUyQyUyMGltYWdlcyUzRGltYWdlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBHZW5lcmF0ZSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QxNSklMEElMEFwcm9tcHRfbGVuJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcy5zaGFwZSU1Qi0xJTVEJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMG91dHB1dCU1QiUzQSUyQyUyMHByb21wdF9sZW4lM0ElNUQlMEFnZW5lcmF0ZWRfdGV4dCUyMCUzRCUyMHByb2Nlc3Nvci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSUwQXByaW50KGdlbmVyYXRlZF90ZXh0KQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MllamaForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>checkpoint = <span class="hljs-string">&quot;meta-llama/Llama-3.2-11B-Vision&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(checkpoint)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;&lt;|image|&gt;If I had to write a haiku for this one&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;https://www.ilankelman.org/stopsigns/australia.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(text=prompt, images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>output = model.generate(**inputs, max_new_tokens=<span class="hljs-number">15</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt_len = inputs.input_ids.shape[-<span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = output[:, prompt_len:]
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_text = processor.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(generated_text)
[<span class="hljs-string">&#x27;, it would be:.\\\\nA stop sign in Chinatown.\\\\n&#x27;</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=a(),h(m.$$.fragment)},l(r){t=d(r,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=b),i=s(r),u(m.$$.fragment,r)},m(r,w){p(r,t,w),p(r,i,w),g(m,r,w),y=!0},p:P,i(r){y||(f(m.$$.fragment,r),y=!0)},o(r){_(m.$$.fragment,r),y=!1},d(r){r&&(l(t),l(i)),M(m,r)}}}function za(v){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=d(i,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,m){p(i,t,m)},p:P,d(i){i&&l(t)}}}function ja(v){let t,b="Example:",i,m,y;return m=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNbGxhbWFGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwTWxsYW1hRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMkxsYW1hLTMuMi0xMUItVmlzaW9uJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkxsYW1hLTMuMi0xMUItVmlzaW9uJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMklmJTIwSSUyMGhhZCUyMHRvJTIwd3JpdGUlMjBhJTIwaGFpa3UlMkMlMjBpdCUyMHdvdWxkJTIwYmUlM0ElMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNENDAlMkMlMjBkb19zYW1wbGUlM0RUcnVlJTJDJTIwdGVtcGVyYXR1cmUlM0QwLjYpJTBBcmVzdWx0JTIwJTNEJTIwdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVEJTBBcHJpbnQocmVzdWx0KQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MllamaForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MllamaForCausalLM.from_pretrained(<span class="hljs-string">&quot;Llama-3.2-11B-Vision&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Llama-3.2-11B-Vision&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;If I had to write a haiku, it would be:&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">40</span>, do_sample=<span class="hljs-literal">True</span>, temperature=<span class="hljs-number">0.6</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>result = tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(result)
If I had to write a haiku, it would be: <span class="hljs-string">&quot;Snowflakes gently fall&quot;</span> - simple, yet peaceful.
I love the idea of snowflakes gently falling, each one`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=a(),h(m.$$.fragment)},l(r){t=d(r,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=b),i=s(r),u(m.$$.fragment,r)},m(r,w){p(r,t,w),p(r,i,w),g(m,r,w),y=!0},p:P,i(r){y||(f(m.$$.fragment,r),y=!0)},o(r){_(m.$$.fragment,r),y=!1},d(r){r&&(l(t),l(i)),M(m,r)}}}function Fa(v){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=d(i,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,m){p(i,t,m)},p:P,d(i){i&&l(t)}}}function Wa(v){let t,b="Example:",i,m,y;return m=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBNbGxhbWFUZXh0TW9kZWwlMEElMEFjaGVja3BvaW50JTIwJTNEJTIwJTIybWV0YS1sbGFtYSUyRkxsYW1hLTMuMi0xMUItVmlzaW9uJTIyJTBBbW9kZWwlMjAlM0QlMjBNbGxhbWFUZXh0TW9kZWwuZnJvbV9wcmV0cmFpbmVkKGNoZWNrcG9pbnQpJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoY2hlY2twb2ludCklMEElMEF0ZXh0JTIwJTNEJTIwJTIyJTNDJTdDaW1hZ2UlN0MlM0VJZiUyMEklMjBoYWQlMjB0byUyMHdyaXRlJTIwYSUyMGhhaWt1JTIwZm9yJTIwdGhpcyUyMG9uZSUyMiUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3Nvcih0ZXh0JTNEdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBcHJpbnQob3V0cHV0Lmxhc3RfaGlkZGVuX3N0YXRlLnNoYXBlKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MllamaTextModel

<span class="hljs-meta">&gt;&gt;&gt; </span>checkpoint = <span class="hljs-string">&quot;meta-llama/Llama-3.2-11B-Vision&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MllamaTextModel.from_pretrained(checkpoint)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(checkpoint)

<span class="hljs-meta">&gt;&gt;&gt; </span>text = <span class="hljs-string">&quot;&lt;|image|&gt;If I had to write a haiku for this one&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(text=text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>output = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(output.last_hidden_state.shape)
torch.Size([<span class="hljs-number">1</span>, <span class="hljs-number">13</span>, <span class="hljs-number">4096</span>])`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=a(),h(m.$$.fragment)},l(r){t=d(r,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=b),i=s(r),u(m.$$.fragment,r)},m(r,w){p(r,t,w),p(r,i,w),g(m,r,w),y=!0},p:P,i(r){y||(f(m.$$.fragment,r),y=!0)},o(r){_(m.$$.fragment,r),y=!1},d(r){r&&(l(t),l(i)),M(m,r)}}}function La(v){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=d(i,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,m){p(i,t,m)},p:P,d(i){i&&l(t)}}}function Va(v){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=d(i,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,m){p(i,t,m)},p:P,d(i){i&&l(t)}}}function Za(v){let t,b="Example:",i,m,y;return m=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNbGxhbWFGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwTWxsYW1hRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMkxsYW1hLTMuMi0xMUItVmlzaW9uJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkxsYW1hLTMuMi0xMUItVmlzaW9uJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMklmJTIwSSUyMGhhZCUyMHRvJTIwd3JpdGUlMjBhJTIwaGFpa3UlMkMlMjBpdCUyMHdvdWxkJTIwYmUlM0ElMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNENDAlMkMlMjBkb19zYW1wbGUlM0RUcnVlJTJDJTIwdGVtcGVyYXR1cmUlM0QwLjYpJTBBcmVzdWx0JTIwJTNEJTIwdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVEJTBBcHJpbnQocmVzdWx0KQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MllamaForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MllamaForCausalLM.from_pretrained(<span class="hljs-string">&quot;Llama-3.2-11B-Vision&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Llama-3.2-11B-Vision&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;If I had to write a haiku, it would be:&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">40</span>, do_sample=<span class="hljs-literal">True</span>, temperature=<span class="hljs-number">0.6</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>result = tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(result)
If I had to write a haiku, it would be: <span class="hljs-string">&quot;Snowflakes gently fall&quot;</span> - simple, yet peaceful.
I love the idea of snowflakes gently falling, each one`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=a(),h(m.$$.fragment)},l(r){t=d(r,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=b),i=s(r),u(m.$$.fragment,r)},m(r,w){p(r,t,w),p(r,i,w),g(m,r,w),y=!0},p:P,i(r){y||(f(m.$$.fragment,r),y=!0)},o(r){_(m.$$.fragment,r),y=!1},d(r){r&&(l(t),l(i)),M(m,r)}}}function Pa(v){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=b},l(i){t=d(i,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(i,m){p(i,t,m)},p:P,d(i){i&&l(t)}}}function Ga(v){let t,b="Example:",i,m,y;return m=new re({props:{code:"ZnJvbSUyMFBJTCUyMGltcG9ydCUyMEltYWdlJTBBaW1wb3J0JTIwcmVxdWVzdHMlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyME1sbGFtYVZpc2lvbk1vZGVsJTBBJTBBY2hlY2twb2ludCUyMCUzRCUyMCUyMm1ldGEtbGxhbWElMkZMbGFtYS0zLjItMTFCLVZpc2lvbiUyMiUwQW1vZGVsJTIwJTNEJTIwTWxsYW1hVmlzaW9uTW9kZWwuZnJvbV9wcmV0cmFpbmVkKGNoZWNrcG9pbnQpJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoY2hlY2twb2ludCklMEElMEF1cmwlMjAlM0QlMjAlMjJodHRwcyUzQSUyRiUyRnd3dy5pbGFua2VsbWFuLm9yZyUyRnN0b3BzaWducyUyRmF1c3RyYWxpYS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IoaW1hZ2VzJTNEaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQXByaW50KG91dHB1dC5sYXN0X2hpZGRlbl9zdGF0ZS5zaGFwZSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MllamaVisionModel

<span class="hljs-meta">&gt;&gt;&gt; </span>checkpoint = <span class="hljs-string">&quot;meta-llama/Llama-3.2-11B-Vision&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MllamaVisionModel.from_pretrained(checkpoint)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(checkpoint)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;https://www.ilankelman.org/stopsigns/australia.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>output = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(output.last_hidden_state.shape)
torch.Size([<span class="hljs-number">1</span>, <span class="hljs-number">1</span>, <span class="hljs-number">4</span>, <span class="hljs-number">1025</span>, <span class="hljs-number">7680</span>])`,wrap:!1}}),{c(){t=c("p"),t.textContent=b,i=a(),h(m.$$.fragment)},l(r){t=d(r,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=b),i=s(r),u(m.$$.fragment,r)},m(r,w){p(r,t,w),p(r,i,w),g(m,r,w),y=!0},p:P,i(r){y||(f(m.$$.fragment,r),y=!0)},o(r){_(m.$$.fragment,r),y=!1},d(r){r&&(l(t),l(i)),M(m,r)}}}function Ba(v){let t,b,i,m,y,r="<em>This model was released on 2024-09-25 and added to Hugging Face Transformers on 2024-09-25.</em>",w,H,k,Z,Vn='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Kt,Ie,eo,Ce,Zn='The <a href="https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/" rel="nofollow">Llama 3.2-Vision</a> collection of multimodal large language models (LLMs) is a collection of pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text + images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image.',to,Je,Pn="<strong>Model Architecture:</strong> Llama 3.2-Vision is built on top of Llama 3.1 text-only model, which is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. To support image recognition tasks, the Llama 3.2-Vision model uses a separately trained vision adapter that integrates with the pre-trained Llama 3.1 language model. The adapter consists of a series of cross-attention layers that feed image encoder representations into the core LLM.",oo,Ue,no,ze,Gn="<li>For image+text and text inputs use <code>MllamaForConditionalGeneration</code>.</li> <li>For text-only inputs use <code>MllamaForCausalLM</code> for generation to avoid loading vision tower.</li> <li>Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images across samples and to a maximum number of tiles within each image.</li> <li>The text passed to the processor should have the <code>&quot;&lt;|image|&gt;&quot;</code> tokens where the images should be inserted.</li> <li>The processor has its own <code>apply_chat_template</code> method to convert chat messages to text that can then be passed as text to the processor. If you’re using <code>transformers&gt;=4.49.0</code>, you can also get a vectorized output from <code>apply_chat_template</code>. See the <strong>Usage Examples</strong> below for more details on how to use it.</li>",ao,ie,so,je,ro,Fe,lo,We,io,Le,co,Ve,mo,Ze,po,C,Pe,jo,_t,Bn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForConditionalGeneration">MllamaForConditionalGeneration</a>. It is used to instantiate an
Mllama model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Mllama-9B.`,Fo,Mt,Nn='e.g. <a href="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision" rel="nofollow">meta-llama/Llama-3.2-11B-Vision</a>',Wo,Tt,qn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Lo,ce,ho,Ge,uo,G,Be,Vo,bt,Rn=`Constructs a Mllama processor which wraps <a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor">MllamaImageProcessor</a> and
<code>PretrainedTokenizerFast</code> into a single processor that inherits both the image processor and
tokenizer functionalities. See the <code>__call__()</code> and <a href="/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode">decode()</a> for more
information.
The preferred way of passing kwargs is as a dictionary per modality, see usage example below.`,Zo,de,Po,me,Ne,Go,yt,Xn="Post-process the output of the model to decode the text.",go,qe,fo,J,Re,Bo,wt,En="Constructs a Mllama image processor.",No,pe,Xe,qo,vt,Hn=`Pad an image to the <code>size</code> x <code>aspect_ratio</code>. For example, if size is {height: 224, width: 224} and aspect ratio is
(1, 2), the image will be padded to 224x448.`,Ro,he,Ee,Xo,kt,Qn="Preprocess a batch of images.",Eo,oe,He,Ho,xt,Yn=`Resizes an image to fit within a tiled canvas while maintaining its aspect ratio.
The optimal canvas size is calculated based on the maximum number of tiles and the tile size.`,Qo,$t,Sn=`The function first determines the best tile arrangement for the image, then resizes the image
to fit within this canvas. The resized image and the number of tiles along the height and width
dimensions are returned.`,_o,Qe,Mo,U,Ye,Yo,It,Dn="The Mllama model which consists of a vision encoder and a language model.",So,Ct,An=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Do,Jt,On=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ao,Q,Se,Oo,Ut,Kn='The <a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForConditionalGeneration">MllamaForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Ko,ue,en,ge,To,De,bo,z,Ae,tn,zt,ea="The Mllama Text Model with a language modeling head on top.",on,jt,ta=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,nn,Ft,oa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,an,Y,Oe,sn,Wt,na='The <a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM">MllamaForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',rn,fe,ln,_e,yo,Ke,wo,j,et,cn,Lt,aa="The Mllama Text Model which consists of transformer with self and cross attention layers.",dn,Vt,sa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,mn,Zt,ra=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,pn,S,tt,hn,Pt,la='The <a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaTextModel">MllamaTextModel</a> forward method, overrides the <code>__call__</code> special method.',un,Me,gn,Te,vo,ot,ko,F,nt,fn,Gt,ia="The Mllama model which consists of a vision encoder and a language model without language modeling head.",_n,Bt,ca=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Mn,Nt,da=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Tn,ne,at,bn,qt,ma='The <a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaModel">MllamaModel</a> forward method, overrides the <code>__call__</code> special method.',yn,be,xo,st,$o,W,rt,wn,Rt,pa="The Mllama Text Model with a language modeling head on top.",vn,Xt,ha=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,kn,Et,ua=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xn,D,lt,$n,Ht,ga='The <a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM">MllamaForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',In,ye,Cn,we,Io,it,Co,L,ct,Jn,Qt,fa="The Mllama Vision Model which consists of two vision encoders.",Un,Yt,_a=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zn,St,Ma=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,jn,A,dt,Fn,Dt,Ta='The <a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaVisionModel">MllamaVisionModel</a> forward method, overrides the <code>__call__</code> special method.',Wn,ve,Ln,ke,Jo,mt,Uo,Ot,zo;return H=new V({props:{title:"Mllama",local:"mllama",headingTag:"h1"}}),Ie=new V({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ue=new V({props:{title:"Usage Tips",local:"usage-tips",headingTag:"h2"}}),ie=new gt({props:{warning:!0,$$slots:{default:[$a]},$$scope:{ctx:v}}}),je=new V({props:{title:"Usage Example",local:"usage-example",headingTag:"h2"}}),Fe=new V({props:{title:"Instruct model",local:"instruct-model",headingTag:"h4"}}),We=new re({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwTWxsYW1hRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTJDJTIwQXV0b1Byb2Nlc3NvciUwQSUwQW1vZGVsX2lkJTIwJTNEJTIwJTIybWV0YS1sbGFtYSUyRkxsYW1hLTMuMi0xMUItVmlzaW9uLUluc3RydWN0JTIyJTBBbW9kZWwlMjAlM0QlMjBNbGxhbWFGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2KSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkKSUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTJDJTIwJTIydXJsJTIyJTNBJTIwJTIyaHR0cHMlM0ElMkYlMkZsbGF2YS12bC5naXRodWIuaW8lMkZzdGF0aWMlMkZpbWFnZXMlMkZ2aWV3LmpwZyUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMldoYXQlMjBkb2VzJTIwdGhlJTIwaW1hZ2UlMjBzaG93JTNGJTIyJTdEJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTBBJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTVEJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUobWVzc2FnZXMlMkMlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTJDJTIwdG9rZW5pemUlM0RUcnVlJTJDJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QyNSklMEFwcmludChwcm9jZXNzb3IuZGVjb2RlKG91dHB1dCU1QjAlNUQpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MllamaForConditionalGeneration, AutoProcessor

model_id = <span class="hljs-string">&quot;meta-llama/Llama-3.2-11B-Vision-Instruct&quot;</span>
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map=<span class="hljs-string">&quot;auto&quot;</span>, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    [
        {
            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, 
            <span class="hljs-string">&quot;content&quot;</span>: [
                {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://llava-vl.github.io/static/images/view.jpg&quot;</span>},
                {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;What does the image show?&quot;</span>}
            ]
        }
    ],
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
output = model.generate(**inputs, max_new_tokens=<span class="hljs-number">25</span>)
<span class="hljs-built_in">print</span>(processor.decode(output[<span class="hljs-number">0</span>]))`,wrap:!1}}),Le=new V({props:{title:"Base model",local:"base-model",headingTag:"h4"}}),Ve=new re({props:{code:"aW1wb3J0JTIwcmVxdWVzdHMlMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBNbGxhbWFGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMkMlMjBBdXRvUHJvY2Vzc29yJTBBJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJtZXRhLWxsYW1hJTJGTGxhbWEtMy4yLTExQi1WaXNpb24lMjIlMEFtb2RlbCUyMCUzRCUyME1sbGFtYUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQlMkMlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYpJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIyJTNDJTdDaW1hZ2UlN0MlM0VJZiUyMEklMjBoYWQlMjB0byUyMHdyaXRlJTIwYSUyMGhhaWt1JTIwZm9yJTIwdGhpcyUyMG9uZSUyMiUwQXVybCUyMCUzRCUyMCUyMmh0dHBzJTNBJTJGJTJGbGxhdmEtdmwuZ2l0aHViLmlvJTJGc3RhdGljJTJGaW1hZ2VzJTJGdmlldy5qcGclMjIlMEFyYXdfaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKHRleHQlM0Rwcm9tcHQlMkMlMjBpbWFnZXMlM0RyYXdfaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBkb19zYW1wbGUlM0RGYWxzZSUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMjUpJTBBcHJpbnQocHJvY2Vzc29yLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MllamaForConditionalGeneration, AutoProcessor

model_id = <span class="hljs-string">&quot;meta-llama/Llama-3.2-11B-Vision&quot;</span>
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map=<span class="hljs-string">&quot;auto&quot;</span>, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

prompt = <span class="hljs-string">&quot;&lt;|image|&gt;If I had to write a haiku for this one&quot;</span>
url = <span class="hljs-string">&quot;https://llava-vl.github.io/static/images/view.jpg&quot;</span>
raw_image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

inputs = processor(text=prompt, images=raw_image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
output = model.generate(**inputs, do_sample=<span class="hljs-literal">False</span>, max_new_tokens=<span class="hljs-number">25</span>)
<span class="hljs-built_in">print</span>(processor.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),Ze=new V({props:{title:"MllamaConfig",local:"transformers.MllamaConfig",headingTag:"h2"}}),Pe=new I({props:{name:"class transformers.MllamaConfig",anchor:"transformers.MllamaConfig",parameters:[{name:"vision_config",val:" = None"},{name:"text_config",val:" = None"},{name:"image_token_index",val:" = 128256"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MllamaConfig.vision_config",description:`<strong>vision_config</strong> (<code>Union[AutoConfig, dict]</code>, <em>optional</em>, defaults to <code>MllamaVisionConfig</code>) &#x2014;
The config object or dictionary of the vision backbone.`,name:"vision_config"},{anchor:"transformers.MllamaConfig.text_config",description:`<strong>text_config</strong> (<code>Union[AutoConfig, dict]</code>, <em>optional</em>, defaults to <code>MllamaTextConfig</code>) &#x2014;
The config object or dictionary of the text backbone.`,name:"text_config"},{anchor:"transformers.MllamaConfig.image_token_index",description:`<strong>image_token_index</strong> (<code>int</code>, <em>optional</em>, defaults to 128256) &#x2014;
The image token index to encode the image prompt.`,name:"image_token_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/configuration_mllama.py#L299"}}),ce=new ft({props:{anchor:"transformers.MllamaConfig.example",$$slots:{default:[Ia]},$$scope:{ctx:v}}}),Ge=new V({props:{title:"MllamaProcessor",local:"transformers.MllamaProcessor",headingTag:"h2"}}),Be=new I({props:{name:"class transformers.MllamaProcessor",anchor:"transformers.MllamaProcessor",parameters:[{name:"image_processor",val:""},{name:"tokenizer",val:""},{name:"chat_template",val:" = None"}],parametersDescription:[{anchor:"transformers.MllamaProcessor.image_processor",description:`<strong>image_processor</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor">MllamaImageProcessor</a>) &#x2014;
The image processor is a required input.`,name:"image_processor"},{anchor:"transformers.MllamaProcessor.tokenizer",description:`<strong>tokenizer</strong> ([<code>PreTrainedTokenizer</code>, <code>PreTrainedTokenizerFast</code>]) &#x2014;
The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.MllamaProcessor.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014; A Jinja template which will be used to convert lists of messages
in a chat into a tokenizable string.`,name:"chat_template"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/processing_mllama.py#L175"}}),de=new ft({props:{anchor:"transformers.MllamaProcessor.example",$$slots:{default:[Ca]},$$scope:{ctx:v}}}),Ne=new I({props:{name:"post_process_image_text_to_text",anchor:"transformers.MllamaProcessor.post_process_image_text_to_text",parameters:[{name:"generated_outputs",val:""},{name:"skip_special_tokens",val:" = True"},{name:"clean_up_tokenization_spaces",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MllamaProcessor.post_process_image_text_to_text.generated_outputs",description:`<strong>generated_outputs</strong> (<code>torch.Tensor</code> or <code>np.ndarray</code>) &#x2014;
The output of the model <code>generate</code> function. The output is expected to be a tensor of shape <code>(batch_size, sequence_length)</code>
or <code>(sequence_length,)</code>.`,name:"generated_outputs"},{anchor:"transformers.MllamaProcessor.post_process_image_text_to_text.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to remove special tokens in the output. Argument passed to the tokenizer&#x2019;s <code>batch_decode</code> method.`,name:"skip_special_tokens"},{anchor:"transformers.MllamaProcessor.post_process_image_text_to_text.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer&#x2019;s <code>batch_decode</code> method.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.MllamaProcessor.post_process_image_text_to_text.*kwargs",description:`*<strong>*kwargs</strong> &#x2014;
Additional arguments to be passed to the tokenizer&#x2019;s <code>batch_decode method</code>.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/processing_mllama.py#L344",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded text.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code></p>
`}}),qe=new V({props:{title:"MllamaImageProcessor",local:"transformers.MllamaImageProcessor",headingTag:"h2"}}),Re=new I({props:{name:"class transformers.MllamaImageProcessor",anchor:"transformers.MllamaImageProcessor",parameters:[{name:"do_convert_rgb",val:": bool = True"},{name:"do_resize",val:": bool = True"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"resample",val:": Resampling = <Resampling.BILINEAR: 2>"},{name:"do_rescale",val:": bool = True"},{name:"rescale_factor",val:": float = 0.00392156862745098"},{name:"do_normalize",val:": bool = True"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"do_pad",val:": bool = True"},{name:"max_image_tiles",val:": int = 4"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MllamaImageProcessor.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
Only has an effect if the input image is in the PIL format.`,name:"do_convert_rgb"},{anchor:"transformers.MllamaImageProcessor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.MllamaImageProcessor.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>, defaults to <code>self.size</code>) &#x2014;
Size of the image tile. Should be a dictionary containing &#x2018;height&#x2019; and &#x2018;width&#x2019; keys, both with integer values.
The height and width values should be equal.`,name:"size"},{anchor:"transformers.MllamaImageProcessor.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>Resampling.BILINEAR</code>) &#x2014;
Resampling filter to use if resizing the image. This can be one of the enum <code>PILImageResampling</code>. Only
has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.MllamaImageProcessor.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.MllamaImageProcessor.rescale_factor",description:`<strong>rescale_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.MllamaImageProcessor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.MllamaImageProcessor.image_mean",description:`<strong>image_mean</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>self.image_mean</code>) &#x2014;
Image mean to use for normalization. Only has an effect if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_mean"},{anchor:"transformers.MllamaImageProcessor.image_std",description:`<strong>image_std</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>self.image_std</code>) &#x2014;
Image standard deviation to use for normalization. Only has an effect if <code>do_normalize</code> is set to
<code>True</code>.`,name:"image_std"},{anchor:"transformers.MllamaImageProcessor.do_pad",description:`<strong>do_pad</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to pad the images to the largest height and width in the batch.`,name:"do_pad"},{anchor:"transformers.MllamaImageProcessor.max_image_tiles",description:`<strong>max_image_tiles</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The maximum number of tiles to split the image into.`,name:"max_image_tiles"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/image_processing_mllama.py#L536"}}),Xe=new I({props:{name:"pad",anchor:"transformers.MllamaImageProcessor.pad",parameters:[{name:"image",val:": ndarray"},{name:"size",val:": dict"},{name:"aspect_ratio",val:": tuple"},{name:"data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}],parametersDescription:[{anchor:"transformers.MllamaImageProcessor.pad.image",description:`<strong>image</strong> (<code>np.ndarray</code>) &#x2014;
Image to resize.`,name:"image"},{anchor:"transformers.MllamaImageProcessor.pad.size",description:`<strong>size</strong> (<code>dict[str, int]</code>) &#x2014;
Size of the output image.`,name:"size"},{anchor:"transformers.MllamaImageProcessor.pad.aspect_ratio",description:`<strong>aspect_ratio</strong> (<code>tuple[int, int]</code>) &#x2014;
The aspect ratio of the image.`,name:"aspect_ratio"},{anchor:"transformers.MllamaImageProcessor.pad.data_format",description:`<strong>data_format</strong> (<code>str</code> or <code>ChannelDimension</code>, <em>optional</em>) &#x2014;
The channel dimension format of the image. If not provided, it will be the same as the input image.`,name:"data_format"},{anchor:"transformers.MllamaImageProcessor.pad.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>) &#x2014;
The channel dimension format of the input image. If not provided, it will be inferred.`,name:"input_data_format"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/image_processing_mllama.py#L789",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The padded image.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>np.ndarray</code></p>
`}}),Ee=new I({props:{name:"preprocess",anchor:"transformers.MllamaImageProcessor.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"do_convert_rgb",val:": typing.Optional[bool] = None"},{name:"do_resize",val:": typing.Optional[bool] = None"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"resample",val:": typing.Optional[PIL.Image.Resampling] = None"},{name:"do_rescale",val:": typing.Optional[bool] = None"},{name:"rescale_factor",val:": typing.Optional[float] = None"},{name:"do_normalize",val:": typing.Optional[bool] = None"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"do_pad",val:": typing.Optional[bool] = None"},{name:"max_image_tiles",val:": typing.Optional[int] = None"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}],parametersDescription:[{anchor:"transformers.MllamaImageProcessor.preprocess.images",description:`<strong>images</strong> (<code>ImageInput</code>) &#x2014;
A list of images to preprocess.`,name:"images"},{anchor:"transformers.MllamaImageProcessor.preprocess.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_convert_rgb</code>) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.MllamaImageProcessor.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_resize</code>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.MllamaImageProcessor.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>, defaults to <code>self.size</code>) &#x2014;
Size of the image tile. Should be a dictionary containing &#x2018;height&#x2019; and &#x2018;width&#x2019; keys, both with integer values.
The height and width values should be equal.`,name:"size"},{anchor:"transformers.MllamaImageProcessor.preprocess.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.resample</code>) &#x2014;
Resampling filter to use if resizing the image. This can be one of the enum <code>PILImageResampling</code>. Only
has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.MllamaImageProcessor.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_rescale</code>) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.MllamaImageProcessor.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>float</code>, <em>optional</em>, defaults to <code>self.rescale_factor</code>) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.MllamaImageProcessor.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_normalize</code>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.MllamaImageProcessor.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>self.image_mean</code>) &#x2014;
Image mean to use for normalization. Only has an effect if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_mean"},{anchor:"transformers.MllamaImageProcessor.preprocess.image_std",description:`<strong>image_std</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>self.image_std</code>) &#x2014;
Image standard deviation to use for normalization. Only has an effect if <code>do_normalize</code> is set to
<code>True</code>.`,name:"image_std"},{anchor:"transformers.MllamaImageProcessor.preprocess.do_pad",description:`<strong>do_pad</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_pad</code>) &#x2014;
Whether or not to pad the images to the largest height and width in the batch.`,name:"do_pad"},{anchor:"transformers.MllamaImageProcessor.preprocess.max_image_tiles",description:`<strong>max_image_tiles</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.max_image_tiles</code>) &#x2014;
The maximum number of tiles to split the image into.`,name:"max_image_tiles"},{anchor:"transformers.MllamaImageProcessor.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"},{anchor:"transformers.MllamaImageProcessor.preprocess.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <code>TensorType</code>, <em>optional</em>) &#x2014;
The type of tensors to return. Can be one of:<ul>
<li>Unset: Return a list of <code>np.ndarray</code>.</li>
<li><code>TensorType.TENSORFLOW</code> or <code>&apos;tf&apos;</code>: Return a batch of type <code>tf.Tensor</code>.</li>
<li><code>TensorType.PYTORCH</code> or <code>&apos;pt&apos;</code>: Return a batch of type <code>torch.Tensor</code>.</li>
<li><code>TensorType.NUMPY</code> or <code>&apos;np&apos;</code>: Return a batch of type <code>np.ndarray</code>.</li>
<li><code>TensorType.JAX</code> or <code>&apos;jax&apos;</code>: Return a batch of type <code>jax.numpy.ndarray</code>.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/image_processing_mllama.py#L601",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<ul>
<li><strong>pixel_values</strong> (<code>TensorType</code>): The preprocessed pixel values.</li>
<li><strong>aspect_ratio_ids</strong> (<code>TensorType</code>): The aspect ratio ids of the images.</li>
<li><strong>num_tiles</strong> (<code>list[list[int]]</code>): The number of tiles for each image in the batch.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>BatchFeature</code> of the following structure</p>
`}}),He=new I({props:{name:"resize",anchor:"transformers.MllamaImageProcessor.resize",parameters:[{name:"image",val:": ndarray"},{name:"size",val:": dict"},{name:"max_image_tiles",val:": int"},{name:"resample",val:": Resampling = <Resampling.BILINEAR: 2>"},{name:"data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}],parametersDescription:[{anchor:"transformers.MllamaImageProcessor.resize.image",description:`<strong>image</strong> (<code>np.ndarray</code>) &#x2014;
Image to resize.`,name:"image"},{anchor:"transformers.MllamaImageProcessor.resize.size",description:`<strong>size</strong> (<code>dict[str, int]</code>) &#x2014;
Size of the output image.`,name:"size"},{anchor:"transformers.MllamaImageProcessor.resize.max_image_tiles",description:`<strong>max_image_tiles</strong> (<code>int</code>) &#x2014;
The maximum number of tiles to split the image into.`,name:"max_image_tiles"},{anchor:"transformers.MllamaImageProcessor.resize.resample",description:`<strong>resample</strong> (<code>PILImageResampling</code>, <em>optional</em>, defaults to <code>PILImageResampling.BICUBIC</code>) &#x2014;
Resampling filter to use when resizing the image.`,name:"resample"},{anchor:"transformers.MllamaImageProcessor.resize.data_format",description:`<strong>data_format</strong> (<code>str</code> or <code>ChannelDimension</code>, <em>optional</em>) &#x2014;
The channel dimension format of the image. If not provided, it will be the same as the input image.`,name:"data_format"},{anchor:"transformers.MllamaImageProcessor.resize.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>) &#x2014;
The channel dimension format of the input image. If not provided, it will be inferred.`,name:"input_data_format"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/image_processing_mllama.py#L836",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The resized image and a tuple containing the number of tiles
along the height and width dimensions.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Union[np.ndarray, tuple[int, int]]</code></p>
`}}),Qe=new V({props:{title:"MllamaForConditionalGeneration",local:"transformers.MllamaForConditionalGeneration",headingTag:"h2"}}),Ye=new I({props:{name:"class transformers.MllamaForConditionalGeneration",anchor:"transformers.MllamaForConditionalGeneration",parameters:[{name:"config",val:": MllamaConfig"}],parametersDescription:[{anchor:"transformers.MllamaForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig">MllamaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1541"}}),Se=new I({props:{name:"forward",anchor:"transformers.MllamaForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"aspect_ratio_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"aspect_ratio_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attention_states",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.MllamaForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MllamaForConditionalGeneration.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor">MllamaImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">MllamaImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaProcessor">MllamaProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor">MllamaImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.MllamaForConditionalGeneration.forward.aspect_ratio_mask",description:`<strong>aspect_ratio_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, max_num_images, max_num_tiles)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding tiles. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tiles that are <strong>not masked</strong>,</li>
<li>0 for tiles that are <strong>masked</strong>.</li>
</ul>`,name:"aspect_ratio_mask"},{anchor:"transformers.MllamaForConditionalGeneration.forward.aspect_ratio_ids",description:`<strong>aspect_ratio_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, max_num_images)</code>, <em>optional</em>) &#x2014;
Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
These ids correspond to indices in the model&#x2019;s list of supported aspect ratios, offset by 1.</p>
<p>For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:</p>
<ul>
<li>An image with aspect ratio [1, 1] would have ID 1</li>
<li>An image with aspect ratio [1, 2] would have ID 2</li>
<li>An image with aspect ratio [2, 1] would have ID 3</li>
</ul>
<p>The id 0 is reserved for padding (i.e., no image).</p>
<p>If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its <code>aspect_ratio_id</code> would be 2.`,name:"aspect_ratio_ids"},{anchor:"transformers.MllamaForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MllamaForConditionalGeneration.forward.cross_attention_mask",description:`<strong>cross_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, seq_length, max_num_images, max_num_tiles)</code>, <em>optional</em>) &#x2014;
Cross-attention mask to control the interaction between text tokens and image tiles.
This 4D tensor defines which image tiles each text token should attend to.</p>
<p>For each text token (in seq_length):</p>
<ul>
<li>1 indicates the token <strong>should attend</strong> to the corresponding image tile</li>
<li>0 indicates the token <strong>should not attend</strong> to the corresponding image tile</li>
</ul>`,name:"cross_attention_mask"},{anchor:"transformers.MllamaForConditionalGeneration.forward.cross_attention_states",description:`<strong>cross_attention_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Output of the vision model, used for cross-attention. This tensor contains the processed image features that
the language model will attend to.`,name:"cross_attention_states"},{anchor:"transformers.MllamaForConditionalGeneration.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MllamaForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MllamaForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MllamaForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MllamaForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MllamaForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.MllamaForConditionalGeneration.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1577",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig"
>MllamaConfig</a>) and inputs.</p>
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
`}}),ue=new gt({props:{$$slots:{default:[Ja]},$$scope:{ctx:v}}}),ge=new ft({props:{anchor:"transformers.MllamaForConditionalGeneration.forward.example",$$slots:{default:[Ua]},$$scope:{ctx:v}}}),De=new V({props:{title:"MllamaForCausalLM",local:"transformers.MllamaForCausalLM",headingTag:"h2"}}),Ae=new I({props:{name:"class transformers.MllamaForCausalLM",anchor:"transformers.MllamaForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MllamaForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM">MllamaForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1287"}}),Oe=new I({props:{name:"forward",anchor:"transformers.MllamaForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"cross_attention_states",val:": typing.Optional[torch.LongTensor] = None"},{name:"cross_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"full_text_row_masked_out_mask",val:": typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.MllamaForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MllamaForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MllamaForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MllamaForCausalLM.forward.cross_attention_states",description:`<strong>cross_attention_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Output of the vision model, used for cross-attention. This tensor contains the processed image features that
the language model will attend to.`,name:"cross_attention_states"},{anchor:"transformers.MllamaForCausalLM.forward.cross_attention_mask",description:`<strong>cross_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, seq_length, max_num_images, max_num_tiles)</code>, <em>optional</em>) &#x2014;
Cross-attention mask to control the interaction between text tokens and image tiles.
This 4D tensor defines which image tiles each text token should attend to.</p>
<p>For each text token (in seq_length):</p>
<ul>
<li>1 indicates the token <strong>should attend</strong> to the corresponding image tile</li>
<li>0 indicates the token <strong>should not attend</strong> to the corresponding image tile</li>
</ul>`,name:"cross_attention_mask"},{anchor:"transformers.MllamaForCausalLM.forward.full_text_row_masked_out_mask",description:`<strong>full_text_row_masked_out_mask</strong> (<code>tuple[torch.Tensor, torch.Tensor]</code>, <em>optional</em>) &#x2014;
A tuple containing two tensors that mask out rows in the cross-attention mechanism:</p>
<ul>
<li>The first tensor has shape <code>(batch_size, 1, seq_length, 1)</code> and contains values of 0 or 1.
A value of 0 indicates that the corresponding text token&#x2019;s entire row in the cross-attention
matrix should be masked out (all image tokens ignored).</li>
<li>The second tensor has the same shape and is used internally to apply the masking during
the forward pass of cross-attention layers.
This mask is derived from the cross_attention_mask and is used to handle cases where a text token
should not attend to any image token.</li>
</ul>`,name:"full_text_row_masked_out_mask"},{anchor:"transformers.MllamaForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MllamaForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MllamaForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MllamaForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MllamaForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.MllamaForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1302",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig"
>MllamaConfig</a>) and inputs.</p>
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
`}}),fe=new gt({props:{$$slots:{default:[za]},$$scope:{ctx:v}}}),_e=new ft({props:{anchor:"transformers.MllamaForCausalLM.forward.example",$$slots:{default:[ja]},$$scope:{ctx:v}}}),Ke=new V({props:{title:"MllamaTextModel",local:"transformers.MllamaTextModel",headingTag:"h2"}}),et=new I({props:{name:"class transformers.MllamaTextModel",anchor:"transformers.MllamaTextModel",parameters:[{name:"config",val:": MllamaTextConfig"}],parametersDescription:[{anchor:"transformers.MllamaTextModel.config",description:`<strong>config</strong> (<code>MllamaTextConfig</code>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1141"}}),tt=new I({props:{name:"forward",anchor:"transformers.MllamaTextModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"cross_attention_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"full_text_row_masked_out_mask",val:": typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}],parametersDescription:[{anchor:"transformers.MllamaTextModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MllamaTextModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MllamaTextModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MllamaTextModel.forward.cross_attention_states",description:`<strong>cross_attention_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Output of the vision model, used for cross-attention. This tensor contains the processed image features that
the language model will attend to.`,name:"cross_attention_states"},{anchor:"transformers.MllamaTextModel.forward.cross_attention_mask",description:`<strong>cross_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, seq_length, max_num_images, max_num_tiles)</code>, <em>optional</em>) &#x2014;
Cross-attention mask to control the interaction between text tokens and image tiles.
This 4D tensor defines which image tiles each text token should attend to.</p>
<p>For each text token (in seq_length):</p>
<ul>
<li>1 indicates the token <strong>should attend</strong> to the corresponding image tile</li>
<li>0 indicates the token <strong>should not attend</strong> to the corresponding image tile</li>
</ul>`,name:"cross_attention_mask"},{anchor:"transformers.MllamaTextModel.forward.full_text_row_masked_out_mask",description:`<strong>full_text_row_masked_out_mask</strong> (<code>tuple[torch.Tensor, torch.Tensor]</code>, <em>optional</em>) &#x2014;
A tuple containing two tensors that mask out rows in the cross-attention mechanism:</p>
<ul>
<li>The first tensor has shape <code>(batch_size, 1, seq_length, 1)</code> and contains values of 0 or 1.
A value of 0 indicates that the corresponding text token&#x2019;s entire row in the cross-attention
matrix should be masked out (all image tokens ignored).</li>
<li>The second tensor has the same shape and is used internally to apply the masking during
the forward pass of cross-attention layers.
This mask is derived from the cross_attention_mask and is used to handle cases where a text token
should not attend to any image token.</li>
</ul>`,name:"full_text_row_masked_out_mask"},{anchor:"transformers.MllamaTextModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MllamaTextModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MllamaTextModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MllamaTextModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1165",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig"
>MllamaConfig</a>) and inputs.</p>
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
`}}),Me=new gt({props:{$$slots:{default:[Fa]},$$scope:{ctx:v}}}),Te=new ft({props:{anchor:"transformers.MllamaTextModel.forward.example",$$slots:{default:[Wa]},$$scope:{ctx:v}}}),ot=new V({props:{title:"MllamaModel",local:"transformers.MllamaModel",headingTag:"h2"}}),nt=new I({props:{name:"class transformers.MllamaModel",anchor:"transformers.MllamaModel",parameters:[{name:"config",val:": MllamaConfig"}],parametersDescription:[{anchor:"transformers.MllamaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig">MllamaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1401"}}),at=new I({props:{name:"forward",anchor:"transformers.MllamaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"aspect_ratio_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"aspect_ratio_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attention_states",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}],parametersDescription:[{anchor:"transformers.MllamaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MllamaModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor">MllamaImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">MllamaImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaProcessor">MllamaProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor">MllamaImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.MllamaModel.forward.aspect_ratio_mask",description:`<strong>aspect_ratio_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, max_num_images, max_num_tiles)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding tiles. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tiles that are <strong>not masked</strong>,</li>
<li>0 for tiles that are <strong>masked</strong>.</li>
</ul>`,name:"aspect_ratio_mask"},{anchor:"transformers.MllamaModel.forward.aspect_ratio_ids",description:`<strong>aspect_ratio_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, max_num_images)</code>, <em>optional</em>) &#x2014;
Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
These ids correspond to indices in the model&#x2019;s list of supported aspect ratios, offset by 1.</p>
<p>For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:</p>
<ul>
<li>An image with aspect ratio [1, 1] would have ID 1</li>
<li>An image with aspect ratio [1, 2] would have ID 2</li>
<li>An image with aspect ratio [2, 1] would have ID 3</li>
</ul>
<p>The id 0 is reserved for padding (i.e., no image).</p>
<p>If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its <code>aspect_ratio_id</code> would be 2.`,name:"aspect_ratio_ids"},{anchor:"transformers.MllamaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MllamaModel.forward.cross_attention_mask",description:`<strong>cross_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, seq_length, max_num_images, max_num_tiles)</code>, <em>optional</em>) &#x2014;
Cross-attention mask to control the interaction between text tokens and image tiles.
This 4D tensor defines which image tiles each text token should attend to.</p>
<p>For each text token (in seq_length):</p>
<ul>
<li>1 indicates the token <strong>should attend</strong> to the corresponding image tile</li>
<li>0 indicates the token <strong>should not attend</strong> to the corresponding image tile</li>
</ul>`,name:"cross_attention_mask"},{anchor:"transformers.MllamaModel.forward.cross_attention_states",description:`<strong>cross_attention_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Output of the vision model, used for cross-attention. This tensor contains the processed image features that
the language model will attend to.`,name:"cross_attention_states"},{anchor:"transformers.MllamaModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MllamaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MllamaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MllamaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MllamaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1433",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig"
>MllamaConfig</a>) and inputs.</p>
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
`}}),be=new gt({props:{$$slots:{default:[La]},$$scope:{ctx:v}}}),st=new V({props:{title:"MllamaForCausalLM",local:"transformers.MllamaForCausalLM",headingTag:"h2"}}),rt=new I({props:{name:"class transformers.MllamaForCausalLM",anchor:"transformers.MllamaForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MllamaForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaForCausalLM">MllamaForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1287"}}),lt=new I({props:{name:"forward",anchor:"transformers.MllamaForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"cross_attention_states",val:": typing.Optional[torch.LongTensor] = None"},{name:"cross_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"full_text_row_masked_out_mask",val:": typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.MllamaForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MllamaForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MllamaForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MllamaForCausalLM.forward.cross_attention_states",description:`<strong>cross_attention_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Output of the vision model, used for cross-attention. This tensor contains the processed image features that
the language model will attend to.`,name:"cross_attention_states"},{anchor:"transformers.MllamaForCausalLM.forward.cross_attention_mask",description:`<strong>cross_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, seq_length, max_num_images, max_num_tiles)</code>, <em>optional</em>) &#x2014;
Cross-attention mask to control the interaction between text tokens and image tiles.
This 4D tensor defines which image tiles each text token should attend to.</p>
<p>For each text token (in seq_length):</p>
<ul>
<li>1 indicates the token <strong>should attend</strong> to the corresponding image tile</li>
<li>0 indicates the token <strong>should not attend</strong> to the corresponding image tile</li>
</ul>`,name:"cross_attention_mask"},{anchor:"transformers.MllamaForCausalLM.forward.full_text_row_masked_out_mask",description:`<strong>full_text_row_masked_out_mask</strong> (<code>tuple[torch.Tensor, torch.Tensor]</code>, <em>optional</em>) &#x2014;
A tuple containing two tensors that mask out rows in the cross-attention mechanism:</p>
<ul>
<li>The first tensor has shape <code>(batch_size, 1, seq_length, 1)</code> and contains values of 0 or 1.
A value of 0 indicates that the corresponding text token&#x2019;s entire row in the cross-attention
matrix should be masked out (all image tokens ignored).</li>
<li>The second tensor has the same shape and is used internally to apply the masking during
the forward pass of cross-attention layers.
This mask is derived from the cross_attention_mask and is used to handle cases where a text token
should not attend to any image token.</li>
</ul>`,name:"full_text_row_masked_out_mask"},{anchor:"transformers.MllamaForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MllamaForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MllamaForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MllamaForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MllamaForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.MllamaForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L1302",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig"
>MllamaConfig</a>) and inputs.</p>
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
`}}),ye=new gt({props:{$$slots:{default:[Va]},$$scope:{ctx:v}}}),we=new ft({props:{anchor:"transformers.MllamaForCausalLM.forward.example",$$slots:{default:[Za]},$$scope:{ctx:v}}}),it=new V({props:{title:"MllamaVisionModel",local:"transformers.MllamaVisionModel",headingTag:"h2"}}),ct=new I({props:{name:"class transformers.MllamaVisionModel",anchor:"transformers.MllamaVisionModel",parameters:[{name:"config",val:": MllamaVisionConfig"}],parametersDescription:[{anchor:"transformers.MllamaVisionModel.config",description:`<strong>config</strong> (<code>MllamaVisionConfig</code>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L944"}}),dt=new I({props:{name:"forward",anchor:"transformers.MllamaVisionModel.forward",parameters:[{name:"pixel_values",val:": Tensor"},{name:"aspect_ratio_ids",val:": Tensor"},{name:"aspect_ratio_mask",val:": Tensor"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MllamaVisionModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor">MllamaImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">MllamaImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaProcessor">MllamaProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaImageProcessor">MllamaImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.MllamaVisionModel.forward.aspect_ratio_ids",description:`<strong>aspect_ratio_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, max_num_images)</code>, <em>optional</em>) &#x2014;
Aspect ratio ids used to select the appropriate precomputed tile embeddings based on the aspect ratio of each input image.
These ids correspond to indices in the model&#x2019;s list of supported aspect ratios, offset by 1.</p>
<p>For example, if the model supports aspect ratios [[1, 1], [1, 2], [2, 1]]:</p>
<ul>
<li>An image with aspect ratio [1, 1] would have ID 1</li>
<li>An image with aspect ratio [1, 2] would have ID 2</li>
<li>An image with aspect ratio [2, 1] would have ID 3</li>
</ul>
<p>The id 0 is reserved for padding (i.e., no image).</p>
<p>If an image has aspect ratio [1, 2], that means it was split into 2 tiles horizontally, and its <code>aspect_ratio_id</code> would be 2.`,name:"aspect_ratio_ids"},{anchor:"transformers.MllamaVisionModel.forward.aspect_ratio_mask",description:`<strong>aspect_ratio_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, max_num_images, max_num_tiles)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding tiles. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tiles that are <strong>not masked</strong>,</li>
<li>0 for tiles that are <strong>masked</strong>.</li>
</ul>`,name:"aspect_ratio_mask"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mllama/modeling_mllama.py#L997",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mllama#transformers.MllamaConfig"
>MllamaConfig</a>) and inputs.</p>
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
`}}),ve=new gt({props:{$$slots:{default:[Pa]},$$scope:{ctx:v}}}),ke=new ft({props:{anchor:"transformers.MllamaVisionModel.forward.example",$$slots:{default:[Ga]},$$scope:{ctx:v}}}),mt=new xa({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mllama.md"}}),{c(){t=c("meta"),b=a(),i=c("p"),m=a(),y=c("p"),y.innerHTML=r,w=a(),h(H.$$.fragment),k=a(),Z=c("div"),Z.innerHTML=Vn,Kt=a(),h(Ie.$$.fragment),eo=a(),Ce=c("p"),Ce.innerHTML=Zn,to=a(),Je=c("p"),Je.innerHTML=Pn,oo=a(),h(Ue.$$.fragment),no=a(),ze=c("ul"),ze.innerHTML=Gn,ao=a(),h(ie.$$.fragment),so=a(),h(je.$$.fragment),ro=a(),h(Fe.$$.fragment),lo=a(),h(We.$$.fragment),io=a(),h(Le.$$.fragment),co=a(),h(Ve.$$.fragment),mo=a(),h(Ze.$$.fragment),po=a(),C=c("div"),h(Pe.$$.fragment),jo=a(),_t=c("p"),_t.innerHTML=Bn,Fo=a(),Mt=c("p"),Mt.innerHTML=Nn,Wo=a(),Tt=c("p"),Tt.innerHTML=qn,Lo=a(),h(ce.$$.fragment),ho=a(),h(Ge.$$.fragment),uo=a(),G=c("div"),h(Be.$$.fragment),Vo=a(),bt=c("p"),bt.innerHTML=Rn,Zo=a(),h(de.$$.fragment),Po=a(),me=c("div"),h(Ne.$$.fragment),Go=a(),yt=c("p"),yt.textContent=Xn,go=a(),h(qe.$$.fragment),fo=a(),J=c("div"),h(Re.$$.fragment),Bo=a(),wt=c("p"),wt.textContent=En,No=a(),pe=c("div"),h(Xe.$$.fragment),qo=a(),vt=c("p"),vt.innerHTML=Hn,Ro=a(),he=c("div"),h(Ee.$$.fragment),Xo=a(),kt=c("p"),kt.textContent=Qn,Eo=a(),oe=c("div"),h(He.$$.fragment),Ho=a(),xt=c("p"),xt.textContent=Yn,Qo=a(),$t=c("p"),$t.textContent=Sn,_o=a(),h(Qe.$$.fragment),Mo=a(),U=c("div"),h(Ye.$$.fragment),Yo=a(),It=c("p"),It.textContent=Dn,So=a(),Ct=c("p"),Ct.innerHTML=An,Do=a(),Jt=c("p"),Jt.innerHTML=On,Ao=a(),Q=c("div"),h(Se.$$.fragment),Oo=a(),Ut=c("p"),Ut.innerHTML=Kn,Ko=a(),h(ue.$$.fragment),en=a(),h(ge.$$.fragment),To=a(),h(De.$$.fragment),bo=a(),z=c("div"),h(Ae.$$.fragment),tn=a(),zt=c("p"),zt.textContent=ea,on=a(),jt=c("p"),jt.innerHTML=ta,nn=a(),Ft=c("p"),Ft.innerHTML=oa,an=a(),Y=c("div"),h(Oe.$$.fragment),sn=a(),Wt=c("p"),Wt.innerHTML=na,rn=a(),h(fe.$$.fragment),ln=a(),h(_e.$$.fragment),yo=a(),h(Ke.$$.fragment),wo=a(),j=c("div"),h(et.$$.fragment),cn=a(),Lt=c("p"),Lt.textContent=aa,dn=a(),Vt=c("p"),Vt.innerHTML=sa,mn=a(),Zt=c("p"),Zt.innerHTML=ra,pn=a(),S=c("div"),h(tt.$$.fragment),hn=a(),Pt=c("p"),Pt.innerHTML=la,un=a(),h(Me.$$.fragment),gn=a(),h(Te.$$.fragment),vo=a(),h(ot.$$.fragment),ko=a(),F=c("div"),h(nt.$$.fragment),fn=a(),Gt=c("p"),Gt.textContent=ia,_n=a(),Bt=c("p"),Bt.innerHTML=ca,Mn=a(),Nt=c("p"),Nt.innerHTML=da,Tn=a(),ne=c("div"),h(at.$$.fragment),bn=a(),qt=c("p"),qt.innerHTML=ma,yn=a(),h(be.$$.fragment),xo=a(),h(st.$$.fragment),$o=a(),W=c("div"),h(rt.$$.fragment),wn=a(),Rt=c("p"),Rt.textContent=pa,vn=a(),Xt=c("p"),Xt.innerHTML=ha,kn=a(),Et=c("p"),Et.innerHTML=ua,xn=a(),D=c("div"),h(lt.$$.fragment),$n=a(),Ht=c("p"),Ht.innerHTML=ga,In=a(),h(ye.$$.fragment),Cn=a(),h(we.$$.fragment),Io=a(),h(it.$$.fragment),Co=a(),L=c("div"),h(ct.$$.fragment),Jn=a(),Qt=c("p"),Qt.textContent=fa,Un=a(),Yt=c("p"),Yt.innerHTML=_a,zn=a(),St=c("p"),St.innerHTML=Ma,jn=a(),A=c("div"),h(dt.$$.fragment),Fn=a(),Dt=c("p"),Dt.innerHTML=Ta,Wn=a(),h(ve.$$.fragment),Ln=a(),h(ke.$$.fragment),Jo=a(),h(mt.$$.fragment),Uo=a(),Ot=c("p"),this.h()},l(e){const o=ka("svelte-u9bgzb",document.head);t=d(o,"META",{name:!0,content:!0}),o.forEach(l),b=s(e),i=d(e,"P",{}),$(i).forEach(l),m=s(e),y=d(e,"P",{"data-svelte-h":!0}),T(y)!=="svelte-3exsda"&&(y.innerHTML=r),w=s(e),u(H.$$.fragment,e),k=s(e),Z=d(e,"DIV",{class:!0,"data-svelte-h":!0}),T(Z)!=="svelte-13t8s2t"&&(Z.innerHTML=Vn),Kt=s(e),u(Ie.$$.fragment,e),eo=s(e),Ce=d(e,"P",{"data-svelte-h":!0}),T(Ce)!=="svelte-69pn6"&&(Ce.innerHTML=Zn),to=s(e),Je=d(e,"P",{"data-svelte-h":!0}),T(Je)!=="svelte-1noqrap"&&(Je.innerHTML=Pn),oo=s(e),u(Ue.$$.fragment,e),no=s(e),ze=d(e,"UL",{"data-svelte-h":!0}),T(ze)!=="svelte-1dx74jo"&&(ze.innerHTML=Gn),ao=s(e),u(ie.$$.fragment,e),so=s(e),u(je.$$.fragment,e),ro=s(e),u(Fe.$$.fragment,e),lo=s(e),u(We.$$.fragment,e),io=s(e),u(Le.$$.fragment,e),co=s(e),u(Ve.$$.fragment,e),mo=s(e),u(Ze.$$.fragment,e),po=s(e),C=d(e,"DIV",{class:!0});var B=$(C);u(Pe.$$.fragment,B),jo=s(B),_t=d(B,"P",{"data-svelte-h":!0}),T(_t)!=="svelte-1bvzi9b"&&(_t.innerHTML=Bn),Fo=s(B),Mt=d(B,"P",{"data-svelte-h":!0}),T(Mt)!=="svelte-8fyj87"&&(Mt.innerHTML=Nn),Wo=s(B),Tt=d(B,"P",{"data-svelte-h":!0}),T(Tt)!=="svelte-1ek1ss9"&&(Tt.innerHTML=qn),Lo=s(B),u(ce.$$.fragment,B),B.forEach(l),ho=s(e),u(Ge.$$.fragment,e),uo=s(e),G=d(e,"DIV",{class:!0});var O=$(G);u(Be.$$.fragment,O),Vo=s(O),bt=d(O,"P",{"data-svelte-h":!0}),T(bt)!=="svelte-18j6mie"&&(bt.innerHTML=Rn),Zo=s(O),u(de.$$.fragment,O),Po=s(O),me=d(O,"DIV",{class:!0});var pt=$(me);u(Ne.$$.fragment,pt),Go=s(pt),yt=d(pt,"P",{"data-svelte-h":!0}),T(yt)!=="svelte-z5vbkk"&&(yt.textContent=Xn),pt.forEach(l),O.forEach(l),go=s(e),u(qe.$$.fragment,e),fo=s(e),J=d(e,"DIV",{class:!0});var N=$(J);u(Re.$$.fragment,N),Bo=s(N),wt=d(N,"P",{"data-svelte-h":!0}),T(wt)!=="svelte-h03vcc"&&(wt.textContent=En),No=s(N),pe=d(N,"DIV",{class:!0});var ht=$(pe);u(Xe.$$.fragment,ht),qo=s(ht),vt=d(ht,"P",{"data-svelte-h":!0}),T(vt)!=="svelte-dymnub"&&(vt.innerHTML=Hn),ht.forEach(l),Ro=s(N),he=d(N,"DIV",{class:!0});var ut=$(he);u(Ee.$$.fragment,ut),Xo=s(ut),kt=d(ut,"P",{"data-svelte-h":!0}),T(kt)!=="svelte-1xds4wy"&&(kt.textContent=Qn),ut.forEach(l),Eo=s(N),oe=d(N,"DIV",{class:!0});var le=$(oe);u(He.$$.fragment,le),Ho=s(le),xt=d(le,"P",{"data-svelte-h":!0}),T(xt)!=="svelte-zg52p1"&&(xt.textContent=Yn),Qo=s(le),$t=d(le,"P",{"data-svelte-h":!0}),T($t)!=="svelte-1tffesx"&&($t.textContent=Sn),le.forEach(l),N.forEach(l),_o=s(e),u(Qe.$$.fragment,e),Mo=s(e),U=d(e,"DIV",{class:!0});var q=$(U);u(Ye.$$.fragment,q),Yo=s(q),It=d(q,"P",{"data-svelte-h":!0}),T(It)!=="svelte-1nm8kx0"&&(It.textContent=Dn),So=s(q),Ct=d(q,"P",{"data-svelte-h":!0}),T(Ct)!=="svelte-q52n56"&&(Ct.innerHTML=An),Do=s(q),Jt=d(q,"P",{"data-svelte-h":!0}),T(Jt)!=="svelte-hswkmf"&&(Jt.innerHTML=On),Ao=s(q),Q=d(q,"DIV",{class:!0});var K=$(Q);u(Se.$$.fragment,K),Oo=s(K),Ut=d(K,"P",{"data-svelte-h":!0}),T(Ut)!=="svelte-10a6yv9"&&(Ut.innerHTML=Kn),Ko=s(K),u(ue.$$.fragment,K),en=s(K),u(ge.$$.fragment,K),K.forEach(l),q.forEach(l),To=s(e),u(De.$$.fragment,e),bo=s(e),z=d(e,"DIV",{class:!0});var R=$(z);u(Ae.$$.fragment,R),tn=s(R),zt=d(R,"P",{"data-svelte-h":!0}),T(zt)!=="svelte-vkcich"&&(zt.textContent=ea),on=s(R),jt=d(R,"P",{"data-svelte-h":!0}),T(jt)!=="svelte-q52n56"&&(jt.innerHTML=ta),nn=s(R),Ft=d(R,"P",{"data-svelte-h":!0}),T(Ft)!=="svelte-hswkmf"&&(Ft.innerHTML=oa),an=s(R),Y=d(R,"DIV",{class:!0});var ee=$(Y);u(Oe.$$.fragment,ee),sn=s(ee),Wt=d(ee,"P",{"data-svelte-h":!0}),T(Wt)!=="svelte-55l5tn"&&(Wt.innerHTML=na),rn=s(ee),u(fe.$$.fragment,ee),ln=s(ee),u(_e.$$.fragment,ee),ee.forEach(l),R.forEach(l),yo=s(e),u(Ke.$$.fragment,e),wo=s(e),j=d(e,"DIV",{class:!0});var X=$(j);u(et.$$.fragment,X),cn=s(X),Lt=d(X,"P",{"data-svelte-h":!0}),T(Lt)!=="svelte-20uxyn"&&(Lt.textContent=aa),dn=s(X),Vt=d(X,"P",{"data-svelte-h":!0}),T(Vt)!=="svelte-q52n56"&&(Vt.innerHTML=sa),mn=s(X),Zt=d(X,"P",{"data-svelte-h":!0}),T(Zt)!=="svelte-hswkmf"&&(Zt.innerHTML=ra),pn=s(X),S=d(X,"DIV",{class:!0});var te=$(S);u(tt.$$.fragment,te),hn=s(te),Pt=d(te,"P",{"data-svelte-h":!0}),T(Pt)!=="svelte-vxvcyf"&&(Pt.innerHTML=la),un=s(te),u(Me.$$.fragment,te),gn=s(te),u(Te.$$.fragment,te),te.forEach(l),X.forEach(l),vo=s(e),u(ot.$$.fragment,e),ko=s(e),F=d(e,"DIV",{class:!0});var E=$(F);u(nt.$$.fragment,E),fn=s(E),Gt=d(E,"P",{"data-svelte-h":!0}),T(Gt)!=="svelte-wyv0kl"&&(Gt.textContent=ia),_n=s(E),Bt=d(E,"P",{"data-svelte-h":!0}),T(Bt)!=="svelte-q52n56"&&(Bt.innerHTML=ca),Mn=s(E),Nt=d(E,"P",{"data-svelte-h":!0}),T(Nt)!=="svelte-hswkmf"&&(Nt.innerHTML=da),Tn=s(E),ne=d(E,"DIV",{class:!0});var At=$(ne);u(at.$$.fragment,At),bn=s(At),qt=d(At,"P",{"data-svelte-h":!0}),T(qt)!=="svelte-qmn5mv"&&(qt.innerHTML=ma),yn=s(At),u(be.$$.fragment,At),At.forEach(l),E.forEach(l),xo=s(e),u(st.$$.fragment,e),$o=s(e),W=d(e,"DIV",{class:!0});var ae=$(W);u(rt.$$.fragment,ae),wn=s(ae),Rt=d(ae,"P",{"data-svelte-h":!0}),T(Rt)!=="svelte-vkcich"&&(Rt.textContent=pa),vn=s(ae),Xt=d(ae,"P",{"data-svelte-h":!0}),T(Xt)!=="svelte-q52n56"&&(Xt.innerHTML=ha),kn=s(ae),Et=d(ae,"P",{"data-svelte-h":!0}),T(Et)!=="svelte-hswkmf"&&(Et.innerHTML=ua),xn=s(ae),D=d(ae,"DIV",{class:!0});var xe=$(D);u(lt.$$.fragment,xe),$n=s(xe),Ht=d(xe,"P",{"data-svelte-h":!0}),T(Ht)!=="svelte-55l5tn"&&(Ht.innerHTML=ga),In=s(xe),u(ye.$$.fragment,xe),Cn=s(xe),u(we.$$.fragment,xe),xe.forEach(l),ae.forEach(l),Io=s(e),u(it.$$.fragment,e),Co=s(e),L=d(e,"DIV",{class:!0});var se=$(L);u(ct.$$.fragment,se),Jn=s(se),Qt=d(se,"P",{"data-svelte-h":!0}),T(Qt)!=="svelte-1knm5fr"&&(Qt.textContent=fa),Un=s(se),Yt=d(se,"P",{"data-svelte-h":!0}),T(Yt)!=="svelte-q52n56"&&(Yt.innerHTML=_a),zn=s(se),St=d(se,"P",{"data-svelte-h":!0}),T(St)!=="svelte-hswkmf"&&(St.innerHTML=Ma),jn=s(se),A=d(se,"DIV",{class:!0});var $e=$(A);u(dt.$$.fragment,$e),Fn=s($e),Dt=d($e,"P",{"data-svelte-h":!0}),T(Dt)!=="svelte-1j3jq77"&&(Dt.innerHTML=Ta),Wn=s($e),u(ve.$$.fragment,$e),Ln=s($e),u(ke.$$.fragment,$e),$e.forEach(l),se.forEach(l),Jo=s(e),u(mt.$$.fragment,e),Uo=s(e),Ot=d(e,"P",{}),$(Ot).forEach(l),this.h()},h(){x(t,"name","hf:doc:metadata"),x(t,"content",Na),x(Z,"class","flex flex-wrap space-x-1"),x(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){n(document.head,t),p(e,b,o),p(e,i,o),p(e,m,o),p(e,y,o),p(e,w,o),g(H,e,o),p(e,k,o),p(e,Z,o),p(e,Kt,o),g(Ie,e,o),p(e,eo,o),p(e,Ce,o),p(e,to,o),p(e,Je,o),p(e,oo,o),g(Ue,e,o),p(e,no,o),p(e,ze,o),p(e,ao,o),g(ie,e,o),p(e,so,o),g(je,e,o),p(e,ro,o),g(Fe,e,o),p(e,lo,o),g(We,e,o),p(e,io,o),g(Le,e,o),p(e,co,o),g(Ve,e,o),p(e,mo,o),g(Ze,e,o),p(e,po,o),p(e,C,o),g(Pe,C,null),n(C,jo),n(C,_t),n(C,Fo),n(C,Mt),n(C,Wo),n(C,Tt),n(C,Lo),g(ce,C,null),p(e,ho,o),g(Ge,e,o),p(e,uo,o),p(e,G,o),g(Be,G,null),n(G,Vo),n(G,bt),n(G,Zo),g(de,G,null),n(G,Po),n(G,me),g(Ne,me,null),n(me,Go),n(me,yt),p(e,go,o),g(qe,e,o),p(e,fo,o),p(e,J,o),g(Re,J,null),n(J,Bo),n(J,wt),n(J,No),n(J,pe),g(Xe,pe,null),n(pe,qo),n(pe,vt),n(J,Ro),n(J,he),g(Ee,he,null),n(he,Xo),n(he,kt),n(J,Eo),n(J,oe),g(He,oe,null),n(oe,Ho),n(oe,xt),n(oe,Qo),n(oe,$t),p(e,_o,o),g(Qe,e,o),p(e,Mo,o),p(e,U,o),g(Ye,U,null),n(U,Yo),n(U,It),n(U,So),n(U,Ct),n(U,Do),n(U,Jt),n(U,Ao),n(U,Q),g(Se,Q,null),n(Q,Oo),n(Q,Ut),n(Q,Ko),g(ue,Q,null),n(Q,en),g(ge,Q,null),p(e,To,o),g(De,e,o),p(e,bo,o),p(e,z,o),g(Ae,z,null),n(z,tn),n(z,zt),n(z,on),n(z,jt),n(z,nn),n(z,Ft),n(z,an),n(z,Y),g(Oe,Y,null),n(Y,sn),n(Y,Wt),n(Y,rn),g(fe,Y,null),n(Y,ln),g(_e,Y,null),p(e,yo,o),g(Ke,e,o),p(e,wo,o),p(e,j,o),g(et,j,null),n(j,cn),n(j,Lt),n(j,dn),n(j,Vt),n(j,mn),n(j,Zt),n(j,pn),n(j,S),g(tt,S,null),n(S,hn),n(S,Pt),n(S,un),g(Me,S,null),n(S,gn),g(Te,S,null),p(e,vo,o),g(ot,e,o),p(e,ko,o),p(e,F,o),g(nt,F,null),n(F,fn),n(F,Gt),n(F,_n),n(F,Bt),n(F,Mn),n(F,Nt),n(F,Tn),n(F,ne),g(at,ne,null),n(ne,bn),n(ne,qt),n(ne,yn),g(be,ne,null),p(e,xo,o),g(st,e,o),p(e,$o,o),p(e,W,o),g(rt,W,null),n(W,wn),n(W,Rt),n(W,vn),n(W,Xt),n(W,kn),n(W,Et),n(W,xn),n(W,D),g(lt,D,null),n(D,$n),n(D,Ht),n(D,In),g(ye,D,null),n(D,Cn),g(we,D,null),p(e,Io,o),g(it,e,o),p(e,Co,o),p(e,L,o),g(ct,L,null),n(L,Jn),n(L,Qt),n(L,Un),n(L,Yt),n(L,zn),n(L,St),n(L,jn),n(L,A),g(dt,A,null),n(A,Fn),n(A,Dt),n(A,Wn),g(ve,A,null),n(A,Ln),g(ke,A,null),p(e,Jo,o),g(mt,e,o),p(e,Uo,o),p(e,Ot,o),zo=!0},p(e,[o]){const B={};o&2&&(B.$$scope={dirty:o,ctx:e}),ie.$set(B);const O={};o&2&&(O.$$scope={dirty:o,ctx:e}),ce.$set(O);const pt={};o&2&&(pt.$$scope={dirty:o,ctx:e}),de.$set(pt);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),ue.$set(N);const ht={};o&2&&(ht.$$scope={dirty:o,ctx:e}),ge.$set(ht);const ut={};o&2&&(ut.$$scope={dirty:o,ctx:e}),fe.$set(ut);const le={};o&2&&(le.$$scope={dirty:o,ctx:e}),_e.$set(le);const q={};o&2&&(q.$$scope={dirty:o,ctx:e}),Me.$set(q);const K={};o&2&&(K.$$scope={dirty:o,ctx:e}),Te.$set(K);const R={};o&2&&(R.$$scope={dirty:o,ctx:e}),be.$set(R);const ee={};o&2&&(ee.$$scope={dirty:o,ctx:e}),ye.$set(ee);const X={};o&2&&(X.$$scope={dirty:o,ctx:e}),we.$set(X);const te={};o&2&&(te.$$scope={dirty:o,ctx:e}),ve.$set(te);const E={};o&2&&(E.$$scope={dirty:o,ctx:e}),ke.$set(E)},i(e){zo||(f(H.$$.fragment,e),f(Ie.$$.fragment,e),f(Ue.$$.fragment,e),f(ie.$$.fragment,e),f(je.$$.fragment,e),f(Fe.$$.fragment,e),f(We.$$.fragment,e),f(Le.$$.fragment,e),f(Ve.$$.fragment,e),f(Ze.$$.fragment,e),f(Pe.$$.fragment,e),f(ce.$$.fragment,e),f(Ge.$$.fragment,e),f(Be.$$.fragment,e),f(de.$$.fragment,e),f(Ne.$$.fragment,e),f(qe.$$.fragment,e),f(Re.$$.fragment,e),f(Xe.$$.fragment,e),f(Ee.$$.fragment,e),f(He.$$.fragment,e),f(Qe.$$.fragment,e),f(Ye.$$.fragment,e),f(Se.$$.fragment,e),f(ue.$$.fragment,e),f(ge.$$.fragment,e),f(De.$$.fragment,e),f(Ae.$$.fragment,e),f(Oe.$$.fragment,e),f(fe.$$.fragment,e),f(_e.$$.fragment,e),f(Ke.$$.fragment,e),f(et.$$.fragment,e),f(tt.$$.fragment,e),f(Me.$$.fragment,e),f(Te.$$.fragment,e),f(ot.$$.fragment,e),f(nt.$$.fragment,e),f(at.$$.fragment,e),f(be.$$.fragment,e),f(st.$$.fragment,e),f(rt.$$.fragment,e),f(lt.$$.fragment,e),f(ye.$$.fragment,e),f(we.$$.fragment,e),f(it.$$.fragment,e),f(ct.$$.fragment,e),f(dt.$$.fragment,e),f(ve.$$.fragment,e),f(ke.$$.fragment,e),f(mt.$$.fragment,e),zo=!0)},o(e){_(H.$$.fragment,e),_(Ie.$$.fragment,e),_(Ue.$$.fragment,e),_(ie.$$.fragment,e),_(je.$$.fragment,e),_(Fe.$$.fragment,e),_(We.$$.fragment,e),_(Le.$$.fragment,e),_(Ve.$$.fragment,e),_(Ze.$$.fragment,e),_(Pe.$$.fragment,e),_(ce.$$.fragment,e),_(Ge.$$.fragment,e),_(Be.$$.fragment,e),_(de.$$.fragment,e),_(Ne.$$.fragment,e),_(qe.$$.fragment,e),_(Re.$$.fragment,e),_(Xe.$$.fragment,e),_(Ee.$$.fragment,e),_(He.$$.fragment,e),_(Qe.$$.fragment,e),_(Ye.$$.fragment,e),_(Se.$$.fragment,e),_(ue.$$.fragment,e),_(ge.$$.fragment,e),_(De.$$.fragment,e),_(Ae.$$.fragment,e),_(Oe.$$.fragment,e),_(fe.$$.fragment,e),_(_e.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(Me.$$.fragment,e),_(Te.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(at.$$.fragment,e),_(be.$$.fragment,e),_(st.$$.fragment,e),_(rt.$$.fragment,e),_(lt.$$.fragment,e),_(ye.$$.fragment,e),_(we.$$.fragment,e),_(it.$$.fragment,e),_(ct.$$.fragment,e),_(dt.$$.fragment,e),_(ve.$$.fragment,e),_(ke.$$.fragment,e),_(mt.$$.fragment,e),zo=!1},d(e){e&&(l(b),l(i),l(m),l(y),l(w),l(k),l(Z),l(Kt),l(eo),l(Ce),l(to),l(Je),l(oo),l(no),l(ze),l(ao),l(so),l(ro),l(lo),l(io),l(co),l(mo),l(po),l(C),l(ho),l(uo),l(G),l(go),l(fo),l(J),l(_o),l(Mo),l(U),l(To),l(bo),l(z),l(yo),l(wo),l(j),l(vo),l(ko),l(F),l(xo),l($o),l(W),l(Io),l(Co),l(L),l(Jo),l(Uo),l(Ot)),l(t),M(H,e),M(Ie,e),M(Ue,e),M(ie,e),M(je,e),M(Fe,e),M(We,e),M(Le,e),M(Ve,e),M(Ze,e),M(Pe),M(ce),M(Ge,e),M(Be),M(de),M(Ne),M(qe,e),M(Re),M(Xe),M(Ee),M(He),M(Qe,e),M(Ye),M(Se),M(ue),M(ge),M(De,e),M(Ae),M(Oe),M(fe),M(_e),M(Ke,e),M(et),M(tt),M(Me),M(Te),M(ot,e),M(nt),M(at),M(be),M(st,e),M(rt),M(lt),M(ye),M(we),M(it,e),M(ct),M(dt),M(ve),M(ke),M(mt,e)}}}const Na='{"title":"Mllama","local":"mllama","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage Tips","local":"usage-tips","sections":[],"depth":2},{"title":"Usage Example","local":"usage-example","sections":[{"title":"Instruct model","local":"instruct-model","sections":[],"depth":4},{"title":"Base model","local":"base-model","sections":[],"depth":4}],"depth":2},{"title":"MllamaConfig","local":"transformers.MllamaConfig","sections":[],"depth":2},{"title":"MllamaProcessor","local":"transformers.MllamaProcessor","sections":[],"depth":2},{"title":"MllamaImageProcessor","local":"transformers.MllamaImageProcessor","sections":[],"depth":2},{"title":"MllamaForConditionalGeneration","local":"transformers.MllamaForConditionalGeneration","sections":[],"depth":2},{"title":"MllamaForCausalLM","local":"transformers.MllamaForCausalLM","sections":[],"depth":2},{"title":"MllamaTextModel","local":"transformers.MllamaTextModel","sections":[],"depth":2},{"title":"MllamaModel","local":"transformers.MllamaModel","sections":[],"depth":2},{"title":"MllamaForCausalLM","local":"transformers.MllamaForCausalLM","sections":[],"depth":2},{"title":"MllamaVisionModel","local":"transformers.MllamaVisionModel","sections":[],"depth":2}],"depth":1}';function qa(v){return ya(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Da extends wa{constructor(t){super(),va(this,t,qa,Ba,ba,{})}}export{Da as component};
