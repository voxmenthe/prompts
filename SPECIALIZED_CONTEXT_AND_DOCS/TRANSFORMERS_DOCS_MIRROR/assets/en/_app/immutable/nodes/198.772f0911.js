import{s as Os,o as Ks,n as x}from"../chunks/scheduler.18a86fab.js";import{S as ea,i as ta,g as c,s as r,r as g,A as na,h as d,f as s,c as i,j as J,x as b,u as h,k as U,l as oa,y as a,a as m,v as u,d as f,t as _,w as y}from"../chunks/index.98837b22.js";import{T as pn}from"../chunks/Tip.77304350.js";import{D as j}from"../chunks/Docstring.a1ef7999.js";import{C as q}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ct}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as A,E as sa}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as aa,a as cs}from"../chunks/HfOption.6641485e.js";function ra(v){let t,M="Click on the Gemma 3 models in the right sidebar for more examples of how to apply Gemma to different vision and language tasks.";return{c(){t=c("p"),t.textContent=M},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-1vrs0gl"&&(t.textContent=M)},m(n,p){m(n,t,p)},p:x,d(n){n&&s(t)}}}function ia(v){let t,M;return t=new q({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJpbWFnZS10ZXh0LXRvLXRleHQlMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmdvb2dsZSUyRmdlbW1hLTMtNGItcHQlMjIlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUwQSklMEFwaXBlbGluZSglMEElMjAlMjAlMjAlMjAlMjJodHRwcyUzQSUyRiUyRmh1Z2dpbmdmYWNlLmNvJTJGZGF0YXNldHMlMkZodWdnaW5nZmFjZSUyRmRvY3VtZW50YXRpb24taW1hZ2VzJTJGcmVzb2x2ZSUyRm1haW4lMkZwaXBlbGluZS1jYXQtY2hvbmsuanBlZyUyMiUyQyUwQSUyMCUyMCUyMCUyMHRleHQlM0QlMjIlM0NzdGFydF9vZl9pbWFnZSUzRSUyMFdoYXQlMjBpcyUyMHNob3duJTIwaW4lMjB0aGlzJTIwaW1hZ2UlM0YlMjIlMEEp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;image-text-to-text&quot;</span>,
    model=<span class="hljs-string">&quot;google/gemma-3-4b-pt&quot;</span>,
    device=<span class="hljs-number">0</span>,
    dtype=torch.bfloat16
)
pipeline(
    <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>,
    text=<span class="hljs-string">&quot;&lt;start_of_image&gt; What is shown in this image?&quot;</span>
)`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,p){u(t,n,p),M=!0},p:x,i(n){M||(f(t.$$.fragment,n),M=!0)},o(n){_(t.$$.fragment,n),M=!1},d(n){y(t,n)}}}function la(v){let t,M;return t=new q({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEdlbW1hM0ZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJnb29nbGUlMkZnZW1tYS0zLTRiLWl0JTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMEEpJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGZ2VtbWEtMy00Yi1pdCUyMiUyQyUwQSUyMCUyMCUyMCUyMHBhZGRpbmdfc2lkZSUzRCUyMmxlZnQlMjIlMEEpJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwJTIyc3lzdGVtJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMllvdSUyMGFyZSUyMGElMjBoZWxwZnVsJTIwYXNzaXN0YW50LiUyMiU3RCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU1RCUwQSUyMCUyMCUyMCUyMCU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUyMCUyMnVybCUyMiUzQSUyMCUyMmh0dHBzJTNBJTJGJTJGaHVnZ2luZ2ZhY2UuY28lMkZkYXRhc2V0cyUyRmh1Z2dpbmdmYWNlJTJGZG9jdW1lbnRhdGlvbi1pbWFnZXMlMkZyZXNvbHZlJTJGbWFpbiUyRnBpcGVsaW5lLWNhdC1jaG9uay5qcGVnJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyV2hhdCUyMGlzJTIwc2hvd24lMjBpbiUyMHRoaXMlMjBpbWFnZSUzRiUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU1RCUwQSUyMCUyMCUyMCUyMCU3RCUyQyUwQSU1RCUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3Nvci5hcHBseV9jaGF0X3RlbXBsYXRlKCUwQSUyMCUyMCUyMCUyMG1lc3NhZ2VzJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMEEpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNENTAlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludChwcm9jZXNzb3IuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, Gemma3ForConditionalGeneration

model = Gemma3ForConditionalGeneration.from_pretrained(
    <span class="hljs-string">&quot;google/gemma-3-4b-it&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
processor = AutoProcessor.from_pretrained(
    <span class="hljs-string">&quot;google/gemma-3-4b-it&quot;</span>,
    padding_side=<span class="hljs-string">&quot;left&quot;</span>
)

messages = [
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>}
        ]
    },
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;What is shown in this image?&quot;</span>},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">True</span>,
    return_dict=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=<span class="hljs-number">50</span>, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(processor.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,p){u(t,n,p),M=!0},p:x,i(n){M||(f(t.$$.fragment,n),M=!0)},o(n){_(t.$$.fragment,n),M=!1},d(n){y(t,n)}}}function ca(v){let t,M;return t=new q({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQtZ2VuZXJhdGlvbiUyMC0tbW9kZWwlMjBnb29nbGUlMkZnZW1tYS0zLTFiLXB0JTIwLS1kZXZpY2UlMjAw",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span> | transformers run --task text-generation --model google/gemma-3-1b-pt --device 0',wrap:!1}}),{c(){g(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,p){u(t,n,p),M=!0},p:x,i(n){M||(f(t.$$.fragment,n),M=!0)},o(n){_(t.$$.fragment,n),M=!1},d(n){y(t,n)}}}function da(v){let t,M,n,p,w,l;return t=new cs({props:{id:"usage",option:"Pipeline",$$slots:{default:[ia]},$$scope:{ctx:v}}}),n=new cs({props:{id:"usage",option:"AutoModel",$$slots:{default:[la]},$$scope:{ctx:v}}}),w=new cs({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ca]},$$scope:{ctx:v}}}),{c(){g(t.$$.fragment),M=r(),g(n.$$.fragment),p=r(),g(w.$$.fragment)},l(T){h(t.$$.fragment,T),M=i(T),h(n.$$.fragment,T),p=i(T),h(w.$$.fragment,T)},m(T,I){u(t,T,I),m(T,M,I),u(n,T,I),m(T,p,I),u(w,T,I),l=!0},p(T,I){const gn={};I&2&&(gn.$$scope={dirty:I,ctx:T}),t.$set(gn);const Ie={};I&2&&(Ie.$$scope={dirty:I,ctx:T}),n.$set(Ie);const H={};I&2&&(H.$$scope={dirty:I,ctx:T}),w.$set(H)},i(T){l||(f(t.$$.fragment,T),f(n.$$.fragment,T),f(w.$$.fragment,T),l=!0)},o(T){_(t.$$.fragment,T),_(n.$$.fragment,T),_(w.$$.fragment,T),l=!1},d(T){T&&(s(M),s(p)),y(t,T),y(n,T),y(w,T)}}}function ma(v){let t,M;return t=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdlbW1hM1RleHRNb2RlbCUyQyUyMEdlbW1hM1RleHRDb25maWclMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwR2VtbWEzVGV4dCUyMGdlbW1hM190ZXh0LTdiJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEdlbW1hM1RleHRDb25maWcoKSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBnZW1tYTNfdGV4dC03YiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzVGV4dE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Gemma3TextModel, Gemma3TextConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Gemma3Text gemma3_text-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Gemma3TextConfig()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the gemma3_text-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3TextModel(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,p){u(t,n,p),M=!0},p:x,i(n){M||(f(t.$$.fragment,n),M=!0)},o(n){_(t.$$.fragment,n),M=!1},d(n){y(t,n)}}}function pa(v){let t,M="Example:",n,p,w;return p=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdlbW1hM0ZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyMEdlbW1hM0NvbmZpZyUyQyUyMFNpZ2xpcFZpc2lvbkNvbmZpZyUyQyUyMEdlbW1hM1RleHRDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwU2lnbGlwLWxpa2UlMjB2aXNpb24lMjBjb25maWclMEF2aXNpb25fY29uZmlnJTIwJTNEJTIwU2lnbGlwVmlzaW9uQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwR2VtbWEzJTIwVGV4dCUyMGNvbmZpZyUwQXRleHRfY29uZmlnJTIwJTNEJTIwR2VtbWEzVGV4dENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEdlbW1hMyUyMGdlbW1hLTMtNGIlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwR2VtbWEzQ29uZmlnKHZpc2lvbl9jb25maWclMkMlMjB0ZXh0X2NvbmZpZyklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwZ2VtbWEtMy00YiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzVGV4dENvbmZpZyhjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Gemma3ForConditionalGeneration, Gemma3Config, SiglipVisionConfig, Gemma3TextConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Siglip-like vision config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>vision_config = SiglipVisionConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Gemma3 Text config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>text_config = Gemma3TextConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Gemma3 gemma-3-4b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Gemma3Config(vision_config, text_config)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the gemma-3-4b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3TextConfig(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,n=r(),g(p.$$.fragment)},l(l){t=d(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=M),n=i(l),h(p.$$.fragment,l)},m(l,T){m(l,t,T),m(l,n,T),u(p,l,T),w=!0},p:x,i(l){w||(f(p.$$.fragment,l),w=!0)},o(l){_(p.$$.fragment,l),w=!1},d(l){l&&(s(t),s(n)),y(p,l)}}}function ga(v){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(n,p){m(n,t,p)},p:x,d(n){n&&s(t)}}}function ha(v){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(n,p){m(n,t,p)},p:x,d(n){n&&s(t)}}}function ua(v){let t,M="Example:",n,p,w;return p=new q({props:{code:"ZnJvbSUyMFBJTCUyMGltcG9ydCUyMEltYWdlJTBBaW1wb3J0JTIwcmVxdWVzdHMlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEdlbW1hM0ZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZnZW1tYTMyLTNiLW1peC0yMjQlMjIpJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZ2VtbWEzMi0zYi1taXgtMjI0JTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMldoZXJlJTIwaXMlMjB0aGUlMjBjYXQlMjBzdGFuZGluZyUzRiUyMiUwQXVybCUyMCUzRCUyMCUyMmh0dHBzJTNBJTJGJTJGaHVnZ2luZ2ZhY2UuY28lMkZkYXRhc2V0cyUyRmh1Z2dpbmdmYWNlJTJGZG9jdW1lbnRhdGlvbi1pbWFnZXMlMkZyZXNvbHZlJTJGbWFpbiUyRnBpcGVsaW5lLWNhdC1jaG9uay5qcGVnJTIyJTBBaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKGltYWdlcyUzRGltYWdlJTJDJTIwdGV4dCUzRHByb21wdCUyQyUyMCUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyklMEFwcm9jZXNzb3IuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, Gemma3ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;google/gemma32-3b-mix-224&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;google/gemma32-3b-mix-224&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Where is the cat standing?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, text=prompt,  return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(**inputs,)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Where is the cat standing?\\nsnow&quot;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,n=r(),g(p.$$.fragment)},l(l){t=d(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=M),n=i(l),h(p.$$.fragment,l)},m(l,T){m(l,t,T),m(l,n,T),u(p,l,T),w=!0},p:x,i(l){w||(f(p.$$.fragment,l),w=!0)},o(l){_(p.$$.fragment,l),w=!1},d(l){l&&(s(t),s(n)),y(p,l)}}}function fa(v){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(n,p){m(n,t,p)},p:x,d(n){n&&s(t)}}}function _a(v){let t,M="Example:",n,p,w;return p=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHZW1tYTNGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmdlbW1hLTItOWIlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZ2VtbWEtMi05YiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJXaGF0JTIwaXMlMjB5b3VyJTIwZmF2b3JpdGUlMjBjb25kaW1lbnQlM0YlMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNEMzApJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Gemma3ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3ForCausalLM.from_pretrained(<span class="hljs-string">&quot;google/gemma-2-9b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/gemma-2-9b&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;What is your favorite condiment?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;What is your favorite condiment?&quot;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,n=r(),g(p.$$.fragment)},l(l){t=d(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=M),n=i(l),h(p.$$.fragment,l)},m(l,T){m(l,t,T),m(l,n,T),u(p,l,T),w=!0},p:x,i(l){w||(f(p.$$.fragment,l),w=!0)},o(l){_(p.$$.fragment,l),w=!1},d(l){l&&(s(t),s(n)),y(p,l)}}}function ya(v){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(n,p){m(n,t,p)},p:x,d(n){n&&s(t)}}}function Ma(v){let t,M="Example:",n,p,w;return p=new q({props:{code:"ZnJvbSUyMFBJTCUyMGltcG9ydCUyMEltYWdlJTBBaW1wb3J0JTIwcmVxdWVzdHMlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEdlbW1hM0ZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZnZW1tYS0zLTRiLWl0JTIyKSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmdlbW1hLTMtNGItaXQlMjIpJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwJTIyc3lzdGVtJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMllvdSUyMGFyZSUyMGElMjBoZWxwZnVsJTIwYXNzaXN0YW50LiUyMiU3RCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU1RCUwQSUyMCUyMCUyMCUyMCU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUyMCUyMnVybCUyMiUzQSUyMCUyMmh0dHBzJTNBJTJGJTJGaHVnZ2luZ2ZhY2UuY28lMkZkYXRhc2V0cyUyRmh1Z2dpbmdmYWNlJTJGZG9jdW1lbnRhdGlvbi1pbWFnZXMlMkZyZXNvbHZlJTJGbWFpbiUyRnBpcGVsaW5lLWNhdC1jaG9uay5qcGVnJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyV2hlcmUlMjBpcyUyMHRoZSUyMGNhdCUyMHN0YW5kaW5nJTNGJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTVEJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBJTIwJTIwJTIwJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUwQSklMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzKSUwQXByb2Nlc3Nvci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, Gemma3ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;google/gemma-3-4b-it&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;google/gemma-3-4b-it&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [
<span class="hljs-meta">... </span>    {
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>,
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;content&quot;</span>: [
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>}
<span class="hljs-meta">... </span>        ]
<span class="hljs-meta">... </span>    },
<span class="hljs-meta">... </span>    {
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>},
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;Where is the cat standing?&quot;</span>},
<span class="hljs-meta">... </span>        ]
<span class="hljs-meta">... </span>    },
<span class="hljs-meta">... </span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor.apply_chat_template(
<span class="hljs-meta">... </span>    messages,
<span class="hljs-meta">... </span>    tokenize=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    return_dict=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>    add_generation_prompt=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;user\\nYou are a helpful assistant.\\n\\n\\n\\n\\n\\nWhere is the cat standing?\\nmodel\\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to&quot;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,n=r(),g(p.$$.fragment)},l(l){t=d(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=M),n=i(l),h(p.$$.fragment,l)},m(l,T){m(l,t,T),m(l,n,T),u(p,l,T),w=!0},p:x,i(l){w||(f(p.$$.fragment,l),w=!0)},o(l){_(p.$$.fragment,l),w=!1},d(l){l&&(s(t),s(n)),y(p,l)}}}function Ta(v){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(n,p){m(n,t,p)},p:x,d(n){n&&s(t)}}}function ba(v){let t,M="Example of single-label classification:",n,p,w;return p=new q({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEdlbW1hM0ZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZnZW1tYS0zLTRiJTIyKSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZ2VtbWEtMy00YiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZ2VtbWEtMy00YiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Gemma3ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/gemma-3-4b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/gemma-3-4b&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/gemma-3-4b&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,n=r(),g(p.$$.fragment)},l(l){t=d(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-ykxpe4"&&(t.textContent=M),n=i(l),h(p.$$.fragment,l)},m(l,T){m(l,t,T),m(l,n,T),u(p,l,T),w=!0},p:x,i(l){w||(f(p.$$.fragment,l),w=!0)},o(l){_(p.$$.fragment,l),w=!1},d(l){l&&(s(t),s(n)),y(p,l)}}}function wa(v){let t,M="Example of multi-label classification:",n,p,w;return p=new q({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEdlbW1hM0ZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZnZW1tYS0zLTRiJTIyKSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZ2VtbWEtMy00YiUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGZ2VtbWEtMy00YiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Gemma3ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/gemma-3-4b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/gemma-3-4b&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma3ForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/gemma-3-4b&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,n=r(),g(p.$$.fragment)},l(l){t=d(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-1l8e32d"&&(t.textContent=M),n=i(l),h(p.$$.fragment,l)},m(l,T){m(l,t,T),m(l,n,T),u(p,l,T),w=!0},p:x,i(l){w||(f(p.$$.fragment,l),w=!0)},o(l){_(p.$$.fragment,l),w=!1},d(l){l&&(s(t),s(n)),y(p,l)}}}function va(v){let t,M,n,p,w,l="<em>This model was released on 2025-03-25 and added to Hugging Face Transformers on 2025-03-12.</em>",T,I,gn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ie,H,_n,je,ds='<a href="https://huggingface.co/papers/2503.19786" rel="nofollow">Gemma 3</a> is a multimodal model with pretrained and instruction-tuned variants, available in 1B, 13B, and 27B parameters. The architecture is mostly the same as the previous Gemma versions. The key differences are alternating 5 local sliding window self-attention layers for every global self-attention layer, support for a longer context length of 128K tokens, and a <a href="./siglip">SigLip</a> encoder that can “pan &amp; scan” high-resolution images to prevent information from disappearing in high resolution images or images with non-square aspect ratios.',yn,Ce,ms="The instruction-tuned variant was post-trained with knowledge distillation and reinforcement learning.",Mn,Ge,ps='You can find all the original Gemma 3 checkpoints under the <a href="https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d" rel="nofollow">Gemma 3</a> release.',Tn,oe,bn,xe,gs='The example below demonstrates how to generate text based on an image with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',wn,se,vn,ke,hs='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Jn,$e,us='The example below uses <a href="../quantization/torchao">torchao</a> to only quantize the weights to int4.',Un,ze,In,We,fs='Use the <a href="https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139" rel="nofollow">AttentionMaskVisualizer</a> to better understand what tokens the model can and cannot attend to.',jn,Ze,Cn,ae,_s='<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/gemma-3-attn-mask.png"/>',Gn,qe,xn,G,Gt,ys='<p>Use <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration">Gemma3ForConditionalGeneration</a> for image-and-text and image-only inputs.</p>',oo,Fe,xt,Ms="Gemma 3 supports multiple input images, but make sure the images are correctly batched before passing them to the processor. Each batch should be a list of one or more images.",so,Be,ao,kt,Ts="<p>Text passed to the processor should have a <code>&lt;start_of_image&gt;</code> token wherever an image should be inserted.</p>",ro,$t,bs='<p>The processor has its own <a href="/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template">apply_chat_template()</a> method to convert chat messages to model inputs.</p>',io,Ne,zt,ws="By default, images aren’t cropped and only the base image is forwarded to the model. In high resolution images or images with non-square aspect ratios, artifacts can result because the vision encoder uses a fixed resolution of 896x896. To prevent these artifacts and improve performance during inference, set <code>do_pan_and_scan=True</code> to crop the image into multiple smaller patches and concatenate them with the base image embedding. You can disable pan and scan for faster inference.",lo,Re,co,Ve,Wt,vs='For Gemma-3 1B checkpoint trained in text-only mode, use <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModelForCausalLM">AutoModelForCausalLM</a> instead.',mo,Ee,kn,Xe,$n,N,Pe,po,Zt,Js="Constructs a SigLIP image processor.",go,re,Qe,ho,qt,Us=`Pan and Scan and image, by cropping into smaller images when the aspect ratio exceeds
minimum allowed ratio.`,uo,ie,Ae,fo,Ft,Is="Preprocess an image or batch of images.",zn,Se,Wn,R,Le,_o,Bt,js="Constructs a fast Gemma3 image processor.",yo,le,Ye,Mo,Nt,Cs=`Pan and Scan an image, by cropping into smaller images when the aspect ratio exceeds
minimum allowed ratio.`,To,Rt,He,Zn,De,qn,Oe,Ke,Fn,et,Bn,S,tt,bo,Vt,Gs=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextModel">Gemma3TextModel</a>. It is used to instantiate an Gemma3Text
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Gemma3Text-7B.
e.g. <a href="https://huggingface.co/google/gemma3_text-7b" rel="nofollow">google/gemma3_text-7b</a>
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,wo,ce,Nn,nt,Rn,k,ot,vo,Et,xs=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration">Gemma3ForConditionalGeneration</a>. It is used to instantiate an
Gemma3ForConditionalGeneration according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the PaliGemma-2B.`,Jo,Xt,ks='e.g. <a href="https://huggingface.co/google/gemma-3-4b" rel="nofollow">google/gemma-3-4b</a>',Uo,Pt,$s=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Io,de,Vn,st,En,$,at,jo,Qt,zs="The bare Gemma3 Text Model outputting raw hidden-states without any specific head on to.",Co,At,Ws=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Go,St,Zs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xo,D,rt,ko,Lt,qs='The <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextModel">Gemma3TextModel</a> forward method, overrides the <code>__call__</code> special method.',$o,me,Xn,it,Pn,C,lt,zo,Yt,Fs="The Base Gemma3 model which consists of a vision backbone and a language model withou language modeling head.,",Wo,Ht,Bs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Zo,Dt,Ns=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,qo,X,ct,Fo,Ot,Rs='The <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Model">Gemma3Model</a> forward method, overrides the <code>__call__</code> special method.',Bo,pe,No,ge,Ro,he,dt,Vo,Kt,Vs="Projects the last hidden state from the vision model into language model space.",Eo,ue,mt,Xo,en,Es=`Obtains multimodal placeholder mask from <code>input_ids</code> or <code>inputs_embeds</code>, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.`,Qn,pt,An,z,gt,Po,tn,Xs="The Gemma3 Model for causal language modeling.",Qo,nn,Ps=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ao,on,Qs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,So,P,ht,Lo,sn,As='The <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForCausalLM">Gemma3ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Yo,fe,Ho,_e,Sn,ut,Ln,W,ft,Do,an,Ss="The Base Gemma3 model which consists of a vision backbone and a language model without language modeling head.,",Oo,rn,Ls=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ko,ln,Ys=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,es,Q,_t,ts,cn,Hs='The <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForConditionalGeneration">Gemma3ForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',ns,ye,os,Me,Yn,yt,Hn,te,Mt,ss,F,Tt,as,dn,Ds='The <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ForSequenceClassification">Gemma3ForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',rs,Te,is,be,ls,we,Dn,bt,On,hn,Kn;return H=new A({props:{title:"Gemma 3",local:"gemma-3",headingTag:"h1"}}),oe=new pn({props:{warning:!1,$$slots:{default:[ra]},$$scope:{ctx:v}}}),se=new aa({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[da]},$$scope:{ctx:v}}}),ze=new q({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMHRvcmNoYW8lMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBUb3JjaEFvQ29uZmlnJTJDJTIwR2VtbWEzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTJDJTIwQXV0b1Byb2Nlc3NvciUwQSUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBUb3JjaEFvQ29uZmlnKCUyMmludDRfd2VpZ2h0X29ubHklMjIlMkMlMjBncm91cF9zaXplJTNEMTI4KSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJnb29nbGUlMkZnZW1tYS0zLTI3Yi1pdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RxdWFudGl6YXRpb25fY29uZmlnJTBBKSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRmdlbW1hLTMtMjdiLWl0JTIyJTJDJTBBJTIwJTIwJTIwJTIwcGFkZGluZ19zaWRlJTNEJTIybGVmdCUyMiUwQSklMEElMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjAlMjJzeXN0ZW0lMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyWW91JTIwYXJlJTIwYSUyMGhlbHBmdWwlMjBhc3Npc3RhbnQuJTIyJTdEJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTJDJTIwJTIydXJsJTIyJTNBJTIwJTIyaHR0cHMlM0ElMkYlMkZodWdnaW5nZmFjZS5jbyUyRmRhdGFzZXRzJTJGaHVnZ2luZ2ZhY2UlMkZkb2N1bWVudGF0aW9uLWltYWdlcyUyRnJlc29sdmUlMkZtYWluJTJGcGlwZWxpbmUtY2F0LWNob25rLmpwZWclMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjAlMjJXaGF0JTIwaXMlMjBzaG93biUyMGluJTIwdGhpcyUyMGltYWdlJTNGJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTVEJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBJTIwJTIwJTIwJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUyQyUwQSkudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0Q1MCUyQyUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyKSUwQXByaW50KHByb2Nlc3Nvci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-comment"># pip install torchao</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TorchAoConfig, Gemma3ForConditionalGeneration, AutoProcessor

quantization_config = TorchAoConfig(<span class="hljs-string">&quot;int4_weight_only&quot;</span>, group_size=<span class="hljs-number">128</span>)
model = Gemma3ForConditionalGeneration.from_pretrained(
    <span class="hljs-string">&quot;google/gemma-3-27b-it&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(
    <span class="hljs-string">&quot;google/gemma-3-27b-it&quot;</span>,
    padding_side=<span class="hljs-string">&quot;left&quot;</span>
)

messages = [
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>}
        ]
    },
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;What is shown in this image?&quot;</span>},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">True</span>,
    return_dict=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=<span class="hljs-number">50</span>, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(processor.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),Ze=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycy51dGlscy5hdHRlbnRpb25fdmlzdWFsaXplciUyMGltcG9ydCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyJTBBJTBBdmlzdWFsaXplciUyMCUzRCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyKCUyMmdvb2dsZSUyRmdlbW1hLTMtNGItaXQlMjIpJTBBdmlzdWFsaXplciglMjIlM0NpbWclM0VXaGF0JTIwaXMlMjBzaG93biUyMGluJTIwdGhpcyUyMGltYWdlJTNGJTIyKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers.utils.attention_visualizer <span class="hljs-keyword">import</span> AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer(<span class="hljs-string">&quot;google/gemma-3-4b-it&quot;</span>)
visualizer(<span class="hljs-string">&quot;&lt;img&gt;What is shown in this image?&quot;</span>)`,wrap:!1}}),qe=new A({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Be=new q({props:{code:"dXJsX2NvdyUyMCUzRCUyMCUyMmh0dHBzJTNBJTJGJTJGbWVkaWEuaXN0b2NrcGhvdG8uY29tJTJGaWQlMkYxMTkyODY3NzUzJTJGcGhvdG8lMkZjb3ctaW4tYmVyY2hpZGEtYmVhY2gtc2luaXNjb2xhLmpwZyUzRnMlM0Q2MTJ4NjEyJTI2dyUzRDAlMjZrJTNEMjAlMjZjJTNEdjBoampuaXdzTU5mSlN1S1dadUluOHBzc21ENWg1YlNOMXBlQmQxQ21INCUzRCUyMiUwQXVybF9jYXQlMjAlM0QlMjAlMjJodHRwcyUzQSUyRiUyRmh1Z2dpbmdmYWNlLmNvJTJGZGF0YXNldHMlMkZodWdnaW5nZmFjZSUyRmRvY3VtZW50YXRpb24taW1hZ2VzJTJGcmVzb2x2ZSUyRm1haW4lMkZwaXBlbGluZS1jYXQtY2hvbmsuanBlZyUyMiUwQSUwQW1lc3NhZ2VzJTIwJTNEJTVCJTBBJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycm9sZSUyMiUzQSUyMCUyMnN5c3RlbSUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjAlMjJZb3UlMjBhcmUlMjBhJTIwaGVscGZ1bCUyMGFzc2lzdGFudC4lMjIlN0QlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMEElMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlMkMlMjAlMjJ1cmwlMjIlM0ElMjB1cmxfY293JTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTJDJTIwJTIydXJsJTIyJTNBJTIwdXJsX2NhdCU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMldoaWNoJTIwaW1hZ2UlMjBpcyUyMGN1dGVyJTNGJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTVE",highlighted:`url_cow = <span class="hljs-string">&quot;https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&amp;w=0&amp;k=20&amp;c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4=&quot;</span>
url_cat = <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>

messages =[
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>}
        ]
    },
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: url_cow},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: url_cat},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;Which image is cuter?&quot;</span>},
        ]
    },
]`,wrap:!1}}),Re=new q({props:{code:"aW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBJTIwJTIwJTIwJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUyQyUwQSUyQiUyMCUyMCUyMGRvX3Bhbl9hbmRfc2NhbiUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjApLnRvKG1vZGVsLmRldmljZSk=",highlighted:`inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors=&quot;pt&quot;,
    add_generation_prompt=True,
<span class="hljs-addition">+   do_pan_and_scan=True,</span>
    ).to(model.device)`,wrap:!1}}),Ee=new q({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGZ2VtbWEtMy0xYi1wdCUyMiUyQyUwQSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJnb29nbGUlMkZnZW1tYS0zLTFiLXB0JTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMEEpJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlBsYW50cyUyMGNyZWF0ZSUyMGVuZXJneSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dF9pZHMlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google/gemma-3-1b-pt&quot;</span>,
)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;google/gemma-3-1b-pt&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),Xe=new A({props:{title:"Gemma3ImageProcessor",local:"transformers.Gemma3ImageProcessor",headingTag:"h2"}}),Pe=new j({props:{name:"class transformers.Gemma3ImageProcessor",anchor:"transformers.Gemma3ImageProcessor",parameters:[{name:"do_resize",val:": bool = True"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"resample",val:": Resampling = <Resampling.BILINEAR: 2>"},{name:"do_rescale",val:": bool = True"},{name:"rescale_factor",val:": typing.Union[int, float] = 0.00392156862745098"},{name:"do_normalize",val:": bool = True"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"do_convert_rgb",val:": typing.Optional[bool] = True"},{name:"do_pan_and_scan",val:": typing.Optional[bool] = None"},{name:"pan_and_scan_min_crop_size",val:": typing.Optional[int] = None"},{name:"pan_and_scan_max_num_crops",val:": typing.Optional[int] = None"},{name:"pan_and_scan_min_ratio_to_activate",val:": typing.Optional[float] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma3ImageProcessor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the image&#x2019;s (height, width) dimensions to the specified <code>size</code>. Can be overridden by
<code>do_resize</code> in the <code>preprocess</code> method.`,name:"do_resize"},{anchor:"transformers.Gemma3ImageProcessor.size",description:`<strong>size</strong> (<code>dict[str, int]</code> <em>optional</em>, defaults to <code>{&quot;height&quot; -- 224, &quot;width&quot;: 224}</code>):
Size of the image after resizing. Can be overridden by <code>size</code> in the <code>preprocess</code> method.`,name:"size"},{anchor:"transformers.Gemma3ImageProcessor.resample",description:`<strong>resample</strong> (<code>PILImageResampling</code>, <em>optional</em>, defaults to <code>Resampling.BILINEAR</code>) &#x2014;
Resampling filter to use if resizing the image. Can be overridden by <code>resample</code> in the <code>preprocess</code> method.`,name:"resample"},{anchor:"transformers.Gemma3ImageProcessor.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to rescale the image by the specified scale <code>rescale_factor</code>. Can be overridden by <code>do_rescale</code> in
the <code>preprocess</code> method.`,name:"do_rescale"},{anchor:"transformers.Gemma3ImageProcessor.rescale_factor",description:`<strong>rescale_factor</strong> (<code>int</code> or <code>float</code>, <em>optional</em>, defaults to <code>1/255</code>) &#x2014;
Scale factor to use if rescaling the image. Can be overridden by <code>rescale_factor</code> in the <code>preprocess</code>
method.`,name:"rescale_factor"},{anchor:"transformers.Gemma3ImageProcessor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize the image by the specified mean and standard deviation. Can be overridden by
<code>do_normalize</code> in the <code>preprocess</code> method.`,name:"do_normalize"},{anchor:"transformers.Gemma3ImageProcessor.image_mean",description:`<strong>image_mean</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>[0.5, 0.5, 0.5]</code>) &#x2014;
Mean to use if normalizing the image. This is a float or list of floats the length of the number of
channels in the image. Can be overridden by the <code>image_mean</code> parameter in the <code>preprocess</code> method.`,name:"image_mean"},{anchor:"transformers.Gemma3ImageProcessor.image_std",description:`<strong>image_std</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>[0.5, 0.5, 0.5]</code>) &#x2014;
Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
number of channels in the image. Can be overridden by the <code>image_std</code> parameter in the <code>preprocess</code> method.
Can be overridden by the <code>image_std</code> parameter in the <code>preprocess</code> method.`,name:"image_std"},{anchor:"transformers.Gemma3ImageProcessor.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.Gemma3ImageProcessor.do_pan_and_scan",description:`<strong>do_pan_and_scan</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to apply <code>pan_and_scan</code> to images.`,name:"do_pan_and_scan"},{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan_min_crop_size",description:`<strong>pan_and_scan_min_crop_size</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Minimum size of each crop in pan and scan.`,name:"pan_and_scan_min_crop_size"},{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan_max_num_crops",description:`<strong>pan_and_scan_max_num_crops</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum number of crops per image in pan and scan.`,name:"pan_and_scan_max_num_crops"},{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan_min_ratio_to_activate",description:`<strong>pan_and_scan_min_ratio_to_activate</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Minimum aspect ratio to activate pan and scan.`,name:"pan_and_scan_min_ratio_to_activate"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/image_processing_gemma3.py#L53"}}),Qe=new j({props:{name:"pan_and_scan",anchor:"transformers.Gemma3ImageProcessor.pan_and_scan",parameters:[{name:"image",val:": ndarray"},{name:"pan_and_scan_min_crop_size",val:": int"},{name:"pan_and_scan_max_num_crops",val:": int"},{name:"pan_and_scan_min_ratio_to_activate",val:": float"},{name:"data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}],parametersDescription:[{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan.image",description:`<strong>image</strong> (<code>np.ndarray</code>) &#x2014;
Image to resize.`,name:"image"},{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan.pan_and_scan_min_crop_size",description:`<strong>pan_and_scan_min_crop_size</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Minimum size of each crop in pan and scan.`,name:"pan_and_scan_min_crop_size"},{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan.pan_and_scan_max_num_crops",description:`<strong>pan_and_scan_max_num_crops</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum number of crops per image in pan and scan.`,name:"pan_and_scan_max_num_crops"},{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan.pan_and_scan_min_ratio_to_activate",description:`<strong>pan_and_scan_min_ratio_to_activate</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Minimum aspect ratio to activate pan and scan.`,name:"pan_and_scan_min_ratio_to_activate"},{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan.data_format",description:`<strong>data_format</strong> (<code>str</code> or <code>ChannelDimension</code>, <em>optional</em>) &#x2014;
The channel dimension format of the image. If not provided, it will be the same as the input image.`,name:"data_format"},{anchor:"transformers.Gemma3ImageProcessor.pan_and_scan.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>) &#x2014;
The channel dimension format of the input image. If not provided, it will be inferred.`,name:"input_data_format"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/image_processing_gemma3.py#L132"}}),Ae=new j({props:{name:"preprocess",anchor:"transformers.Gemma3ImageProcessor.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"do_resize",val:": typing.Optional[bool] = None"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"resample",val:": Resampling = None"},{name:"do_rescale",val:": typing.Optional[bool] = None"},{name:"rescale_factor",val:": typing.Optional[float] = None"},{name:"do_normalize",val:": typing.Optional[bool] = None"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"data_format",val:": typing.Optional[transformers.image_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels_first'>"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"do_convert_rgb",val:": typing.Optional[bool] = None"},{name:"do_pan_and_scan",val:": typing.Optional[bool] = None"},{name:"pan_and_scan_min_crop_size",val:": typing.Optional[int] = None"},{name:"pan_and_scan_max_num_crops",val:": typing.Optional[int] = None"},{name:"pan_and_scan_min_ratio_to_activate",val:": typing.Optional[float] = None"}],parametersDescription:[{anchor:"transformers.Gemma3ImageProcessor.preprocess.images",description:`<strong>images</strong> (<code>ImageInput</code>) &#x2014;
Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
passing in images with pixel values between 0 and 1, set <code>do_rescale=False</code>.`,name:"images"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_resize</code>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>, defaults to <code>self.size</code>) &#x2014;
Size of the image after resizing.`,name:"size"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.resample</code>) &#x2014;
Resampling filter to use if resizing the image. This can be one of the enum <code>PILImageResampling</code>. Only
has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_rescale</code>) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>float</code>, <em>optional</em>, defaults to <code>self.rescale_factor</code>) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_normalize</code>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>self.image_mean</code>) &#x2014;
Image mean to use for normalization. Only has an effect if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_mean"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.image_std",description:`<strong>image_std</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>self.image_std</code>) &#x2014;
Image standard deviation to use for normalization. Only has an effect if <code>do_normalize</code> is set to
<code>True</code>.`,name:"image_std"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <code>TensorType</code>, <em>optional</em>) &#x2014;
The type of tensors to return. Can be one of:<ul>
<li>Unset: Return a list of <code>np.ndarray</code>.</li>
<li><code>TensorType.TENSORFLOW</code> or <code>&apos;tf&apos;</code>: Return a batch of type <code>tf.Tensor</code>.</li>
<li><code>TensorType.PYTORCH</code> or <code>&apos;pt&apos;</code>: Return a batch of type <code>torch.Tensor</code>.</li>
<li><code>TensorType.NUMPY</code> or <code>&apos;np&apos;</code>: Return a batch of type <code>np.ndarray</code>.</li>
<li><code>TensorType.JAX</code> or <code>&apos;jax&apos;</code>: Return a batch of type <code>jax.numpy.ndarray</code>.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.data_format",description:`<strong>data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>, defaults to <code>ChannelDimension.FIRST</code>) &#x2014;
The channel dimension format for the output image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li>Unset: Use the channel dimension format of the input image.</li>
</ul>`,name:"data_format"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_convert_rgb</code>) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.do_pan_and_scan",description:`<strong>do_pan_and_scan</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_pan_and_scan</code>) &#x2014;
Whether to apply <code>pan_and_scan</code> to images.`,name:"do_pan_and_scan"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.pan_and_scan_min_crop_size",description:`<strong>pan_and_scan_min_crop_size</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.pan_and_scan_min_crop_size</code>) &#x2014;
Minimum size of each crop in pan and scan.`,name:"pan_and_scan_min_crop_size"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.pan_and_scan_max_num_crops",description:`<strong>pan_and_scan_max_num_crops</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.pan_and_scan_max_num_crops</code>) &#x2014;
Maximum number of crops per image in pan and scan.`,name:"pan_and_scan_max_num_crops"},{anchor:"transformers.Gemma3ImageProcessor.preprocess.pan_and_scan_min_ratio_to_activate",description:`<strong>pan_and_scan_min_ratio_to_activate</strong> (<code>float</code>, <em>optional</em>, defaults to <code>self.pan_and_scan_min_ratio_to_activate</code>) &#x2014;
Minimum aspect ratio to activate pan and scan.`,name:"pan_and_scan_min_ratio_to_activate"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/image_processing_gemma3.py#L239"}}),Se=new A({props:{title:"Gemma3ImageProcessorFast",local:"transformers.Gemma3ImageProcessorFast",headingTag:"h2"}}),Le=new j({props:{name:"class transformers.Gemma3ImageProcessorFast",anchor:"transformers.Gemma3ImageProcessorFast",parameters:[{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.gemma3.image_processing_gemma3_fast.Gemma3FastImageProcessorKwargs]"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/image_processing_gemma3_fast.py#L75"}}),Ye=new j({props:{name:"pan_and_scan_batched",anchor:"transformers.Gemma3ImageProcessorFast.pan_and_scan_batched",parameters:[{name:"images",val:": torch.Tensor"},{name:"pan_and_scan_min_crop_size",val:": int"},{name:"pan_and_scan_max_num_crops",val:": int"},{name:"pan_and_scan_min_ratio_to_activate",val:": float"}],parametersDescription:[{anchor:"transformers.Gemma3ImageProcessorFast.pan_and_scan_batched.image",description:`<strong>image</strong> (<code>torch.Tensor</code>) &#x2014;
Image to resize.`,name:"image"},{anchor:"transformers.Gemma3ImageProcessorFast.pan_and_scan_batched.pan_and_scan_min_crop_size",description:`<strong>pan_and_scan_min_crop_size</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Minimum size of each crop in pan and scan.`,name:"pan_and_scan_min_crop_size"},{anchor:"transformers.Gemma3ImageProcessorFast.pan_and_scan_batched.pan_and_scan_max_num_crops",description:`<strong>pan_and_scan_max_num_crops</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum number of crops per image in pan and scan.`,name:"pan_and_scan_max_num_crops"},{anchor:"transformers.Gemma3ImageProcessorFast.pan_and_scan_batched.pan_and_scan_min_ratio_to_activate",description:`<strong>pan_and_scan_min_ratio_to_activate</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Minimum aspect ratio to activate pan and scan.`,name:"pan_and_scan_min_ratio_to_activate"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/image_processing_gemma3_fast.py#L94"}}),He=new j({props:{name:"preprocess",anchor:"transformers.Gemma3ImageProcessorFast.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.gemma3.image_processing_gemma3_fast.Gemma3FastImageProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.images",description:`<strong>images</strong> (<code>Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[&apos;PIL.Image.Image&apos;], list[numpy.ndarray], list[&apos;torch.Tensor&apos;]]</code>) &#x2014;
Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
passing in images with pixel values between 0 and 1, set <code>do_rescale=False</code>.`,name:"images"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
Describes the maximum input dimensions to the model.`,name:"size"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.default_to_square",description:`<strong>default_to_square</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to default to a square image when resizing, if size is an int.`,name:"default_to_square"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.resample",description:`<strong>resample</strong> (<code>Union[PILImageResampling, F.InterpolationMode, NoneType]</code>) &#x2014;
Resampling filter to use if resizing the image. This can be one of the enum <code>PILImageResampling</code>. Only
has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.do_center_crop",description:`<strong>do_center_crop</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to center crop the image.`,name:"do_center_crop"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.crop_size",description:`<strong>crop_size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
Size of the output image after applying <code>center_crop</code>.`,name:"crop_size"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>Union[int, float, NoneType]</code>) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>Union[float, list[float], NoneType]</code>) &#x2014;
Image mean to use for normalization. Only has an effect if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_mean"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.image_std",description:`<strong>image_std</strong> (<code>Union[float, list[float], NoneType]</code>) &#x2014;
Image standard deviation to use for normalization. Only has an effect if <code>do_normalize</code> is set to
<code>True</code>.`,name:"image_std"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.return_tensors",description:"<strong>return_tensors</strong> (<code>Union[str, ~utils.generic.TensorType, NoneType]</code>) &#x2014;\nReturns stacked tensors if set to `pt, otherwise returns a list of tensors.",name:"return_tensors"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.data_format",description:`<strong>data_format</strong> (<code>~image_utils.ChannelDimension</code>, <em>optional</em>) &#x2014;
Only <code>ChannelDimension.FIRST</code> is supported. Added for compatibility with slow processors.`,name:"data_format"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>Union[str, ~image_utils.ChannelDimension, NoneType]</code>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.device",description:`<strong>device</strong> (<code>torch.device</code>, <em>optional</em>) &#x2014;
The device to process the images on. If unset, the device is inferred from the input images.`,name:"device"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.disable_grouping",description:`<strong>disable_grouping</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to disable grouping of images by size to process them individually and not in batches.
If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
empirical observations, as detailed here: <a href="https://github.com/huggingface/transformers/pull/38157" rel="nofollow">https://github.com/huggingface/transformers/pull/38157</a>`,name:"disable_grouping"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.do_pan_and_scan",description:`<strong>do_pan_and_scan</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to apply <code>pan_and_scan</code> to images.`,name:"do_pan_and_scan"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.pan_and_scan_min_crop_size",description:`<strong>pan_and_scan_min_crop_size</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Minimum size of each crop in pan and scan.`,name:"pan_and_scan_min_crop_size"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.pan_and_scan_max_num_crops",description:`<strong>pan_and_scan_max_num_crops</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum number of crops per image in pan and scan.`,name:"pan_and_scan_max_num_crops"},{anchor:"transformers.Gemma3ImageProcessorFast.preprocess.pan_and_scan_min_ratio_to_activate",description:`<strong>pan_and_scan_min_ratio_to_activate</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Minimum aspect ratio to activate pan and scan.`,name:"pan_and_scan_min_ratio_to_activate"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/image_processing_gemma3_fast.py#L179",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<ul>
<li><strong>data</strong> (<code>dict</code>) — Dictionary of lists/arrays/tensors returned by the <strong>call</strong> method (‘pixel_values’, etc.).</li>
<li><strong>tensor_type</strong> (<code>Union[None, str, TensorType]</code>, <em>optional</em>) — You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
initialization.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>&lt;class 'transformers.image_processing_base.BatchFeature'&gt;</code></p>
`}}),De=new A({props:{title:"Gemma3Processor",local:"transformers.Gemma3Processor",headingTag:"h2"}}),Ke=new j({props:{name:"class transformers.Gemma3Processor",anchor:"transformers.Gemma3Processor",parameters:[{name:"image_processor",val:""},{name:"tokenizer",val:""},{name:"chat_template",val:" = None"},{name:"image_seq_length",val:": int = 256"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/processing_gemma3.py#L53"}}),et=new A({props:{title:"Gemma3TextConfig",local:"transformers.Gemma3TextConfig",headingTag:"h2"}}),tt=new j({props:{name:"class transformers.Gemma3TextConfig",anchor:"transformers.Gemma3TextConfig",parameters:[{name:"vocab_size",val:" = 262208"},{name:"hidden_size",val:" = 2304"},{name:"intermediate_size",val:" = 9216"},{name:"num_hidden_layers",val:" = 26"},{name:"num_attention_heads",val:" = 8"},{name:"num_key_value_heads",val:" = 4"},{name:"head_dim",val:" = 256"},{name:"hidden_activation",val:" = 'gelu_pytorch_tanh'"},{name:"max_position_embeddings",val:" = 131072"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"bos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = True"},{name:"rope_theta",val:" = 1000000.0"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"query_pre_attn_scalar",val:" = 256"},{name:"sliding_window",val:" = 4096"},{name:"layer_types",val:" = None"},{name:"final_logit_softcapping",val:" = None"},{name:"attn_logit_softcapping",val:" = None"},{name:"rope_scaling",val:" = None"},{name:"rope_local_base_freq",val:" = 10000.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma3TextConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 262208) &#x2014;
Vocabulary size of the Gemma3Text model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextModel">Gemma3TextModel</a>`,name:"vocab_size"},{anchor:"transformers.Gemma3TextConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2304) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Gemma3TextConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 9216) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Gemma3TextConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 26) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.Gemma3TextConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Gemma3TextConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Gemma3TextConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The attention head dimension.`,name:"head_dim"},{anchor:"transformers.Gemma3TextConfig.hidden_activation",description:`<strong>hidden_activation</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu_pytorch_tanh&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder. Will default to <code>&quot;gelu_pytorch_tanh&quot;</code>
if not specified. <code>&quot;gelu_pytorch_tanh&quot;</code> uses an approximation of the <code>&quot;gelu&quot;</code> activation function.`,name:"hidden_activation"},{anchor:"transformers.Gemma3TextConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 131072) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Gemma3TextConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Gemma3TextConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Gemma3TextConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Gemma3TextConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.Gemma3TextConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Gemma3TextConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Gemma3TextConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Gemma3TextConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 1000000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Gemma3TextConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.Gemma3TextConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Gemma3TextConfig.query_pre_attn_scalar",description:`<strong>query_pre_attn_scalar</strong> (<code>float</code>, <em>optional</em>, defaults to 256) &#x2014;
Scaling factor used on the attention scores`,name:"query_pre_attn_scalar"},{anchor:"transformers.Gemma3TextConfig.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
In Gemma3Text, every other layer uses sliding window attention. This is the size of the sliding window.`,name:"sliding_window"},{anchor:"transformers.Gemma3TextConfig.layer_types",description:`<strong>layer_types</strong> (<code>list</code>, <em>optional</em>) &#x2014;
Attention pattern for each layer.`,name:"layer_types"},{anchor:"transformers.Gemma3TextConfig.final_logit_softcapping",description:`<strong>final_logit_softcapping</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Scaling factor when applying tanh softcapping on the logits.`,name:"final_logit_softcapping"},{anchor:"transformers.Gemma3TextConfig.attn_logit_softcapping",description:`<strong>attn_logit_softcapping</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Scaling factor when applying tanh softcapping on the attention scores.`,name:"attn_logit_softcapping"},{anchor:"transformers.Gemma3TextConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings used in global attention. NOTE: if you apply new rope type
and you expect the model to work on longer <code>max_position_embeddings</code>, we recommend you to update this value
accordingly.
Expected contents:
<code>rope_type</code> (<code>str</code>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;linear&#x2019;, &#x2018;dynamic&#x2019;, &#x2018;yarn&#x2019;, &#x2018;longrope&#x2019;,
&#x2018;llama3&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<code>factor</code> (<code>float</code>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings. In
most scaling types, a <code>factor</code> of x will enable the model to handle sequences of length x <em>
original maximum pre-trained length.
<code>original_max_position_embeddings</code> (<code>int</code>, </em>optional<em>):
Used with &#x2018;dynamic&#x2019;, &#x2018;longrope&#x2019; and &#x2018;llama3&#x2019;. The original max position embeddings used during
pretraining.
<code>attention_factor</code> (<code>float</code>, </em>optional<em>):
Used with &#x2018;yarn&#x2019; and &#x2018;longrope&#x2019;. The scaling factor to be applied on the attention
computation. If unspecified, it defaults to value recommended by the implementation, using the
<code>factor</code> field to infer the suggested value.
<code>beta_fast</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for extrapolation (only) in the linear
ramp function. If unspecified, it defaults to 32.
<code>beta_slow</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for interpolation (only) in the linear
ramp function. If unspecified, it defaults to 1.
<code>short_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Gemma3TextConfig.rope_local_base_freq",description:`<strong>rope_local_base_freq</strong> (float, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings for local attention.`,name:"rope_local_base_freq"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/configuration_gemma3.py#L34"}}),ce=new Ct({props:{anchor:"transformers.Gemma3TextConfig.example",$$slots:{default:[ma]},$$scope:{ctx:v}}}),nt=new A({props:{title:"Gemma3Config",local:"transformers.Gemma3Config",headingTag:"h2"}}),ot=new j({props:{name:"class transformers.Gemma3Config",anchor:"transformers.Gemma3Config",parameters:[{name:"text_config",val:": typing.Union[transformers.models.gemma3.configuration_gemma3.Gemma3TextConfig, dict[str, typing.Any], NoneType] = None"},{name:"vision_config",val:": typing.Union[transformers.models.siglip.configuration_siglip.SiglipVisionConfig, dict[str, typing.Any], NoneType] = None"},{name:"mm_tokens_per_image",val:": int = 256"},{name:"boi_token_index",val:": int = 255999"},{name:"eoi_token_index",val:": int = 256000"},{name:"image_token_index",val:": int = 262144"},{name:"initializer_range",val:": float = 0.02"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma3Config.text_config",description:`<strong>text_config</strong> (<code>Union[Gemma3TextConfig, dict]</code>, <em>optional</em>) &#x2014;
The config object of the text backbone.`,name:"text_config"},{anchor:"transformers.Gemma3Config.vision_config",description:`<strong>vision_config</strong> (<code>Union[AutoConfig, dict]</code>,  <em>optional</em>) &#x2014;
Custom vision config or dict.`,name:"vision_config"},{anchor:"transformers.Gemma3Config.mm_tokens_per_image",description:`<strong>mm_tokens_per_image</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The number of tokens per image embedding.`,name:"mm_tokens_per_image"},{anchor:"transformers.Gemma3Config.boi_token_index",description:`<strong>boi_token_index</strong> (<code>int</code>, <em>optional</em>, defaults to 255999) &#x2014;
The begin-of-image token index to wrap the image prompt.`,name:"boi_token_index"},{anchor:"transformers.Gemma3Config.eoi_token_index",description:`<strong>eoi_token_index</strong> (<code>int</code>, <em>optional</em>, defaults to 256000) &#x2014;
The end-of-image token index to wrap the image prompt.`,name:"eoi_token_index"},{anchor:"transformers.Gemma3Config.image_token_index",description:`<strong>image_token_index</strong> (<code>int</code>, <em>optional</em>, defaults to 262144) &#x2014;
The image token index to encode the image prompt.`,name:"image_token_index"},{anchor:"transformers.Gemma3Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/configuration_gemma3.py#L253"}}),de=new Ct({props:{anchor:"transformers.Gemma3Config.example",$$slots:{default:[pa]},$$scope:{ctx:v}}}),st=new A({props:{title:"Gemma3TextModel",local:"transformers.Gemma3TextModel",headingTag:"h2"}}),at=new j({props:{name:"class transformers.Gemma3TextModel",anchor:"transformers.Gemma3TextModel",parameters:[{name:"config",val:": Gemma3TextConfig"}],parametersDescription:[{anchor:"transformers.Gemma3TextModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextConfig">Gemma3TextConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L447"}}),rt=new j({props:{name:"forward",anchor:"transformers.Gemma3TextModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Gemma3TextModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma3TextModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma3TextModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma3TextModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma3TextModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma3TextModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Gemma3TextModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Gemma3TextModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Gemma3TextModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L476",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config"
>Gemma3Config</a>) and inputs.</p>
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
`}}),me=new pn({props:{$$slots:{default:[ga]},$$scope:{ctx:v}}}),it=new A({props:{title:"Gemma3Model",local:"transformers.Gemma3Model",headingTag:"h2"}}),lt=new j({props:{name:"class transformers.Gemma3Model",anchor:"transformers.Gemma3Model",parameters:[{name:"config",val:": Gemma3Config"}],parametersDescription:[{anchor:"transformers.Gemma3Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config">Gemma3Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L757"}}),ct=new j({props:{name:"forward",anchor:"transformers.Gemma3Model.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"pixel_values",val:": FloatTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**lm_kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma3Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma3Model.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor">Gemma3ImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">Gemma3ImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Processor">Gemma3Processor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor">Gemma3ImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.Gemma3Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma3Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma3Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma3Model.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Gemma3Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Gemma3Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma3Model.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.text_config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.text_config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Gemma3Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Gemma3Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Gemma3Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Gemma3Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L824",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.gemma3.modeling_gemma3.Gemma3ModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config"
>Gemma3Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>)</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
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
<p><strong>image_hidden_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) — A <code>torch.FloatTensor</code> of size <code>(batch_size, num_images, sequence_length, hidden_size)</code>.
image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.gemma3.modeling_gemma3.Gemma3ModelOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),pe=new pn({props:{$$slots:{default:[ha]},$$scope:{ctx:v}}}),ge=new Ct({props:{anchor:"transformers.Gemma3Model.forward.example",$$slots:{default:[ua]},$$scope:{ctx:v}}}),dt=new j({props:{name:"get_image_features",anchor:"transformers.Gemma3Model.get_image_features",parameters:[{name:"pixel_values",val:": Tensor"}],parametersDescription:[{anchor:"transformers.Gemma3Model.get_image_features.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor]</code> of shape <code>(batch_size, channels, height, width)</code>) &#x2014;
The tensors corresponding to the input images.`,name:"pixel_values"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L786",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Image feature tensor of shape <code>(num_images, image_length, embed_dim)</code>).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>image_features (<code>torch.Tensor</code>)</p>
`}}),mt=new j({props:{name:"get_placeholder_mask",anchor:"transformers.Gemma3Model.get_placeholder_mask",parameters:[{name:"input_ids",val:": LongTensor"},{name:"inputs_embeds",val:": FloatTensor"},{name:"image_features",val:": FloatTensor"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L800"}}),pt=new A({props:{title:"Gemma3ForCausalLM",local:"transformers.Gemma3ForCausalLM",headingTag:"h2"}}),gt=new j({props:{name:"class transformers.Gemma3ForCausalLM",anchor:"transformers.Gemma3ForCausalLM",parameters:[{name:"config",val:": Gemma3TextConfig"}],parametersDescription:[{anchor:"transformers.Gemma3ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3TextConfig">Gemma3TextConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L587"}}),ht=new j({props:{name:"forward",anchor:"transformers.Gemma3ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma3ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma3ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma3ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma3ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma3ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma3ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Gemma3ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Gemma3ForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Gemma3ForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Gemma3ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Gemma3ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L603",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config"
>Gemma3Config</a>) and inputs.</p>
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
`}}),fe=new pn({props:{$$slots:{default:[fa]},$$scope:{ctx:v}}}),_e=new Ct({props:{anchor:"transformers.Gemma3ForCausalLM.forward.example",$$slots:{default:[_a]},$$scope:{ctx:v}}}),ut=new A({props:{title:"Gemma3ForConditionalGeneration",local:"transformers.Gemma3ForConditionalGeneration",headingTag:"h2"}}),ft=new j({props:{name:"class transformers.Gemma3ForConditionalGeneration",anchor:"transformers.Gemma3ForConditionalGeneration",parameters:[{name:"config",val:": Gemma3Config"}],parametersDescription:[{anchor:"transformers.Gemma3ForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config">Gemma3Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L964"}}),_t=new j({props:{name:"forward",anchor:"transformers.Gemma3ForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"pixel_values",val:": FloatTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**lm_kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma3ForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor">Gemma3ImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">Gemma3ImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Processor">Gemma3Processor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor">Gemma3ImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.text_config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.text_config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Gemma3ForConditionalGeneration.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L1007",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.gemma3.modeling_gemma3.Gemma3CausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config"
>Gemma3Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.text_config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>)</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
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
<li>
<p><strong>image_hidden_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) — A <code>torch.FloatTensor</code> of size <code>(batch_size, num_images, sequence_length, hidden_size)</code>.
image_hidden_states of the model produced by the vision encoder after projecting last hidden state.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.gemma3.modeling_gemma3.Gemma3CausalLMOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ye=new pn({props:{$$slots:{default:[ya]},$$scope:{ctx:v}}}),Me=new Ct({props:{anchor:"transformers.Gemma3ForConditionalGeneration.forward.example",$$slots:{default:[Ma]},$$scope:{ctx:v}}}),yt=new A({props:{title:"Gemma3ForSequenceClassification",local:"transformers.Gemma3ForSequenceClassification",headingTag:"h2"}}),Mt=new j({props:{name:"class transformers.Gemma3ForSequenceClassification",anchor:"transformers.Gemma3ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L1207"}}),Tt=new j({props:{name:"forward",anchor:"transformers.Gemma3ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Gemma3ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma3ForSequenceClassification.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor">Gemma3ImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">Gemma3ImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Processor">Gemma3Processor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3ImageProcessor">Gemma3ImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.Gemma3ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma3ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma3ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma3ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma3ForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.Gemma3ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.Gemma3ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py#L1229",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gemma3#transformers.Gemma3Config"
>Gemma3Config</a>) and inputs.</p>
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
`}}),Te=new pn({props:{$$slots:{default:[Ta]},$$scope:{ctx:v}}}),be=new Ct({props:{anchor:"transformers.Gemma3ForSequenceClassification.forward.example",$$slots:{default:[ba]},$$scope:{ctx:v}}}),we=new Ct({props:{anchor:"transformers.Gemma3ForSequenceClassification.forward.example-2",$$slots:{default:[wa]},$$scope:{ctx:v}}}),bt=new sa({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gemma3.md"}}),{c(){t=c("meta"),M=r(),n=c("p"),p=r(),w=c("p"),w.innerHTML=l,T=r(),I=c("div"),I.innerHTML=gn,Ie=r(),g(H.$$.fragment),_n=r(),je=c("p"),je.innerHTML=ds,yn=r(),Ce=c("p"),Ce.textContent=ms,Mn=r(),Ge=c("p"),Ge.innerHTML=ps,Tn=r(),g(oe.$$.fragment),bn=r(),xe=c("p"),xe.innerHTML=gs,wn=r(),g(se.$$.fragment),vn=r(),ke=c("p"),ke.innerHTML=hs,Jn=r(),$e=c("p"),$e.innerHTML=us,Un=r(),g(ze.$$.fragment),In=r(),We=c("p"),We.innerHTML=fs,jn=r(),g(Ze.$$.fragment),Cn=r(),ae=c("div"),ae.innerHTML=_s,Gn=r(),g(qe.$$.fragment),xn=r(),G=c("ul"),Gt=c("li"),Gt.innerHTML=ys,oo=r(),Fe=c("li"),xt=c("p"),xt.textContent=Ms,so=r(),g(Be.$$.fragment),ao=r(),kt=c("li"),kt.innerHTML=Ts,ro=r(),$t=c("li"),$t.innerHTML=bs,io=r(),Ne=c("li"),zt=c("p"),zt.innerHTML=ws,lo=r(),g(Re.$$.fragment),co=r(),Ve=c("li"),Wt=c("p"),Wt.innerHTML=vs,mo=r(),g(Ee.$$.fragment),kn=r(),g(Xe.$$.fragment),$n=r(),N=c("div"),g(Pe.$$.fragment),po=r(),Zt=c("p"),Zt.textContent=Js,go=r(),re=c("div"),g(Qe.$$.fragment),ho=r(),qt=c("p"),qt.textContent=Us,uo=r(),ie=c("div"),g(Ae.$$.fragment),fo=r(),Ft=c("p"),Ft.textContent=Is,zn=r(),g(Se.$$.fragment),Wn=r(),R=c("div"),g(Le.$$.fragment),_o=r(),Bt=c("p"),Bt.textContent=js,yo=r(),le=c("div"),g(Ye.$$.fragment),Mo=r(),Nt=c("p"),Nt.textContent=Cs,To=r(),Rt=c("div"),g(He.$$.fragment),Zn=r(),g(De.$$.fragment),qn=r(),Oe=c("div"),g(Ke.$$.fragment),Fn=r(),g(et.$$.fragment),Bn=r(),S=c("div"),g(tt.$$.fragment),bo=r(),Vt=c("p"),Vt.innerHTML=Gs,wo=r(),g(ce.$$.fragment),Nn=r(),g(nt.$$.fragment),Rn=r(),k=c("div"),g(ot.$$.fragment),vo=r(),Et=c("p"),Et.innerHTML=xs,Jo=r(),Xt=c("p"),Xt.innerHTML=ks,Uo=r(),Pt=c("p"),Pt.innerHTML=$s,Io=r(),g(de.$$.fragment),Vn=r(),g(st.$$.fragment),En=r(),$=c("div"),g(at.$$.fragment),jo=r(),Qt=c("p"),Qt.textContent=zs,Co=r(),At=c("p"),At.innerHTML=Ws,Go=r(),St=c("p"),St.innerHTML=Zs,xo=r(),D=c("div"),g(rt.$$.fragment),ko=r(),Lt=c("p"),Lt.innerHTML=qs,$o=r(),g(me.$$.fragment),Xn=r(),g(it.$$.fragment),Pn=r(),C=c("div"),g(lt.$$.fragment),zo=r(),Yt=c("p"),Yt.textContent=Fs,Wo=r(),Ht=c("p"),Ht.innerHTML=Bs,Zo=r(),Dt=c("p"),Dt.innerHTML=Ns,qo=r(),X=c("div"),g(ct.$$.fragment),Fo=r(),Ot=c("p"),Ot.innerHTML=Rs,Bo=r(),g(pe.$$.fragment),No=r(),g(ge.$$.fragment),Ro=r(),he=c("div"),g(dt.$$.fragment),Vo=r(),Kt=c("p"),Kt.textContent=Vs,Eo=r(),ue=c("div"),g(mt.$$.fragment),Xo=r(),en=c("p"),en.innerHTML=Es,Qn=r(),g(pt.$$.fragment),An=r(),z=c("div"),g(gt.$$.fragment),Po=r(),tn=c("p"),tn.textContent=Xs,Qo=r(),nn=c("p"),nn.innerHTML=Ps,Ao=r(),on=c("p"),on.innerHTML=Qs,So=r(),P=c("div"),g(ht.$$.fragment),Lo=r(),sn=c("p"),sn.innerHTML=As,Yo=r(),g(fe.$$.fragment),Ho=r(),g(_e.$$.fragment),Sn=r(),g(ut.$$.fragment),Ln=r(),W=c("div"),g(ft.$$.fragment),Do=r(),an=c("p"),an.textContent=Ss,Oo=r(),rn=c("p"),rn.innerHTML=Ls,Ko=r(),ln=c("p"),ln.innerHTML=Ys,es=r(),Q=c("div"),g(_t.$$.fragment),ts=r(),cn=c("p"),cn.innerHTML=Hs,ns=r(),g(ye.$$.fragment),os=r(),g(Me.$$.fragment),Yn=r(),g(yt.$$.fragment),Hn=r(),te=c("div"),g(Mt.$$.fragment),ss=r(),F=c("div"),g(Tt.$$.fragment),as=r(),dn=c("p"),dn.innerHTML=Ds,rs=r(),g(Te.$$.fragment),is=r(),g(be.$$.fragment),ls=r(),g(we.$$.fragment),Dn=r(),g(bt.$$.fragment),On=r(),hn=c("p"),this.h()},l(e){const o=na("svelte-u9bgzb",document.head);t=d(o,"META",{name:!0,content:!0}),o.forEach(s),M=i(e),n=d(e,"P",{}),J(n).forEach(s),p=i(e),w=d(e,"P",{"data-svelte-h":!0}),b(w)!=="svelte-1ejsfye"&&(w.innerHTML=l),T=i(e),I=d(e,"DIV",{style:!0,"data-svelte-h":!0}),b(I)!=="svelte-ithiq1"&&(I.innerHTML=gn),Ie=i(e),h(H.$$.fragment,e),_n=i(e),je=d(e,"P",{"data-svelte-h":!0}),b(je)!=="svelte-1v4cpdt"&&(je.innerHTML=ds),yn=i(e),Ce=d(e,"P",{"data-svelte-h":!0}),b(Ce)!=="svelte-1ky12wo"&&(Ce.textContent=ms),Mn=i(e),Ge=d(e,"P",{"data-svelte-h":!0}),b(Ge)!=="svelte-1oxj0jm"&&(Ge.innerHTML=ps),Tn=i(e),h(oe.$$.fragment,e),bn=i(e),xe=d(e,"P",{"data-svelte-h":!0}),b(xe)!=="svelte-2n7mbe"&&(xe.innerHTML=gs),wn=i(e),h(se.$$.fragment,e),vn=i(e),ke=d(e,"P",{"data-svelte-h":!0}),b(ke)!=="svelte-nf5ooi"&&(ke.innerHTML=hs),Jn=i(e),$e=d(e,"P",{"data-svelte-h":!0}),b($e)!=="svelte-w36i1c"&&($e.innerHTML=us),Un=i(e),h(ze.$$.fragment,e),In=i(e),We=d(e,"P",{"data-svelte-h":!0}),b(We)!=="svelte-w3z5ks"&&(We.innerHTML=fs),jn=i(e),h(Ze.$$.fragment,e),Cn=i(e),ae=d(e,"DIV",{class:!0,"data-svelte-h":!0}),b(ae)!=="svelte-1ymje3b"&&(ae.innerHTML=_s),Gn=i(e),h(qe.$$.fragment,e),xn=i(e),G=d(e,"UL",{});var Z=J(G);Gt=d(Z,"LI",{"data-svelte-h":!0}),b(Gt)!=="svelte-htio6n"&&(Gt.innerHTML=ys),oo=i(Z),Fe=d(Z,"LI",{});var wt=J(Fe);xt=d(wt,"P",{"data-svelte-h":!0}),b(xt)!=="svelte-1mvssnm"&&(xt.textContent=Ms),so=i(wt),h(Be.$$.fragment,wt),wt.forEach(s),ao=i(Z),kt=d(Z,"LI",{"data-svelte-h":!0}),b(kt)!=="svelte-d7vme7"&&(kt.innerHTML=Ts),ro=i(Z),$t=d(Z,"LI",{"data-svelte-h":!0}),b($t)!=="svelte-quf5q0"&&($t.innerHTML=bs),io=i(Z),Ne=d(Z,"LI",{});var vt=J(Ne);zt=d(vt,"P",{"data-svelte-h":!0}),b(zt)!=="svelte-1sn6840"&&(zt.innerHTML=ws),lo=i(vt),h(Re.$$.fragment,vt),vt.forEach(s),co=i(Z),Ve=d(Z,"LI",{});var Jt=J(Ve);Wt=d(Jt,"P",{"data-svelte-h":!0}),b(Wt)!=="svelte-1xybe85"&&(Wt.innerHTML=vs),mo=i(Jt),h(Ee.$$.fragment,Jt),Jt.forEach(s),Z.forEach(s),kn=i(e),h(Xe.$$.fragment,e),$n=i(e),N=d(e,"DIV",{class:!0});var L=J(N);h(Pe.$$.fragment,L),po=i(L),Zt=d(L,"P",{"data-svelte-h":!0}),b(Zt)!=="svelte-19glyt2"&&(Zt.textContent=Js),go=i(L),re=d(L,"DIV",{class:!0});var Ut=J(re);h(Qe.$$.fragment,Ut),ho=i(Ut),qt=d(Ut,"P",{"data-svelte-h":!0}),b(qt)!=="svelte-1h1kawo"&&(qt.textContent=Us),Ut.forEach(s),uo=i(L),ie=d(L,"DIV",{class:!0});var It=J(ie);h(Ae.$$.fragment,It),fo=i(It),Ft=d(It,"P",{"data-svelte-h":!0}),b(Ft)!=="svelte-1x3yxsa"&&(Ft.textContent=Is),It.forEach(s),L.forEach(s),zn=i(e),h(Se.$$.fragment,e),Wn=i(e),R=d(e,"DIV",{class:!0});var Y=J(R);h(Le.$$.fragment,Y),_o=i(Y),Bt=d(Y,"P",{"data-svelte-h":!0}),b(Bt)!=="svelte-ydno5i"&&(Bt.textContent=js),yo=i(Y),le=d(Y,"DIV",{class:!0});var jt=J(le);h(Ye.$$.fragment,jt),Mo=i(jt),Nt=d(jt,"P",{"data-svelte-h":!0}),b(Nt)!=="svelte-7j9k7g"&&(Nt.textContent=Cs),jt.forEach(s),To=i(Y),Rt=d(Y,"DIV",{class:!0});var un=J(Rt);h(He.$$.fragment,un),un.forEach(s),Y.forEach(s),Zn=i(e),h(De.$$.fragment,e),qn=i(e),Oe=d(e,"DIV",{class:!0});var fn=J(Oe);h(Ke.$$.fragment,fn),fn.forEach(s),Fn=i(e),h(et.$$.fragment,e),Bn=i(e),S=d(e,"DIV",{class:!0});var ne=J(S);h(tt.$$.fragment,ne),bo=i(ne),Vt=d(ne,"P",{"data-svelte-h":!0}),b(Vt)!=="svelte-11ock3l"&&(Vt.innerHTML=Gs),wo=i(ne),h(ce.$$.fragment,ne),ne.forEach(s),Nn=i(e),h(nt.$$.fragment,e),Rn=i(e),k=d(e,"DIV",{class:!0});var V=J(k);h(ot.$$.fragment,V),vo=i(V),Et=d(V,"P",{"data-svelte-h":!0}),b(Et)!=="svelte-12150up"&&(Et.innerHTML=xs),Jo=i(V),Xt=d(V,"P",{"data-svelte-h":!0}),b(Xt)!=="svelte-1wi79wt"&&(Xt.innerHTML=ks),Uo=i(V),Pt=d(V,"P",{"data-svelte-h":!0}),b(Pt)!=="svelte-1ek1ss9"&&(Pt.innerHTML=$s),Io=i(V),h(de.$$.fragment,V),V.forEach(s),Vn=i(e),h(st.$$.fragment,e),En=i(e),$=d(e,"DIV",{class:!0});var E=J($);h(at.$$.fragment,E),jo=i(E),Qt=d(E,"P",{"data-svelte-h":!0}),b(Qt)!=="svelte-1fgwbjl"&&(Qt.textContent=zs),Co=i(E),At=d(E,"P",{"data-svelte-h":!0}),b(At)!=="svelte-q52n56"&&(At.innerHTML=Ws),Go=i(E),St=d(E,"P",{"data-svelte-h":!0}),b(St)!=="svelte-hswkmf"&&(St.innerHTML=Zs),xo=i(E),D=d(E,"DIV",{class:!0});var mn=J(D);h(rt.$$.fragment,mn),ko=i(mn),Lt=d(mn,"P",{"data-svelte-h":!0}),b(Lt)!=="svelte-llk0dl"&&(Lt.innerHTML=qs),$o=i(mn),h(me.$$.fragment,mn),mn.forEach(s),E.forEach(s),Xn=i(e),h(it.$$.fragment,e),Pn=i(e),C=d(e,"DIV",{class:!0});var B=J(C);h(lt.$$.fragment,B),zo=i(B),Yt=d(B,"P",{"data-svelte-h":!0}),b(Yt)!=="svelte-13jeddb"&&(Yt.textContent=Fs),Wo=i(B),Ht=d(B,"P",{"data-svelte-h":!0}),b(Ht)!=="svelte-q52n56"&&(Ht.innerHTML=Bs),Zo=i(B),Dt=d(B,"P",{"data-svelte-h":!0}),b(Dt)!=="svelte-hswkmf"&&(Dt.innerHTML=Ns),qo=i(B),X=d(B,"DIV",{class:!0});var ve=J(X);h(ct.$$.fragment,ve),Fo=i(ve),Ot=d(ve,"P",{"data-svelte-h":!0}),b(Ot)!=="svelte-wpdo01"&&(Ot.innerHTML=Rs),Bo=i(ve),h(pe.$$.fragment,ve),No=i(ve),h(ge.$$.fragment,ve),ve.forEach(s),Ro=i(B),he=d(B,"DIV",{class:!0});var eo=J(he);h(dt.$$.fragment,eo),Vo=i(eo),Kt=d(eo,"P",{"data-svelte-h":!0}),b(Kt)!=="svelte-hfwqg7"&&(Kt.textContent=Vs),eo.forEach(s),Eo=i(B),ue=d(B,"DIV",{class:!0});var to=J(ue);h(mt.$$.fragment,to),Xo=i(to),en=d(to,"P",{"data-svelte-h":!0}),b(en)!=="svelte-3ue1dv"&&(en.innerHTML=Es),to.forEach(s),B.forEach(s),Qn=i(e),h(pt.$$.fragment,e),An=i(e),z=d(e,"DIV",{class:!0});var O=J(z);h(gt.$$.fragment,O),Po=i(O),tn=d(O,"P",{"data-svelte-h":!0}),b(tn)!=="svelte-iz2hq5"&&(tn.textContent=Xs),Qo=i(O),nn=d(O,"P",{"data-svelte-h":!0}),b(nn)!=="svelte-q52n56"&&(nn.innerHTML=Ps),Ao=i(O),on=d(O,"P",{"data-svelte-h":!0}),b(on)!=="svelte-hswkmf"&&(on.innerHTML=Qs),So=i(O),P=d(O,"DIV",{class:!0});var Je=J(P);h(ht.$$.fragment,Je),Lo=i(Je),sn=d(Je,"P",{"data-svelte-h":!0}),b(sn)!=="svelte-7fou19"&&(sn.innerHTML=As),Yo=i(Je),h(fe.$$.fragment,Je),Ho=i(Je),h(_e.$$.fragment,Je),Je.forEach(s),O.forEach(s),Sn=i(e),h(ut.$$.fragment,e),Ln=i(e),W=d(e,"DIV",{class:!0});var K=J(W);h(ft.$$.fragment,K),Do=i(K),an=d(K,"P",{"data-svelte-h":!0}),b(an)!=="svelte-vqujyp"&&(an.textContent=Ss),Oo=i(K),rn=d(K,"P",{"data-svelte-h":!0}),b(rn)!=="svelte-q52n56"&&(rn.innerHTML=Ls),Ko=i(K),ln=d(K,"P",{"data-svelte-h":!0}),b(ln)!=="svelte-hswkmf"&&(ln.innerHTML=Ys),es=i(K),Q=d(K,"DIV",{class:!0});var Ue=J(Q);h(_t.$$.fragment,Ue),ts=i(Ue),cn=d(Ue,"P",{"data-svelte-h":!0}),b(cn)!=="svelte-1890ovn"&&(cn.innerHTML=Hs),ns=i(Ue),h(ye.$$.fragment,Ue),os=i(Ue),h(Me.$$.fragment,Ue),Ue.forEach(s),K.forEach(s),Yn=i(e),h(yt.$$.fragment,e),Hn=i(e),te=d(e,"DIV",{class:!0});var no=J(te);h(Mt.$$.fragment,no),ss=i(no),F=d(no,"DIV",{class:!0});var ee=J(F);h(Tt.$$.fragment,ee),as=i(ee),dn=d(ee,"P",{"data-svelte-h":!0}),b(dn)!=="svelte-163oydx"&&(dn.innerHTML=Ds),rs=i(ee),h(Te.$$.fragment,ee),is=i(ee),h(be.$$.fragment,ee),ls=i(ee),h(we.$$.fragment,ee),ee.forEach(s),no.forEach(s),Dn=i(e),h(bt.$$.fragment,e),On=i(e),hn=d(e,"P",{}),J(hn).forEach(s),this.h()},h(){U(t,"name","hf:doc:metadata"),U(t,"content",Ja),oa(I,"float","right"),U(ae,"class","flex justify-center"),U(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Rt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){a(document.head,t),m(e,M,o),m(e,n,o),m(e,p,o),m(e,w,o),m(e,T,o),m(e,I,o),m(e,Ie,o),u(H,e,o),m(e,_n,o),m(e,je,o),m(e,yn,o),m(e,Ce,o),m(e,Mn,o),m(e,Ge,o),m(e,Tn,o),u(oe,e,o),m(e,bn,o),m(e,xe,o),m(e,wn,o),u(se,e,o),m(e,vn,o),m(e,ke,o),m(e,Jn,o),m(e,$e,o),m(e,Un,o),u(ze,e,o),m(e,In,o),m(e,We,o),m(e,jn,o),u(Ze,e,o),m(e,Cn,o),m(e,ae,o),m(e,Gn,o),u(qe,e,o),m(e,xn,o),m(e,G,o),a(G,Gt),a(G,oo),a(G,Fe),a(Fe,xt),a(Fe,so),u(Be,Fe,null),a(G,ao),a(G,kt),a(G,ro),a(G,$t),a(G,io),a(G,Ne),a(Ne,zt),a(Ne,lo),u(Re,Ne,null),a(G,co),a(G,Ve),a(Ve,Wt),a(Ve,mo),u(Ee,Ve,null),m(e,kn,o),u(Xe,e,o),m(e,$n,o),m(e,N,o),u(Pe,N,null),a(N,po),a(N,Zt),a(N,go),a(N,re),u(Qe,re,null),a(re,ho),a(re,qt),a(N,uo),a(N,ie),u(Ae,ie,null),a(ie,fo),a(ie,Ft),m(e,zn,o),u(Se,e,o),m(e,Wn,o),m(e,R,o),u(Le,R,null),a(R,_o),a(R,Bt),a(R,yo),a(R,le),u(Ye,le,null),a(le,Mo),a(le,Nt),a(R,To),a(R,Rt),u(He,Rt,null),m(e,Zn,o),u(De,e,o),m(e,qn,o),m(e,Oe,o),u(Ke,Oe,null),m(e,Fn,o),u(et,e,o),m(e,Bn,o),m(e,S,o),u(tt,S,null),a(S,bo),a(S,Vt),a(S,wo),u(ce,S,null),m(e,Nn,o),u(nt,e,o),m(e,Rn,o),m(e,k,o),u(ot,k,null),a(k,vo),a(k,Et),a(k,Jo),a(k,Xt),a(k,Uo),a(k,Pt),a(k,Io),u(de,k,null),m(e,Vn,o),u(st,e,o),m(e,En,o),m(e,$,o),u(at,$,null),a($,jo),a($,Qt),a($,Co),a($,At),a($,Go),a($,St),a($,xo),a($,D),u(rt,D,null),a(D,ko),a(D,Lt),a(D,$o),u(me,D,null),m(e,Xn,o),u(it,e,o),m(e,Pn,o),m(e,C,o),u(lt,C,null),a(C,zo),a(C,Yt),a(C,Wo),a(C,Ht),a(C,Zo),a(C,Dt),a(C,qo),a(C,X),u(ct,X,null),a(X,Fo),a(X,Ot),a(X,Bo),u(pe,X,null),a(X,No),u(ge,X,null),a(C,Ro),a(C,he),u(dt,he,null),a(he,Vo),a(he,Kt),a(C,Eo),a(C,ue),u(mt,ue,null),a(ue,Xo),a(ue,en),m(e,Qn,o),u(pt,e,o),m(e,An,o),m(e,z,o),u(gt,z,null),a(z,Po),a(z,tn),a(z,Qo),a(z,nn),a(z,Ao),a(z,on),a(z,So),a(z,P),u(ht,P,null),a(P,Lo),a(P,sn),a(P,Yo),u(fe,P,null),a(P,Ho),u(_e,P,null),m(e,Sn,o),u(ut,e,o),m(e,Ln,o),m(e,W,o),u(ft,W,null),a(W,Do),a(W,an),a(W,Oo),a(W,rn),a(W,Ko),a(W,ln),a(W,es),a(W,Q),u(_t,Q,null),a(Q,ts),a(Q,cn),a(Q,ns),u(ye,Q,null),a(Q,os),u(Me,Q,null),m(e,Yn,o),u(yt,e,o),m(e,Hn,o),m(e,te,o),u(Mt,te,null),a(te,ss),a(te,F),u(Tt,F,null),a(F,as),a(F,dn),a(F,rs),u(Te,F,null),a(F,is),u(be,F,null),a(F,ls),u(we,F,null),m(e,Dn,o),u(bt,e,o),m(e,On,o),m(e,hn,o),Kn=!0},p(e,[o]){const Z={};o&2&&(Z.$$scope={dirty:o,ctx:e}),oe.$set(Z);const wt={};o&2&&(wt.$$scope={dirty:o,ctx:e}),se.$set(wt);const vt={};o&2&&(vt.$$scope={dirty:o,ctx:e}),ce.$set(vt);const Jt={};o&2&&(Jt.$$scope={dirty:o,ctx:e}),de.$set(Jt);const L={};o&2&&(L.$$scope={dirty:o,ctx:e}),me.$set(L);const Ut={};o&2&&(Ut.$$scope={dirty:o,ctx:e}),pe.$set(Ut);const It={};o&2&&(It.$$scope={dirty:o,ctx:e}),ge.$set(It);const Y={};o&2&&(Y.$$scope={dirty:o,ctx:e}),fe.$set(Y);const jt={};o&2&&(jt.$$scope={dirty:o,ctx:e}),_e.$set(jt);const un={};o&2&&(un.$$scope={dirty:o,ctx:e}),ye.$set(un);const fn={};o&2&&(fn.$$scope={dirty:o,ctx:e}),Me.$set(fn);const ne={};o&2&&(ne.$$scope={dirty:o,ctx:e}),Te.$set(ne);const V={};o&2&&(V.$$scope={dirty:o,ctx:e}),be.$set(V);const E={};o&2&&(E.$$scope={dirty:o,ctx:e}),we.$set(E)},i(e){Kn||(f(H.$$.fragment,e),f(oe.$$.fragment,e),f(se.$$.fragment,e),f(ze.$$.fragment,e),f(Ze.$$.fragment,e),f(qe.$$.fragment,e),f(Be.$$.fragment,e),f(Re.$$.fragment,e),f(Ee.$$.fragment,e),f(Xe.$$.fragment,e),f(Pe.$$.fragment,e),f(Qe.$$.fragment,e),f(Ae.$$.fragment,e),f(Se.$$.fragment,e),f(Le.$$.fragment,e),f(Ye.$$.fragment,e),f(He.$$.fragment,e),f(De.$$.fragment,e),f(Ke.$$.fragment,e),f(et.$$.fragment,e),f(tt.$$.fragment,e),f(ce.$$.fragment,e),f(nt.$$.fragment,e),f(ot.$$.fragment,e),f(de.$$.fragment,e),f(st.$$.fragment,e),f(at.$$.fragment,e),f(rt.$$.fragment,e),f(me.$$.fragment,e),f(it.$$.fragment,e),f(lt.$$.fragment,e),f(ct.$$.fragment,e),f(pe.$$.fragment,e),f(ge.$$.fragment,e),f(dt.$$.fragment,e),f(mt.$$.fragment,e),f(pt.$$.fragment,e),f(gt.$$.fragment,e),f(ht.$$.fragment,e),f(fe.$$.fragment,e),f(_e.$$.fragment,e),f(ut.$$.fragment,e),f(ft.$$.fragment,e),f(_t.$$.fragment,e),f(ye.$$.fragment,e),f(Me.$$.fragment,e),f(yt.$$.fragment,e),f(Mt.$$.fragment,e),f(Tt.$$.fragment,e),f(Te.$$.fragment,e),f(be.$$.fragment,e),f(we.$$.fragment,e),f(bt.$$.fragment,e),Kn=!0)},o(e){_(H.$$.fragment,e),_(oe.$$.fragment,e),_(se.$$.fragment,e),_(ze.$$.fragment,e),_(Ze.$$.fragment,e),_(qe.$$.fragment,e),_(Be.$$.fragment,e),_(Re.$$.fragment,e),_(Ee.$$.fragment,e),_(Xe.$$.fragment,e),_(Pe.$$.fragment,e),_(Qe.$$.fragment,e),_(Ae.$$.fragment,e),_(Se.$$.fragment,e),_(Le.$$.fragment,e),_(Ye.$$.fragment,e),_(He.$$.fragment,e),_(De.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(ce.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(de.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(me.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(ct.$$.fragment,e),_(pe.$$.fragment,e),_(ge.$$.fragment,e),_(dt.$$.fragment,e),_(mt.$$.fragment,e),_(pt.$$.fragment,e),_(gt.$$.fragment,e),_(ht.$$.fragment,e),_(fe.$$.fragment,e),_(_e.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(_t.$$.fragment,e),_(ye.$$.fragment,e),_(Me.$$.fragment,e),_(yt.$$.fragment,e),_(Mt.$$.fragment,e),_(Tt.$$.fragment,e),_(Te.$$.fragment,e),_(be.$$.fragment,e),_(we.$$.fragment,e),_(bt.$$.fragment,e),Kn=!1},d(e){e&&(s(M),s(n),s(p),s(w),s(T),s(I),s(Ie),s(_n),s(je),s(yn),s(Ce),s(Mn),s(Ge),s(Tn),s(bn),s(xe),s(wn),s(vn),s(ke),s(Jn),s($e),s(Un),s(In),s(We),s(jn),s(Cn),s(ae),s(Gn),s(xn),s(G),s(kn),s($n),s(N),s(zn),s(Wn),s(R),s(Zn),s(qn),s(Oe),s(Fn),s(Bn),s(S),s(Nn),s(Rn),s(k),s(Vn),s(En),s($),s(Xn),s(Pn),s(C),s(Qn),s(An),s(z),s(Sn),s(Ln),s(W),s(Yn),s(Hn),s(te),s(Dn),s(On),s(hn)),s(t),y(H,e),y(oe,e),y(se,e),y(ze,e),y(Ze,e),y(qe,e),y(Be),y(Re),y(Ee),y(Xe,e),y(Pe),y(Qe),y(Ae),y(Se,e),y(Le),y(Ye),y(He),y(De,e),y(Ke),y(et,e),y(tt),y(ce),y(nt,e),y(ot),y(de),y(st,e),y(at),y(rt),y(me),y(it,e),y(lt),y(ct),y(pe),y(ge),y(dt),y(mt),y(pt,e),y(gt),y(ht),y(fe),y(_e),y(ut,e),y(ft),y(_t),y(ye),y(Me),y(yt,e),y(Mt),y(Tt),y(Te),y(be),y(we),y(bt,e)}}}const Ja='{"title":"Gemma 3","local":"gemma-3","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Gemma3ImageProcessor","local":"transformers.Gemma3ImageProcessor","sections":[],"depth":2},{"title":"Gemma3ImageProcessorFast","local":"transformers.Gemma3ImageProcessorFast","sections":[],"depth":2},{"title":"Gemma3Processor","local":"transformers.Gemma3Processor","sections":[],"depth":2},{"title":"Gemma3TextConfig","local":"transformers.Gemma3TextConfig","sections":[],"depth":2},{"title":"Gemma3Config","local":"transformers.Gemma3Config","sections":[],"depth":2},{"title":"Gemma3TextModel","local":"transformers.Gemma3TextModel","sections":[],"depth":2},{"title":"Gemma3Model","local":"transformers.Gemma3Model","sections":[],"depth":2},{"title":"Gemma3ForCausalLM","local":"transformers.Gemma3ForCausalLM","sections":[],"depth":2},{"title":"Gemma3ForConditionalGeneration","local":"transformers.Gemma3ForConditionalGeneration","sections":[],"depth":2},{"title":"Gemma3ForSequenceClassification","local":"transformers.Gemma3ForSequenceClassification","sections":[],"depth":2}],"depth":1}';function Ua(v){return Ks(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Wa extends ea{constructor(t){super(),ta(this,t,Ua,va,Os,{})}}export{Wa as component};
