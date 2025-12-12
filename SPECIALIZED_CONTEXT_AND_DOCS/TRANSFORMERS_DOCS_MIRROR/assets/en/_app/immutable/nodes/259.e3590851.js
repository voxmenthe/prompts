import{s as ls,o as ds,n as W}from"../chunks/scheduler.18a86fab.js";import{S as cs,i as ps,g as l,s as r,r as m,A as ms,h as d,f as s,c as i,j as L,x as k,u,k as x,l as us,y as a,a as c,v as h,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as Nt}from"../chunks/Tip.77304350.js";import{D as z}from"../chunks/Docstring.a1ef7999.js";import{C as K}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Co}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as D,E as hs}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as fs,a as xn}from"../chunks/HfOption.6641485e.js";function gs(T){let t,p="Click on the Llama models in the right sidebar for more examples of how to apply Llama to different language tasks.";return{c(){t=l("p"),t.textContent=p},l(o){t=d(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-wjjvrl"&&(t.textContent=p)},m(o,b){c(o,t,b)},p:W,d(o){o&&s(t)}}}function _s(T){let t,p;return t=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmh1Z2d5bGxhbWElMkZsbGFtYS03YiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZSUzRDAlMEEpJTBBcGlwZWxpbmUoJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;huggyllama/llama-7b&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>)`,wrap:!1}}),{c(){m(t.$$.fragment)},l(o){u(t.$$.fragment,o)},m(o,b){h(t,o,b),p=!0},p:W,i(o){p||(f(t.$$.fragment,o),p=!0)},o(o){g(t.$$.fragment,o),p=!1},d(o){_(t,o)}}}function bs(T){let t,p;return t=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyaHVnZ3lsbGFtYSUyRmxsYW1hLTdiJTIyJTJDJTBBKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmh1Z2d5bGxhbWElMkZsbGFtYS03YiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMEEpJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlBsYW50cyUyMGNyZWF0ZSUyMGVuZXJneSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dF9pZHMlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;huggyllama/llama-7b&quot;</span>,
)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;huggyllama/llama-7b&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){m(t.$$.fragment)},l(o){u(t.$$.fragment,o)},m(o,b){h(t,o,b),p=!0},p:W,i(o){p||(f(t.$$.fragment,o),p=!0)},o(o){g(t.$$.fragment,o),p=!1},d(o){_(t,o)}}}function ks(T){let t,p;return t=new K({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQtZ2VuZXJhdGlvbiUyMC0tbW9kZWwlMjBodWdneWxsYW1hJTJGbGxhbWEtN2IlMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span> | transformers run --task text-generation --model huggyllama/llama-7b --device 0',wrap:!1}}),{c(){m(t.$$.fragment)},l(o){u(t.$$.fragment,o)},m(o,b){h(t,o,b),p=!0},p:W,i(o){p||(f(t.$$.fragment,o),p=!0)},o(o){g(t.$$.fragment,o),p=!1},d(o){_(t,o)}}}function ys(T){let t,p,o,b,$,v;return t=new xn({props:{id:"usage",option:"Pipeline",$$slots:{default:[_s]},$$scope:{ctx:T}}}),o=new xn({props:{id:"usage",option:"AutoModel",$$slots:{default:[bs]},$$scope:{ctx:T}}}),$=new xn({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ks]},$$scope:{ctx:T}}}),{c(){m(t.$$.fragment),p=r(),m(o.$$.fragment),b=r(),m($.$$.fragment)},l(y){u(t.$$.fragment,y),p=i(y),u(o.$$.fragment,y),b=i(y),u($.$$.fragment,y)},m(y,C){h(t,y,C),c(y,p,C),h(o,y,C),c(y,b,C),h($,y,C),v=!0},p(y,C){const At={};C&2&&(At.$$scope={dirty:C,ctx:y}),t.$set(At);const _e={};C&2&&(_e.$$scope={dirty:C,ctx:y}),o.$set(_e);const E={};C&2&&(E.$$scope={dirty:C,ctx:y}),$.$set(E)},i(y){v||(f(t.$$.fragment,y),f(o.$$.fragment,y),f($.$$.fragment,y),v=!0)},o(y){g(t.$$.fragment,y),g(o.$$.fragment,y),g($.$$.fragment,y),v=!1},d(y){y&&(s(p),s(b)),_(t,y),_(o,y),_($,y)}}}function vs(T){let t,p;return t=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExsYW1hTW9kZWwlMkMlMjBMbGFtYUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBMTGFNQSUyMGxsYW1hLTdiJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMExsYW1hQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwbGxhbWEtN2IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMExsYW1hTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LlamaModel, LlamaConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a LLaMA llama-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = LlamaConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the llama-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LlamaModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){m(t.$$.fragment)},l(o){u(t.$$.fragment,o)},m(o,b){h(t,o,b),p=!0},p:W,i(o){p||(f(t.$$.fragment,o),p=!0)},o(o){g(t.$$.fragment,o),p=!1},d(o){_(t,o)}}}function Ts(T){let t,p="sequence pair mask has the following format:",o,b,$;return b=new K({props:{code:"MCUyMDAlMjAwJTIwMCUyMDAlMjAwJTIwMCUyMDAlMjAwJTIwMCUyMDAlMjAxJTIwMSUyMDElMjAxJTIwMSUyMDElMjAxJTIwMSUyMDElMEElN0MlMjBmaXJzdCUyMHNlcXVlbmNlJTIwJTIwJTIwJTIwJTdDJTIwc2Vjb25kJTIwc2VxdWVuY2UlMjAlN0M=",highlighted:`0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1 1
| first sequence    | second sequence |`,wrap:!1}}),{c(){t=l("p"),t.textContent=p,o=r(),m(b.$$.fragment)},l(v){t=d(v,"P",{"data-svelte-h":!0}),k(t)!=="svelte-16klr56"&&(t.textContent=p),o=i(v),u(b.$$.fragment,v)},m(v,y){c(v,t,y),c(v,o,y),h(b,v,y),$=!0},p:W,i(v){$||(f(b.$$.fragment,v),$=!0)},o(v){g(b.$$.fragment,v),$=!1},d(v){v&&(s(t),s(o)),_(b,v)}}}function ws(T){let t,p;return t=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExsYW1hVG9rZW5pemVyRmFzdCUwQSUwQXRva2VuaXplciUyMCUzRCUyMExsYW1hVG9rZW5pemVyRmFzdC5mcm9tX3ByZXRyYWluZWQoJTIyaGYtaW50ZXJuYWwtdGVzdGluZyUyRmxsYW1hLXRva2VuaXplciUyMiklMEF0b2tlbml6ZXIuZW5jb2RlKCUyMkhlbGxvJTIwdGhpcyUyMGlzJTIwYSUyMHRlc3QlMjIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LlamaTokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LlamaTokenizerFast.from_pretrained(<span class="hljs-string">&quot;hf-internal-testing/llama-tokenizer&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.encode(<span class="hljs-string">&quot;Hello this is a test&quot;</span>)
[<span class="hljs-number">1</span>, <span class="hljs-number">15043</span>, <span class="hljs-number">445</span>, <span class="hljs-number">338</span>, <span class="hljs-number">263</span>, <span class="hljs-number">1243</span>]`,wrap:!1}}),{c(){m(t.$$.fragment)},l(o){u(t.$$.fragment,o)},m(o,b){h(t,o,b),p=!0},p:W,i(o){p||(f(t.$$.fragment,o),p=!0)},o(o){g(t.$$.fragment,o),p=!1},d(o){_(t,o)}}}function $s(T){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=p},l(o){t=d(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,b){c(o,t,b)},p:W,d(o){o&&s(t)}}}function Ms(T){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=p},l(o){t=d(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,b){c(o,t,b)},p:W,d(o){o&&s(t)}}}function xs(T){let t,p="Example:",o,b,$;return b=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMbGFtYUZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBMbGFtYUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWxsYW1hJTJGTGxhbWEtMi03Yi1oZiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWxsYW1hJTJGTGxhbWEtMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LlamaForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = LlamaForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-llama/Llama-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-llama/Llama-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=l("p"),t.textContent=p,o=r(),m(b.$$.fragment)},l(v){t=d(v,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(v),u(b.$$.fragment,v)},m(v,y){c(v,t,y),c(v,o,y),h(b,v,y),$=!0},p:W,i(v){$||(f(b.$$.fragment,v),$=!0)},o(v){g(b.$$.fragment,v),$=!1},d(v){v&&(s(t),s(o)),_(b,v)}}}function Ls(T){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=p},l(o){t=d(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,b){c(o,t,b)},p:W,d(o){o&&s(t)}}}function zs(T){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=p},l(o){t=d(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,b){c(o,t,b)},p:W,d(o){o&&s(t)}}}function Cs(T){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=p},l(o){t=d(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,b){c(o,t,b)},p:W,d(o){o&&s(t)}}}function Fs(T){let t,p,o,b,$,v="<em>This model was released on 2023-02-27 and added to Hugging Face Transformers on 2023-03-16.</em>",y,C,At='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',_e,E,Xt,be,Ln='<a href="https://huggingface.co/papers/2302.13971" rel="nofollow">Llama</a> is a family of large language models ranging from 7B to 65B parameters. These models are focused on efficient inference (important for serving language models) by training a smaller model on more tokens rather than training a larger model on fewer tokens. The Llama model is based on the GPT architecture, but it uses pre-normalization to improve training stability, replaces ReLU with SwiGLU to improve performance, and replaces absolute positional embeddings with rotary positional embeddings (RoPE) to better handle longer sequence lengths.',Dt,ke,zn='You can find all the original Llama checkpoints under the <a href="https://huggingface.co/huggyllama" rel="nofollow">Huggy Llama</a> organization.',Yt,ee,St,ye,Cn='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Qt,te,Ot,ve,Fn='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Kt,Te,qn='The example below uses <a href="../quantization/torchao">torchao</a> to only quantize the weights to int4.',eo,we,to,$e,Un='Use the <a href="https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139" rel="nofollow">AttentionMaskVisualizer</a> to better understand what tokens the model can and cannot attend to.',oo,Me,no,oe,jn='<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llama-attn-mask.png"/>',so,xe,ao,Le,In='<li>The tokenizer is a byte-pair encoding model based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>. During decoding, if the first token is the start of the word (for example, “Banana”), the tokenizer doesn’t prepend the prefix space to the string.</li>',ro,ze,io,I,Ce,Fo,dt,Wn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel">LlamaModel</a>. It is used to instantiate an LLaMA
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the LLaMA-7B.
e.g. <a href="https://huggingface.co/meta-llama/Llama-2-7b-hf" rel="nofollow">meta-llama/Llama-2-7b-hf</a>`,qo,ct,Jn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Uo,ne,lo,Fe,co,F,qe,jo,pt,Zn=`Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
no padding token in the original model.`,Io,mt,Ue,Wo,se,je,Jo,ut,Bn=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Zo,J,Ie,Bo,ht,Pn="Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT",Po,ae,Eo,ft,En="if token_ids_1 is None, only returns the first portion of the mask (0s).",Go,re,We,No,gt,Gn="Save the vocabulary and special tokens file to a directory.",po,Je,mo,w,Ze,Ao,_t,Nn="Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.",Ho,bt,An="This uses notably ByteFallback and no normalization.",Vo,ie,Ro,kt,Hn=`If you want to change the <code>bos_token</code> or the <code>eos_token</code>, make sure to specify them when initializing the model, or
call <code>tokenizer.update_post_processor()</code> to make sure that the post-processing is correctly done (otherwise the
values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
[post-processors] (<a href="https://huggingface.co/docs/tokenizers/api/post-processors" rel="nofollow">https://huggingface.co/docs/tokenizers/api/post-processors</a>) documentation.`,Xo,yt,Vn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Do,vt,Be,Yo,le,Pe,So,Tt,Rn=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> or <code>encode_plus</code> methods.`,Qo,G,Ee,Oo,wt,Xn=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Ko,$t,Dn="Should be overridden in a subclass if the model has a special way of building those.",en,de,Ge,tn,Mt,Yn="Updates the underlying post processor with the current <code>bos_token</code> and <code>eos_token</code>.",on,xt,Ne,uo,Ae,ho,q,He,nn,Lt,Sn="The bare Llama Model outputting raw hidden-states without any specific head on top.",sn,zt,Qn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,an,Ct,On=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,rn,N,Ve,ln,Ft,Kn='The <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel">LlamaModel</a> forward method, overrides the <code>__call__</code> special method.',dn,ce,fo,Re,go,U,Xe,cn,qt,es="The Llama Model for causal language modeling.",pn,Ut,ts=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,mn,jt,os=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,un,Z,De,hn,It,ns='The <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForCausalLM">LlamaForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',fn,pe,gn,me,_o,Ye,bo,Y,Se,_n,A,Qe,bn,Wt,ss="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",kn,ue,ko,Oe,yo,S,Ke,yn,H,et,vn,Jt,as="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Tn,he,vo,tt,To,Q,ot,wn,V,nt,$n,Zt,rs="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Mn,fe,wo,st,$o,Ht,Mo;return E=new D({props:{title:"Llama",local:"llama",headingTag:"h1"}}),ee=new Nt({props:{warning:!1,$$slots:{default:[gs]},$$scope:{ctx:T}}}),te=new fs({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[ys]},$$scope:{ctx:T}}}),we=new K({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMHRvcmNoYW8lMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBUb3JjaEFvQ29uZmlnJTJDJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMFRvcmNoQW9Db25maWcoJTIyaW50NF93ZWlnaHRfb25seSUyMiUyQyUyMGdyb3VwX3NpemUlM0QxMjgpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyaHVnZ3lsbGFtYSUyRmxsYW1hLTMwYiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RxdWFudGl6YXRpb25fY29uZmlnJTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmh1Z2d5bGxhbWElMkZsbGFtYS0zMGIlMjIpJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlBsYW50cyUyMGNyZWF0ZSUyMGVuZXJneSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dF9pZHMlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-comment"># pip install torchao</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

quantization_config = TorchAoConfig(<span class="hljs-string">&quot;int4_weight_only&quot;</span>, group_size=<span class="hljs-number">128</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;huggyllama/llama-30b&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;huggyllama/llama-30b&quot;</span>)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),Me=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycy51dGlscy5hdHRlbnRpb25fdmlzdWFsaXplciUyMGltcG9ydCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyJTBBJTBBdmlzdWFsaXplciUyMCUzRCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyKCUyMmh1Z2d5bGxhbWElMkZsbGFtYS03YiUyMiklMEF2aXN1YWxpemVyKCUyMlBsYW50cyUyMGNyZWF0ZSUyMGVuZXJneSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjIp",highlighted:`<span class="hljs-keyword">from</span> transformers.utils.attention_visualizer <span class="hljs-keyword">import</span> AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer(<span class="hljs-string">&quot;huggyllama/llama-7b&quot;</span>)
visualizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>)`,wrap:!1}}),xe=new D({props:{title:"Notes",local:"notes",headingTag:"h2"}}),ze=new D({props:{title:"LlamaConfig",local:"transformers.LlamaConfig",headingTag:"h2"}}),Ce=new z({props:{name:"class transformers.LlamaConfig",anchor:"transformers.LlamaConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 11008"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"pretraining_tp",val:" = 1"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"head_dim",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LlamaConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel">LlamaModel</a>`,name:"vocab_size"},{anchor:"transformers.LlamaConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.LlamaConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.LlamaConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.LlamaConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.LlamaConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.LlamaConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.LlamaConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
Llama 2 up to 4096, CodeLlama up to 16384.`,name:"max_position_embeddings"},{anchor:"transformers.LlamaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.LlamaConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.LlamaConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.LlamaConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.LlamaConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.LlamaConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.LlamaConfig.pretraining_tp",description:`<strong>pretraining_tp</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Experimental feature. Tensor parallelism rank used during pretraining. Please refer to <a href="https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism" rel="nofollow">this
document</a> to
understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
results. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this issue</a>.`,name:"pretraining_tp"},{anchor:"transformers.LlamaConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.LlamaConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.LlamaConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.LlamaConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.LlamaConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.LlamaConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.LlamaConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The attention head dimension. If None, it will default to hidden_size // num_attention_heads`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/configuration_llama.py#L26"}}),ne=new Co({props:{anchor:"transformers.LlamaConfig.example",$$slots:{default:[vs]},$$scope:{ctx:T}}}),Fe=new D({props:{title:"LlamaTokenizer",local:"transformers.LlamaTokenizer",headingTag:"h2"}}),qe=new z({props:{name:"class transformers.LlamaTokenizer",anchor:"transformers.LlamaTokenizer",parameters:[{name:"vocab_file",val:""},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = None"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"add_bos_token",val:" = True"},{name:"add_eos_token",val:" = False"},{name:"clean_up_tokenization_spaces",val:" = False"},{name:"use_default_system_prompt",val:" = False"},{name:"spaces_between_special_tokens",val:" = False"},{name:"legacy",val:" = None"},{name:"add_prefix_space",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LlamaTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.LlamaTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LlamaTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.`,name:"bos_token"},{anchor:"transformers.LlamaTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.LlamaTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation.`,name:"pad_token"},{anchor:"transformers.LlamaTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict[str, Any]</code>, <code>Optional</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.LlamaTokenizer.add_bos_token",description:`<strong>add_bos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add an <code>bos_token</code> at the start of sequences.`,name:"add_bos_token"},{anchor:"transformers.LlamaTokenizer.add_eos_token",description:`<strong>add_eos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an <code>eos_token</code> at the end of sequences.`,name:"add_eos_token"},{anchor:"transformers.LlamaTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.LlamaTokenizer.use_default_system_prompt",description:`<strong>use_default_system_prompt</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the default system prompt for Llama should be used.`,name:"use_default_system_prompt"},{anchor:"transformers.LlamaTokenizer.spaces_between_special_tokens",description:`<strong>spaces_between_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add spaces between special tokens.`,name:"spaces_between_special_tokens"},{anchor:"transformers.LlamaTokenizer.legacy",description:`<strong>legacy</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not the <code>legacy</code> behavior of the tokenizer should be used. Legacy is before the merge of #24622
and #25224 which includes fixes to properly handle tokens that appear after special tokens.
Make sure to also set <code>from_slow</code> to <code>True</code>.
A simple example:</p>
<ul>
<li><code>legacy=True</code>:</li>
</ul>`,name:"legacy"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L56"}}),Ue=new z({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.LlamaTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L333"}}),je=new z({props:{name:"get_special_tokens_mask",anchor:"transformers.LlamaTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.LlamaTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.LlamaTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.LlamaTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L344",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ie=new z({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.LlamaTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.LlamaTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids.`,name:"token_ids_0"},{anchor:"transformers.LlamaTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L381",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#token-type-ids">token type IDs</a> according to the given sequence(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ae=new Co({props:{anchor:"transformers.LlamaTokenizer.create_token_type_ids_from_sequences.example",$$slots:{default:[Ts]},$$scope:{ctx:T}}}),We=new z({props:{name:"save_vocabulary",anchor:"transformers.LlamaTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:""},{name:"filename_prefix",val:": typing.Optional[str] = None"}],parametersDescription:[{anchor:"transformers.LlamaTokenizer.save_vocabulary.save_directory",description:`<strong>save_directory</strong> (<code>str</code>) &#x2014;
The directory in which to save the vocabulary.`,name:"save_directory"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L306",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Paths to the files saved.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Tuple(str)</code></p>
`}}),Je=new D({props:{title:"LlamaTokenizerFast",local:"transformers.LlamaTokenizerFast",headingTag:"h2"}}),Ze=new z({props:{name:"class transformers.LlamaTokenizerFast",anchor:"transformers.LlamaTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = False"},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"add_bos_token",val:" = True"},{name:"add_eos_token",val:" = False"},{name:"use_default_system_prompt",val:" = False"},{name:"legacy",val:" = None"},{name:"add_prefix_space",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LlamaTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a .model extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.LlamaTokenizerFast.tokenizer_file",description:`<strong>tokenizer_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
<a href="https://github.com/huggingface/tokenizers" rel="nofollow">tokenizers</a> file (generally has a .json extension) that
contains everything needed to load the tokenizer.`,name:"tokenizer_file"},{anchor:"transformers.LlamaTokenizerFast.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.LlamaTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LlamaTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.`,name:"bos_token"},{anchor:"transformers.LlamaTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.LlamaTokenizerFast.add_bos_token",description:`<strong>add_bos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add an <code>bos_token</code> at the start of sequences.`,name:"add_bos_token"},{anchor:"transformers.LlamaTokenizerFast.add_eos_token",description:`<strong>add_eos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an <code>eos_token</code> at the end of sequences.`,name:"add_eos_token"},{anchor:"transformers.LlamaTokenizerFast.use_default_system_prompt",description:`<strong>use_default_system_prompt</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the default system prompt for Llama should be used`,name:"use_default_system_prompt"},{anchor:"transformers.LlamaTokenizerFast.legacy",description:`<strong>legacy</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not the <code>legacy</code> behavior of the tokenizer should be used. Legacy is before the merge of #24622
and #25224 which includes fixes to properly handle tokens that appear after special tokens.
Make sure to also set <code>from_slow</code> to <code>True</code>.
A simple example:</p>
<ul>
<li><code>legacy=True</code>:</li>
</ul>`,name:"legacy"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama_fast.py#L46"}}),ie=new Co({props:{anchor:"transformers.LlamaTokenizerFast.example",$$slots:{default:[ws]},$$scope:{ctx:T}}}),Be=new z({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.LlamaTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama_fast.py#L239"}}),Pe=new z({props:{name:"get_special_tokens_mask",anchor:"transformers.LlamaTokenizerFast.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.LlamaTokenizerFast.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids of the first sequence.`,name:"token_ids_0"},{anchor:"transformers.LlamaTokenizerFast.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
List of ids of the second sequence.`,name:"token_ids_1"},{anchor:"transformers.LlamaTokenizerFast.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3913",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]</p>
`}}),Ee=new z({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.LlamaTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.LlamaTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.LlamaTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ge=new z({props:{name:"update_post_processor",anchor:"transformers.LlamaTokenizerFast.update_post_processor",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama_fast.py#L174"}}),Ne=new z({props:{name:"save_vocabulary",anchor:"transformers.LlamaTokenizerFast.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama_fast.py#L218"}}),Ae=new D({props:{title:"LlamaModel",local:"transformers.LlamaModel",headingTag:"h2"}}),He=new z({props:{name:"class transformers.LlamaModel",anchor:"transformers.LlamaModel",parameters:[{name:"config",val:": LlamaConfig"}],parametersDescription:[{anchor:"transformers.LlamaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig">LlamaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L334"}}),Ve=new z({props:{name:"forward",anchor:"transformers.LlamaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.LlamaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LlamaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LlamaModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LlamaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LlamaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LlamaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.LlamaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L351",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig"
>LlamaConfig</a>) and inputs.</p>
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
`}}),ce=new Nt({props:{$$slots:{default:[$s]},$$scope:{ctx:T}}}),Re=new D({props:{title:"LlamaForCausalLM",local:"transformers.LlamaForCausalLM",headingTag:"h2"}}),Xe=new z({props:{name:"class transformers.LlamaForCausalLM",anchor:"transformers.LlamaForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LlamaForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForCausalLM">LlamaForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L413"}}),De=new z({props:{name:"forward",anchor:"transformers.LlamaForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.LlamaForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LlamaForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LlamaForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LlamaForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LlamaForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LlamaForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.LlamaForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.LlamaForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.LlamaForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L427",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig"
>LlamaConfig</a>) and inputs.</p>
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
`}}),pe=new Nt({props:{$$slots:{default:[Ms]},$$scope:{ctx:T}}}),me=new Co({props:{anchor:"transformers.LlamaForCausalLM.forward.example",$$slots:{default:[xs]},$$scope:{ctx:T}}}),Ye=new D({props:{title:"LlamaForSequenceClassification",local:"transformers.LlamaForSequenceClassification",headingTag:"h2"}}),Se=new z({props:{name:"class transformers.LlamaForSequenceClassification",anchor:"transformers.LlamaForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L488"}}),Qe=new z({props:{name:"forward",anchor:"transformers.LlamaForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.LlamaForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LlamaForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LlamaForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LlamaForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LlamaForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LlamaForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.LlamaForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L111",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
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
`}}),ue=new Nt({props:{$$slots:{default:[Ls]},$$scope:{ctx:T}}}),Oe=new D({props:{title:"LlamaForQuestionAnswering",local:"transformers.LlamaForQuestionAnswering",headingTag:"h2"}}),Ke=new z({props:{name:"class transformers.LlamaForQuestionAnswering",anchor:"transformers.LlamaForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L491"}}),et=new z({props:{name:"forward",anchor:"transformers.LlamaForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.LlamaForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LlamaForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LlamaForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LlamaForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LlamaForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LlamaForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.LlamaForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L191",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
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
`}}),he=new Nt({props:{$$slots:{default:[zs]},$$scope:{ctx:T}}}),tt=new D({props:{title:"LlamaForTokenClassification",local:"transformers.LlamaForTokenClassification",headingTag:"h2"}}),ot=new z({props:{name:"class transformers.LlamaForTokenClassification",anchor:"transformers.LlamaForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L495"}}),nt=new z({props:{name:"forward",anchor:"transformers.LlamaForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LlamaForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LlamaForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LlamaForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LlamaForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LlamaForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LlamaForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.LlamaForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L254",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
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
`}}),fe=new Nt({props:{$$slots:{default:[Cs]},$$scope:{ctx:T}}}),st=new hs({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llama.md"}}),{c(){t=l("meta"),p=r(),o=l("p"),b=r(),$=l("p"),$.innerHTML=v,y=r(),C=l("div"),C.innerHTML=At,_e=r(),m(E.$$.fragment),Xt=r(),be=l("p"),be.innerHTML=Ln,Dt=r(),ke=l("p"),ke.innerHTML=zn,Yt=r(),m(ee.$$.fragment),St=r(),ye=l("p"),ye.innerHTML=Cn,Qt=r(),m(te.$$.fragment),Ot=r(),ve=l("p"),ve.innerHTML=Fn,Kt=r(),Te=l("p"),Te.innerHTML=qn,eo=r(),m(we.$$.fragment),to=r(),$e=l("p"),$e.innerHTML=Un,oo=r(),m(Me.$$.fragment),no=r(),oe=l("div"),oe.innerHTML=jn,so=r(),m(xe.$$.fragment),ao=r(),Le=l("ul"),Le.innerHTML=In,ro=r(),m(ze.$$.fragment),io=r(),I=l("div"),m(Ce.$$.fragment),Fo=r(),dt=l("p"),dt.innerHTML=Wn,qo=r(),ct=l("p"),ct.innerHTML=Jn,Uo=r(),m(ne.$$.fragment),lo=r(),m(Fe.$$.fragment),co=r(),F=l("div"),m(qe.$$.fragment),jo=r(),pt=l("p"),pt.textContent=Zn,Io=r(),mt=l("div"),m(Ue.$$.fragment),Wo=r(),se=l("div"),m(je.$$.fragment),Jo=r(),ut=l("p"),ut.innerHTML=Bn,Zo=r(),J=l("div"),m(Ie.$$.fragment),Bo=r(),ht=l("p"),ht.textContent=Pn,Po=r(),m(ae.$$.fragment),Eo=r(),ft=l("p"),ft.textContent=En,Go=r(),re=l("div"),m(We.$$.fragment),No=r(),gt=l("p"),gt.textContent=Gn,po=r(),m(Je.$$.fragment),mo=r(),w=l("div"),m(Ze.$$.fragment),Ao=r(),_t=l("p"),_t.textContent=Nn,Ho=r(),bt=l("p"),bt.textContent=An,Vo=r(),m(ie.$$.fragment),Ro=r(),kt=l("p"),kt.innerHTML=Hn,Xo=r(),yt=l("p"),yt.innerHTML=Vn,Do=r(),vt=l("div"),m(Be.$$.fragment),Yo=r(),le=l("div"),m(Pe.$$.fragment),So=r(),Tt=l("p"),Tt.innerHTML=Rn,Qo=r(),G=l("div"),m(Ee.$$.fragment),Oo=r(),wt=l("p"),wt.innerHTML=Xn,Ko=r(),$t=l("p"),$t.textContent=Dn,en=r(),de=l("div"),m(Ge.$$.fragment),tn=r(),Mt=l("p"),Mt.innerHTML=Yn,on=r(),xt=l("div"),m(Ne.$$.fragment),uo=r(),m(Ae.$$.fragment),ho=r(),q=l("div"),m(He.$$.fragment),nn=r(),Lt=l("p"),Lt.textContent=Sn,sn=r(),zt=l("p"),zt.innerHTML=Qn,an=r(),Ct=l("p"),Ct.innerHTML=On,rn=r(),N=l("div"),m(Ve.$$.fragment),ln=r(),Ft=l("p"),Ft.innerHTML=Kn,dn=r(),m(ce.$$.fragment),fo=r(),m(Re.$$.fragment),go=r(),U=l("div"),m(Xe.$$.fragment),cn=r(),qt=l("p"),qt.textContent=es,pn=r(),Ut=l("p"),Ut.innerHTML=ts,mn=r(),jt=l("p"),jt.innerHTML=os,un=r(),Z=l("div"),m(De.$$.fragment),hn=r(),It=l("p"),It.innerHTML=ns,fn=r(),m(pe.$$.fragment),gn=r(),m(me.$$.fragment),_o=r(),m(Ye.$$.fragment),bo=r(),Y=l("div"),m(Se.$$.fragment),_n=r(),A=l("div"),m(Qe.$$.fragment),bn=r(),Wt=l("p"),Wt.innerHTML=ss,kn=r(),m(ue.$$.fragment),ko=r(),m(Oe.$$.fragment),yo=r(),S=l("div"),m(Ke.$$.fragment),yn=r(),H=l("div"),m(et.$$.fragment),vn=r(),Jt=l("p"),Jt.innerHTML=as,Tn=r(),m(he.$$.fragment),vo=r(),m(tt.$$.fragment),To=r(),Q=l("div"),m(ot.$$.fragment),wn=r(),V=l("div"),m(nt.$$.fragment),$n=r(),Zt=l("p"),Zt.innerHTML=rs,Mn=r(),m(fe.$$.fragment),wo=r(),m(st.$$.fragment),$o=r(),Ht=l("p"),this.h()},l(e){const n=ms("svelte-u9bgzb",document.head);t=d(n,"META",{name:!0,content:!0}),n.forEach(s),p=i(e),o=d(e,"P",{}),L(o).forEach(s),b=i(e),$=d(e,"P",{"data-svelte-h":!0}),k($)!=="svelte-1qy8a1t"&&($.innerHTML=v),y=i(e),C=d(e,"DIV",{style:!0,"data-svelte-h":!0}),k(C)!=="svelte-11gpmgv"&&(C.innerHTML=At),_e=i(e),u(E.$$.fragment,e),Xt=i(e),be=d(e,"P",{"data-svelte-h":!0}),k(be)!=="svelte-1y53rsp"&&(be.innerHTML=Ln),Dt=i(e),ke=d(e,"P",{"data-svelte-h":!0}),k(ke)!=="svelte-cvrdq4"&&(ke.innerHTML=zn),Yt=i(e),u(ee.$$.fragment,e),St=i(e),ye=d(e,"P",{"data-svelte-h":!0}),k(ye)!=="svelte-x9rs6r"&&(ye.innerHTML=Cn),Qt=i(e),u(te.$$.fragment,e),Ot=i(e),ve=d(e,"P",{"data-svelte-h":!0}),k(ve)!=="svelte-nf5ooi"&&(ve.innerHTML=Fn),Kt=i(e),Te=d(e,"P",{"data-svelte-h":!0}),k(Te)!=="svelte-w36i1c"&&(Te.innerHTML=qn),eo=i(e),u(we.$$.fragment,e),to=i(e),$e=d(e,"P",{"data-svelte-h":!0}),k($e)!=="svelte-w3z5ks"&&($e.innerHTML=Un),oo=i(e),u(Me.$$.fragment,e),no=i(e),oe=d(e,"DIV",{class:!0,"data-svelte-h":!0}),k(oe)!=="svelte-phc3mn"&&(oe.innerHTML=jn),so=i(e),u(xe.$$.fragment,e),ao=i(e),Le=d(e,"UL",{"data-svelte-h":!0}),k(Le)!=="svelte-1pzp5g6"&&(Le.innerHTML=In),ro=i(e),u(ze.$$.fragment,e),io=i(e),I=d(e,"DIV",{class:!0});var B=L(I);u(Ce.$$.fragment,B),Fo=i(B),dt=d(B,"P",{"data-svelte-h":!0}),k(dt)!=="svelte-7fhu95"&&(dt.innerHTML=Wn),qo=i(B),ct=d(B,"P",{"data-svelte-h":!0}),k(ct)!=="svelte-1ek1ss9"&&(ct.innerHTML=Jn),Uo=i(B),u(ne.$$.fragment,B),B.forEach(s),lo=i(e),u(Fe.$$.fragment,e),co=i(e),F=d(e,"DIV",{class:!0});var j=L(F);u(qe.$$.fragment,j),jo=i(j),pt=d(j,"P",{"data-svelte-h":!0}),k(pt)!=="svelte-qfiu5a"&&(pt.textContent=Zn),Io=i(j),mt=d(j,"DIV",{class:!0});var Vt=L(mt);u(Ue.$$.fragment,Vt),Vt.forEach(s),Wo=i(j),se=d(j,"DIV",{class:!0});var at=L(se);u(je.$$.fragment,at),Jo=i(at),ut=d(at,"P",{"data-svelte-h":!0}),k(ut)!=="svelte-1f4f5kp"&&(ut.innerHTML=Bn),at.forEach(s),Zo=i(j),J=d(j,"DIV",{class:!0});var P=L(J);u(Ie.$$.fragment,P),Bo=i(P),ht=d(P,"P",{"data-svelte-h":!0}),k(ht)!=="svelte-13bfd60"&&(ht.textContent=Pn),Po=i(P),u(ae.$$.fragment,P),Eo=i(P),ft=d(P,"P",{"data-svelte-h":!0}),k(ft)!=="svelte-wtrslu"&&(ft.textContent=En),P.forEach(s),Go=i(j),re=d(j,"DIV",{class:!0});var rt=L(re);u(We.$$.fragment,rt),No=i(rt),gt=d(rt,"P",{"data-svelte-h":!0}),k(gt)!=="svelte-1slb66l"&&(gt.textContent=Gn),rt.forEach(s),j.forEach(s),po=i(e),u(Je.$$.fragment,e),mo=i(e),w=d(e,"DIV",{class:!0});var M=L(w);u(Ze.$$.fragment,M),Ao=i(M),_t=d(M,"P",{"data-svelte-h":!0}),k(_t)!=="svelte-15tdcz8"&&(_t.textContent=Nn),Ho=i(M),bt=d(M,"P",{"data-svelte-h":!0}),k(bt)!=="svelte-llhmpa"&&(bt.textContent=An),Vo=i(M),u(ie.$$.fragment,M),Ro=i(M),kt=d(M,"P",{"data-svelte-h":!0}),k(kt)!=="svelte-cnb6q1"&&(kt.innerHTML=Hn),Xo=i(M),yt=d(M,"P",{"data-svelte-h":!0}),k(yt)!=="svelte-gxzj9w"&&(yt.innerHTML=Vn),Do=i(M),vt=d(M,"DIV",{class:!0});var Rt=L(vt);u(Be.$$.fragment,Rt),Rt.forEach(s),Yo=i(M),le=d(M,"DIV",{class:!0});var it=L(le);u(Pe.$$.fragment,it),So=i(it),Tt=d(it,"P",{"data-svelte-h":!0}),k(Tt)!=="svelte-1wmjg8a"&&(Tt.innerHTML=Rn),it.forEach(s),Qo=i(M),G=d(M,"DIV",{class:!0});var O=L(G);u(Ee.$$.fragment,O),Oo=i(O),wt=d(O,"P",{"data-svelte-h":!0}),k(wt)!=="svelte-zj1vf1"&&(wt.innerHTML=Xn),Ko=i(O),$t=d(O,"P",{"data-svelte-h":!0}),k($t)!=="svelte-9vptpw"&&($t.textContent=Dn),O.forEach(s),en=i(M),de=d(M,"DIV",{class:!0});var lt=L(de);u(Ge.$$.fragment,lt),tn=i(lt),Mt=d(lt,"P",{"data-svelte-h":!0}),k(Mt)!=="svelte-nfci2w"&&(Mt.innerHTML=Yn),lt.forEach(s),on=i(M),xt=d(M,"DIV",{class:!0});var is=L(xt);u(Ne.$$.fragment,is),is.forEach(s),M.forEach(s),uo=i(e),u(Ae.$$.fragment,e),ho=i(e),q=d(e,"DIV",{class:!0});var R=L(q);u(He.$$.fragment,R),nn=i(R),Lt=d(R,"P",{"data-svelte-h":!0}),k(Lt)!=="svelte-ahmvbp"&&(Lt.textContent=Sn),sn=i(R),zt=d(R,"P",{"data-svelte-h":!0}),k(zt)!=="svelte-q52n56"&&(zt.innerHTML=Qn),an=i(R),Ct=d(R,"P",{"data-svelte-h":!0}),k(Ct)!=="svelte-hswkmf"&&(Ct.innerHTML=On),rn=i(R),N=d(R,"DIV",{class:!0});var Bt=L(N);u(Ve.$$.fragment,Bt),ln=i(Bt),Ft=d(Bt,"P",{"data-svelte-h":!0}),k(Ft)!=="svelte-1wrnj28"&&(Ft.innerHTML=Kn),dn=i(Bt),u(ce.$$.fragment,Bt),Bt.forEach(s),R.forEach(s),fo=i(e),u(Re.$$.fragment,e),go=i(e),U=d(e,"DIV",{class:!0});var X=L(U);u(Xe.$$.fragment,X),cn=i(X),qt=d(X,"P",{"data-svelte-h":!0}),k(qt)!=="svelte-a2k4ga"&&(qt.textContent=es),pn=i(X),Ut=d(X,"P",{"data-svelte-h":!0}),k(Ut)!=="svelte-q52n56"&&(Ut.innerHTML=ts),mn=i(X),jt=d(X,"P",{"data-svelte-h":!0}),k(jt)!=="svelte-hswkmf"&&(jt.innerHTML=os),un=i(X),Z=d(X,"DIV",{class:!0});var ge=L(Z);u(De.$$.fragment,ge),hn=i(ge),It=d(ge,"P",{"data-svelte-h":!0}),k(It)!=="svelte-1p7qkf4"&&(It.innerHTML=ns),fn=i(ge),u(pe.$$.fragment,ge),gn=i(ge),u(me.$$.fragment,ge),ge.forEach(s),X.forEach(s),_o=i(e),u(Ye.$$.fragment,e),bo=i(e),Y=d(e,"DIV",{class:!0});var xo=L(Y);u(Se.$$.fragment,xo),_n=i(xo),A=d(xo,"DIV",{class:!0});var Pt=L(A);u(Qe.$$.fragment,Pt),bn=i(Pt),Wt=d(Pt,"P",{"data-svelte-h":!0}),k(Wt)!=="svelte-1sal4ui"&&(Wt.innerHTML=ss),kn=i(Pt),u(ue.$$.fragment,Pt),Pt.forEach(s),xo.forEach(s),ko=i(e),u(Oe.$$.fragment,e),yo=i(e),S=d(e,"DIV",{class:!0});var Lo=L(S);u(Ke.$$.fragment,Lo),yn=i(Lo),H=d(Lo,"DIV",{class:!0});var Et=L(H);u(et.$$.fragment,Et),vn=i(Et),Jt=d(Et,"P",{"data-svelte-h":!0}),k(Jt)!=="svelte-dyrov9"&&(Jt.innerHTML=as),Tn=i(Et),u(he.$$.fragment,Et),Et.forEach(s),Lo.forEach(s),vo=i(e),u(tt.$$.fragment,e),To=i(e),Q=d(e,"DIV",{class:!0});var zo=L(Q);u(ot.$$.fragment,zo),wn=i(zo),V=d(zo,"DIV",{class:!0});var Gt=L(V);u(nt.$$.fragment,Gt),$n=i(Gt),Zt=d(Gt,"P",{"data-svelte-h":!0}),k(Zt)!=="svelte-1py4aay"&&(Zt.innerHTML=rs),Mn=i(Gt),u(fe.$$.fragment,Gt),Gt.forEach(s),zo.forEach(s),wo=i(e),u(st.$$.fragment,e),$o=i(e),Ht=d(e,"P",{}),L(Ht).forEach(s),this.h()},h(){x(t,"name","hf:doc:metadata"),x(t,"content",qs),us(C,"float","right"),x(oe,"class","flex justify-center"),x(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(mt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(xt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){a(document.head,t),c(e,p,n),c(e,o,n),c(e,b,n),c(e,$,n),c(e,y,n),c(e,C,n),c(e,_e,n),h(E,e,n),c(e,Xt,n),c(e,be,n),c(e,Dt,n),c(e,ke,n),c(e,Yt,n),h(ee,e,n),c(e,St,n),c(e,ye,n),c(e,Qt,n),h(te,e,n),c(e,Ot,n),c(e,ve,n),c(e,Kt,n),c(e,Te,n),c(e,eo,n),h(we,e,n),c(e,to,n),c(e,$e,n),c(e,oo,n),h(Me,e,n),c(e,no,n),c(e,oe,n),c(e,so,n),h(xe,e,n),c(e,ao,n),c(e,Le,n),c(e,ro,n),h(ze,e,n),c(e,io,n),c(e,I,n),h(Ce,I,null),a(I,Fo),a(I,dt),a(I,qo),a(I,ct),a(I,Uo),h(ne,I,null),c(e,lo,n),h(Fe,e,n),c(e,co,n),c(e,F,n),h(qe,F,null),a(F,jo),a(F,pt),a(F,Io),a(F,mt),h(Ue,mt,null),a(F,Wo),a(F,se),h(je,se,null),a(se,Jo),a(se,ut),a(F,Zo),a(F,J),h(Ie,J,null),a(J,Bo),a(J,ht),a(J,Po),h(ae,J,null),a(J,Eo),a(J,ft),a(F,Go),a(F,re),h(We,re,null),a(re,No),a(re,gt),c(e,po,n),h(Je,e,n),c(e,mo,n),c(e,w,n),h(Ze,w,null),a(w,Ao),a(w,_t),a(w,Ho),a(w,bt),a(w,Vo),h(ie,w,null),a(w,Ro),a(w,kt),a(w,Xo),a(w,yt),a(w,Do),a(w,vt),h(Be,vt,null),a(w,Yo),a(w,le),h(Pe,le,null),a(le,So),a(le,Tt),a(w,Qo),a(w,G),h(Ee,G,null),a(G,Oo),a(G,wt),a(G,Ko),a(G,$t),a(w,en),a(w,de),h(Ge,de,null),a(de,tn),a(de,Mt),a(w,on),a(w,xt),h(Ne,xt,null),c(e,uo,n),h(Ae,e,n),c(e,ho,n),c(e,q,n),h(He,q,null),a(q,nn),a(q,Lt),a(q,sn),a(q,zt),a(q,an),a(q,Ct),a(q,rn),a(q,N),h(Ve,N,null),a(N,ln),a(N,Ft),a(N,dn),h(ce,N,null),c(e,fo,n),h(Re,e,n),c(e,go,n),c(e,U,n),h(Xe,U,null),a(U,cn),a(U,qt),a(U,pn),a(U,Ut),a(U,mn),a(U,jt),a(U,un),a(U,Z),h(De,Z,null),a(Z,hn),a(Z,It),a(Z,fn),h(pe,Z,null),a(Z,gn),h(me,Z,null),c(e,_o,n),h(Ye,e,n),c(e,bo,n),c(e,Y,n),h(Se,Y,null),a(Y,_n),a(Y,A),h(Qe,A,null),a(A,bn),a(A,Wt),a(A,kn),h(ue,A,null),c(e,ko,n),h(Oe,e,n),c(e,yo,n),c(e,S,n),h(Ke,S,null),a(S,yn),a(S,H),h(et,H,null),a(H,vn),a(H,Jt),a(H,Tn),h(he,H,null),c(e,vo,n),h(tt,e,n),c(e,To,n),c(e,Q,n),h(ot,Q,null),a(Q,wn),a(Q,V),h(nt,V,null),a(V,$n),a(V,Zt),a(V,Mn),h(fe,V,null),c(e,wo,n),h(st,e,n),c(e,$o,n),c(e,Ht,n),Mo=!0},p(e,[n]){const B={};n&2&&(B.$$scope={dirty:n,ctx:e}),ee.$set(B);const j={};n&2&&(j.$$scope={dirty:n,ctx:e}),te.$set(j);const Vt={};n&2&&(Vt.$$scope={dirty:n,ctx:e}),ne.$set(Vt);const at={};n&2&&(at.$$scope={dirty:n,ctx:e}),ae.$set(at);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),ie.$set(P);const rt={};n&2&&(rt.$$scope={dirty:n,ctx:e}),ce.$set(rt);const M={};n&2&&(M.$$scope={dirty:n,ctx:e}),pe.$set(M);const Rt={};n&2&&(Rt.$$scope={dirty:n,ctx:e}),me.$set(Rt);const it={};n&2&&(it.$$scope={dirty:n,ctx:e}),ue.$set(it);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),he.$set(O);const lt={};n&2&&(lt.$$scope={dirty:n,ctx:e}),fe.$set(lt)},i(e){Mo||(f(E.$$.fragment,e),f(ee.$$.fragment,e),f(te.$$.fragment,e),f(we.$$.fragment,e),f(Me.$$.fragment,e),f(xe.$$.fragment,e),f(ze.$$.fragment,e),f(Ce.$$.fragment,e),f(ne.$$.fragment,e),f(Fe.$$.fragment,e),f(qe.$$.fragment,e),f(Ue.$$.fragment,e),f(je.$$.fragment,e),f(Ie.$$.fragment,e),f(ae.$$.fragment,e),f(We.$$.fragment,e),f(Je.$$.fragment,e),f(Ze.$$.fragment,e),f(ie.$$.fragment,e),f(Be.$$.fragment,e),f(Pe.$$.fragment,e),f(Ee.$$.fragment,e),f(Ge.$$.fragment,e),f(Ne.$$.fragment,e),f(Ae.$$.fragment,e),f(He.$$.fragment,e),f(Ve.$$.fragment,e),f(ce.$$.fragment,e),f(Re.$$.fragment,e),f(Xe.$$.fragment,e),f(De.$$.fragment,e),f(pe.$$.fragment,e),f(me.$$.fragment,e),f(Ye.$$.fragment,e),f(Se.$$.fragment,e),f(Qe.$$.fragment,e),f(ue.$$.fragment,e),f(Oe.$$.fragment,e),f(Ke.$$.fragment,e),f(et.$$.fragment,e),f(he.$$.fragment,e),f(tt.$$.fragment,e),f(ot.$$.fragment,e),f(nt.$$.fragment,e),f(fe.$$.fragment,e),f(st.$$.fragment,e),Mo=!0)},o(e){g(E.$$.fragment,e),g(ee.$$.fragment,e),g(te.$$.fragment,e),g(we.$$.fragment,e),g(Me.$$.fragment,e),g(xe.$$.fragment,e),g(ze.$$.fragment,e),g(Ce.$$.fragment,e),g(ne.$$.fragment,e),g(Fe.$$.fragment,e),g(qe.$$.fragment,e),g(Ue.$$.fragment,e),g(je.$$.fragment,e),g(Ie.$$.fragment,e),g(ae.$$.fragment,e),g(We.$$.fragment,e),g(Je.$$.fragment,e),g(Ze.$$.fragment,e),g(ie.$$.fragment,e),g(Be.$$.fragment,e),g(Pe.$$.fragment,e),g(Ee.$$.fragment,e),g(Ge.$$.fragment,e),g(Ne.$$.fragment,e),g(Ae.$$.fragment,e),g(He.$$.fragment,e),g(Ve.$$.fragment,e),g(ce.$$.fragment,e),g(Re.$$.fragment,e),g(Xe.$$.fragment,e),g(De.$$.fragment,e),g(pe.$$.fragment,e),g(me.$$.fragment,e),g(Ye.$$.fragment,e),g(Se.$$.fragment,e),g(Qe.$$.fragment,e),g(ue.$$.fragment,e),g(Oe.$$.fragment,e),g(Ke.$$.fragment,e),g(et.$$.fragment,e),g(he.$$.fragment,e),g(tt.$$.fragment,e),g(ot.$$.fragment,e),g(nt.$$.fragment,e),g(fe.$$.fragment,e),g(st.$$.fragment,e),Mo=!1},d(e){e&&(s(p),s(o),s(b),s($),s(y),s(C),s(_e),s(Xt),s(be),s(Dt),s(ke),s(Yt),s(St),s(ye),s(Qt),s(Ot),s(ve),s(Kt),s(Te),s(eo),s(to),s($e),s(oo),s(no),s(oe),s(so),s(ao),s(Le),s(ro),s(io),s(I),s(lo),s(co),s(F),s(po),s(mo),s(w),s(uo),s(ho),s(q),s(fo),s(go),s(U),s(_o),s(bo),s(Y),s(ko),s(yo),s(S),s(vo),s(To),s(Q),s(wo),s($o),s(Ht)),s(t),_(E,e),_(ee,e),_(te,e),_(we,e),_(Me,e),_(xe,e),_(ze,e),_(Ce),_(ne),_(Fe,e),_(qe),_(Ue),_(je),_(Ie),_(ae),_(We),_(Je,e),_(Ze),_(ie),_(Be),_(Pe),_(Ee),_(Ge),_(Ne),_(Ae,e),_(He),_(Ve),_(ce),_(Re,e),_(Xe),_(De),_(pe),_(me),_(Ye,e),_(Se),_(Qe),_(ue),_(Oe,e),_(Ke),_(et),_(he),_(tt,e),_(ot),_(nt),_(fe),_(st,e)}}}const qs='{"title":"Llama","local":"llama","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"LlamaConfig","local":"transformers.LlamaConfig","sections":[],"depth":2},{"title":"LlamaTokenizer","local":"transformers.LlamaTokenizer","sections":[],"depth":2},{"title":"LlamaTokenizerFast","local":"transformers.LlamaTokenizerFast","sections":[],"depth":2},{"title":"LlamaModel","local":"transformers.LlamaModel","sections":[],"depth":2},{"title":"LlamaForCausalLM","local":"transformers.LlamaForCausalLM","sections":[],"depth":2},{"title":"LlamaForSequenceClassification","local":"transformers.LlamaForSequenceClassification","sections":[],"depth":2},{"title":"LlamaForQuestionAnswering","local":"transformers.LlamaForQuestionAnswering","sections":[],"depth":2},{"title":"LlamaForTokenClassification","local":"transformers.LlamaForTokenClassification","sections":[],"depth":2}],"depth":1}';function Us(T){return ds(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Gs extends cs{constructor(t){super(),ps(this,t,Us,Fs,ls,{})}}export{Gs as component};
