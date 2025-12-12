import{s as Un,o as Zn,n as z}from"../chunks/scheduler.18a86fab.js";import{S as xn,i as Pn,g as u,s as i,r as g,A as Wn,h,f as a,c as l,j as V,x as w,u as _,k as X,l as zn,y as m,a as d,v as b,d as T,t as y,w as M}from"../chunks/index.98837b22.js";import{T as De}from"../chunks/Tip.77304350.js";import{D as H}from"../chunks/Docstring.a1ef7999.js";import{C as Y}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ke}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ae,E as Fn}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as In,a as tn}from"../chunks/HfOption.6641485e.js";function On(v){let t,p='This model was contributed by <a href="https://huggingface.co/ArthurZ" rel="nofollow">ArthurZ</a>, <a href="https://huggingface.co/ybelkada" rel="nofollow">ybelkada</a>, and <a href="https://huggingface.co/patrickvonplaten" rel="nofollow">patrickvonplaten</a>.',n,r,f="Click on the OPT models in the right sidebar for more examples of how to apply OPT to different language tasks.";return{c(){t=u("p"),t.innerHTML=p,n=i(),r=u("p"),r.textContent=f},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-3ptpcm"&&(t.innerHTML=p),n=l(o),r=h(o,"P",{"data-svelte-h":!0}),w(r)!=="svelte-5fyd5t"&&(r.textContent=f)},m(o,c){d(o,t,c),d(o,n,c),d(o,r,c)},p:z,d(o){o&&(a(t),a(n),a(r))}}}function Bn(v){let t,p;return t=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMjBtb2RlbCUzRCUyMmZhY2Vib29rJTJGb3B0LTEyNW0lMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMjBkZXZpY2UlM0QwKSUwQXBpcGVsaW5lKCUyMk9uY2UlMjB1cG9uJTIwYSUyMHRpbWUlMkMlMjBpbiUyMGElMjBsYW5kJTIwZmFyJTJDJTIwZmFyJTIwYXdheSUyQyUyMiUyQyUyMG1heF9sZW5ndGglM0Q1MCUyQyUyMG51bV9yZXR1cm5fc2VxdWVuY2VzJTNEMSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(task=<span class="hljs-string">&quot;text-generation&quot;</span>, model=<span class="hljs-string">&quot;facebook/opt-125m&quot;</span>, dtype=torch.float16, device=<span class="hljs-number">0</span>)
pipeline(<span class="hljs-string">&quot;Once upon a time, in a land far, far away,&quot;</span>, max_length=<span class="hljs-number">50</span>, num_return_sequences=<span class="hljs-number">1</span>)`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){_(t.$$.fragment,n)},m(n,r){b(t,n,r),p=!0},p:z,i(n){p||(T(t.$$.fragment,n),p=!0)},o(n){y(t.$$.fragment,n),p=!1},d(n){M(t,n)}}}function qn(v){let t,p;return t=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZvcHQtMzUwbSUyMiUyQyUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZvcHQtMzUwbSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAoJTIyT25jZSUyMHVwb24lMjBhJTIwdGltZSUyQyUyMGluJTIwYSUyMGxhbmQlMjBmYXIlMkMlMjBmYXIlMjBhd2F5JTJDJTIwJTIyKSUwQSUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqbW9kZWxfaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QzMCUyQyUyMGRvX3NhbXBsZSUzREZhbHNlKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyklNUIwJTVE",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>, dtype=torch.float16, device_map=<span class="hljs-string">&quot;auto&quot;</span>, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>)

prompt = (<span class="hljs-string">&quot;Once upon a time, in a land far, far away, &quot;</span>)

model_inputs = tokenizer([prompt], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=<span class="hljs-number">30</span>, do_sample=<span class="hljs-literal">False</span>)
tokenizer.batch_decode(generated_ids)[<span class="hljs-number">0</span>]`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){_(t.$$.fragment,n)},m(n,r){b(t,n,r),p=!0},p:z,i(n){p||(T(t.$$.fragment,n),p=!0)},o(n){y(t.$$.fragment,n),p=!1},d(n){M(t,n)}}}function Vn(v){let t,p;return t=new Y({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQtZ2VuZXJhdGlvbiUyMC0tbW9kZWwlMjBmYWNlYm9vayUyRm9wdC0xMjVtJTIwLS1kZXZpY2UlMjAw",highlighted:'echo -e <span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span> | transformers run --task text-generation --model facebook/opt-125m --device <span class="hljs-number">0</span>',wrap:!1}}),{c(){g(t.$$.fragment)},l(n){_(t.$$.fragment,n)},m(n,r){b(t,n,r),p=!0},p:z,i(n){p||(T(t.$$.fragment,n),p=!0)},o(n){y(t.$$.fragment,n),p=!1},d(n){M(t,n)}}}function Xn(v){let t,p,n,r,f,o;return t=new tn({props:{id:"usage",option:"Pipeline",$$slots:{default:[Bn]},$$scope:{ctx:v}}}),n=new tn({props:{id:"usage",option:"AutoModel",$$slots:{default:[qn]},$$scope:{ctx:v}}}),f=new tn({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Vn]},$$scope:{ctx:v}}}),{c(){g(t.$$.fragment),p=i(),g(n.$$.fragment),r=i(),g(f.$$.fragment)},l(c){_(t.$$.fragment,c),p=l(c),_(n.$$.fragment,c),r=l(c),_(f.$$.fragment,c)},m(c,k){b(t,c,k),d(c,p,k),b(n,c,k),d(c,r,k),b(f,c,k),o=!0},p(c,k){const Ee={};k&2&&(Ee.$$scope={dirty:k,ctx:c}),t.$set(Ee);const re={};k&2&&(re.$$scope={dirty:k,ctx:c}),n.$set(re);const G={};k&2&&(G.$$scope={dirty:k,ctx:c}),f.$set(G)},i(c){o||(T(t.$$.fragment,c),T(n.$$.fragment,c),T(f.$$.fragment,c),o=!0)},o(c){y(t.$$.fragment,c),y(n.$$.fragment,c),y(f.$$.fragment,c),o=!1},d(c){c&&(a(p),a(r)),M(t,c),M(n,c),M(f,c)}}}function Gn(v){let t,p="Example:",n,r,f;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME9QVENvbmZpZyUyQyUyME9QVE1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyME9QVCUyMGZhY2Vib29rJTJGb3B0LWxhcmdlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME9QVENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBmYWNlYm9vayUyRm9wdC1sYXJnZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwT1BUTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OPTConfig, OPTModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a OPT facebook/opt-large style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = OPTConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the facebook/opt-large style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OPTModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=i(),g(r.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),n=l(o),_(r.$$.fragment,o)},m(o,c){d(o,t,c),d(o,n,c),b(r,o,c),f=!0},p:z,i(o){f||(T(r.$$.fragment,o),f=!0)},o(o){y(r.$$.fragment,o),f=!1},d(o){o&&(a(t),a(n)),M(r,o)}}}function Qn(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,r){d(n,t,r)},p:z,d(n){n&&a(t)}}}function Rn(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,r){d(n,t,r)},p:z,d(n){n&&a(t)}}}function Nn(v){let t,p="Example:",n,r,f;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBPUFRGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwT1BURm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGb3B0LTM1MG0lMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZvcHQtMzUwbSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, OPTForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = OPTForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious. I&#x27;m just a little bit of a weirdo.&quot;</span>`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=i(),g(r.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),n=l(o),_(r.$$.fragment,o)},m(o,c){d(o,t,c),d(o,n,c),b(r,o,c),f=!0},p:z,i(o){f||(T(r.$$.fragment,o),f=!0)},o(o){y(r.$$.fragment,o),f=!1},d(o){o&&(a(t),a(n)),M(r,o)}}}function Hn(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,r){d(n,t,r)},p:z,d(n){n&&a(t)}}}function Yn(v){let t,p="Example of single-label classification:",n,r,f;return r=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME9QVEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm9wdC0zNTBtJTIyKSUwQW1vZGVsJTIwJTNEJTIwT1BURm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZvcHQtMzUwbSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwT1BURm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZvcHQtMzUwbSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, OPTForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OPTForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OPTForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=i(),g(r.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-ykxpe4"&&(t.textContent=p),n=l(o),_(r.$$.fragment,o)},m(o,c){d(o,t,c),d(o,n,c),b(r,o,c),f=!0},p:z,i(o){f||(T(r.$$.fragment,o),f=!0)},o(o){y(r.$$.fragment,o),f=!1},d(o){o&&(a(t),a(n)),M(r,o)}}}function Sn(v){let t,p="Example of multi-label classification:",n,r,f;return r=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME9QVEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm9wdC0zNTBtJTIyKSUwQW1vZGVsJTIwJTNEJTIwT1BURm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZvcHQtMzUwbSUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwT1BURm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZmFjZWJvb2slMkZvcHQtMzUwbSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, OPTForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OPTForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OPTForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/opt-350m&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=i(),g(r.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-1l8e32d"&&(t.textContent=p),n=l(o),_(r.$$.fragment,o)},m(o,c){d(o,t,c),d(o,n,c),b(r,o,c),f=!0},p:z,i(o){f||(T(r.$$.fragment,o),f=!0)},o(o){y(r.$$.fragment,o),f=!1},d(o){o&&(a(t),a(n)),M(r,o)}}}function Ln(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,r){d(n,t,r)},p:z,d(n){n&&a(t)}}}function En(v){let t,p="Example:",n,r,f;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBPUFRGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9yY2gubWFudWFsX3NlZWQoNCklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm9wdC0zNTBtJTIyKSUwQSUwQSUyMyUyMG5vdGUlM0ElMjB3ZSUyMGFyZSUyMGxvYWRpbmclMjBhJTIwT1BURm9yUXVlc3Rpb25BbnN3ZXJpbmclMjBmcm9tJTIwdGhlJTIwaHViJTIwaGVyZSUyQyUwQSUyMyUyMHNvJTIwdGhlJTIwaGVhZCUyMHdpbGwlMjBiZSUyMHJhbmRvbWx5JTIwaW5pdGlhbGl6ZWQlMkMlMjBoZW5jZSUyMHRoZSUyMHByZWRpY3Rpb25zJTIwd2lsbCUyMGJlJTIwcmFuZG9tJTBBbW9kZWwlMjAlM0QlMjBPUFRGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZvcHQtMzUwbSUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBYW5zd2VyX29mZnNldCUyMCUzRCUyMGxlbih0b2tlbml6ZXIocXVlc3Rpb24pJTVCMCU1RCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCJTBBJTIwJTIwJTIwJTIwMCUyQyUyMGFuc3dlcl9vZmZzZXQlMjAlMkIlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfb2Zmc2V0JTIwJTJCJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElMEElNUQlMEFwcmVkaWN0ZWQlMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyklMEFwcmVkaWN0ZWQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, OPTForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>torch.manual_seed(<span class="hljs-number">4</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># note: we are loading a OPTForQuestionAnswering from the hub here,</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># so the head will be randomly initialized, hence the predictions will be random</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OPTForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;facebook/opt-350m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_offset = <span class="hljs-built_in">len</span>(tokenizer(question)[<span class="hljs-number">0</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>, answer_offset + answer_start_index : answer_offset + answer_end_index + <span class="hljs-number">1</span>
<span class="hljs-meta">... </span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted = tokenizer.decode(predict_answer_tokens)
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted
<span class="hljs-string">&#x27; a nice puppet&#x27;</span>`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=i(),g(r.$$.fragment)},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),n=l(o),_(r.$$.fragment,o)},m(o,c){d(o,t,c),d(o,n,c),b(r,o,c),f=!0},p:z,i(o){f||(T(r.$$.fragment,o),f=!0)},o(o){y(r.$$.fragment,o),f=!1},d(o){o&&(a(t),a(n)),M(r,o)}}}function An(v){let t,p,n,r,f,o="<em>This model was released on 2022-05-02 and added to Hugging Face Transformers on 2022-05-12.</em>",c,k,Ee='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',re,G,et,ie,nn='<a href="https://huggingface.co/papers/2205.01068" rel="nofollow">OPT</a> is a suite of open-source decoder-only pre-trained transformers whose parameters range from 125M to 175B. OPT models are designed for causal language modeling and aim to enable responsible and reproducible research at scale. OPT-175B is comparable in performance to GPT-3 with only 1/7th the carbon footprint.',tt,le,on='You can find all the original OPT checkpoints under the <a href="https://huggingface.co/collections/facebook/opt-66ed00e15599f02966818844" rel="nofollow">OPT</a> collection.',nt,S,ot,de,sn='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',st,L,at,ce,an='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',rt,pe,rn='The example below uses <a href="..quantization/bitsandbytes">bitsandbytes</a> to quantize the weights to 8-bits.',it,me,lt,ue,dt,he,ln="<li><p>OPT adds an <code>EOS</code> token <code>&lt;/s&gt;</code> to the beginning of every prompt.</p></li> <li><p>The <code>head_mask</code> argument is ignored if the attention implementation isn’t <code>&quot;eager&quot;</code>. Set <code>attn_implementation=&quot;eager&quot;</code> to enable the <code>head_mask</code>.</p></li>",ct,fe,pt,ge,dn='<li>Refer to this <a href="https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing" rel="nofollow">notebook</a> for an example of fine-tuning OPT with PEFT, bitsandbytes, and Transformers.</li> <li>The <a href="https://huggingface.co/blog/accelerate-large-models" rel="nofollow">How 🤗 Accelerate runs very large models thanks to PyTorch</a> blog post demonstrates how to run OPT for inference.</li>',mt,_e,ut,Z,be,$t,We,cn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTModel">OPTModel</a>. It is used to instantiate a OPT model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the OPT
<a href="https://huggingface.co/facebook/opt-350m" rel="nofollow">facebook/opt-350m</a> architecture.`,Jt,ze,pn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,jt,E,ht,Te,ft,j,ye,Ct,Fe,mn="The bare Opt Model outputting raw hidden-states without any specific head on top.",Ut,Ie,un=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Zt,Oe,hn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xt,Q,Me,Pt,Be,fn='The <a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTModel">OPTModel</a> forward method, overrides the <code>__call__</code> special method.',Wt,A,gt,we,_t,R,ve,zt,F,ke,Ft,qe,gn='The <a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForCausalLM">OPTForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',It,D,Ot,K,bt,$e,Tt,$,Je,Bt,Ve,_n="The OPT Model transformer with a sequence classification head on top (linear layer).",qt,Xe,bn=`<a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForSequenceClassification">OPTForSequenceClassification</a> uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do.`,Vt,Ge,Tn=`Since it does classification on the last token, it requires to know the position of the last token. If a
<code>pad_token_id</code> is defined in the configuration, it finds the last token that is not a padding token in each row. If
no <code>pad_token_id</code> is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when <code>inputs_embeds</code> are passed instead of <code>input_ids</code>, it does the same (take the last value in
each row of the batch).`,Xt,Qe,yn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Gt,Re,Mn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Qt,U,je,Rt,Ne,wn='The <a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForSequenceClassification">OPTForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Nt,ee,Ht,te,Yt,ne,yt,Ce,Mt,C,Ue,St,He,vn=`The Opt transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Lt,Ye,kn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Et,Se,$n=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,At,I,Ze,Dt,Le,Jn='The <a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTForQuestionAnswering">OPTForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Kt,oe,en,se,wt,xe,vt,Ae,kt;return G=new ae({props:{title:"OPT",local:"opt",headingTag:"h1"}}),S=new De({props:{warning:!1,$$slots:{default:[On]},$$scope:{ctx:v}}}),L=new In({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Xn]},$$scope:{ctx:v}}}),me=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwaW5mZXJfZGV2aWNlJTBBJTBBZGV2aWNlJTIwJTNEJTIwaW5mZXJfZGV2aWNlKCklMEElMEFibmJfY29uZmlnJTIwJTNEJTIwQml0c0FuZEJ5dGVzQ29uZmlnKGxvYWRfaW5fOGJpdCUzRFRydWUpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZvcHQtMTNiJTIyJTJDJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMkMlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEYm5iX2NvbmZpZykudG8oZGV2aWNlKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGb3B0LTEzYiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAoJTIyT25jZSUyMHVwb24lMjBhJTIwdGltZSUyQyUyMGluJTIwYSUyMGxhbmQlMjBmYXIlMkMlMjBmYXIlMjBhd2F5JTJDJTIwJTIyKSUwQSUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqbW9kZWxfaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QzMCUyQyUyMGRvX3NhbXBsZSUzREZhbHNlKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyklNUIwJTVE",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, infer_device

device = infer_device()

bnb_config = BitsAndBytesConfig(load_in_8bit=<span class="hljs-literal">True</span>)
model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/opt-13b&quot;</span>, dtype=torch.float16, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, quantization_config=bnb_config).to(device)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/opt-13b&quot;</span>)

prompt = (<span class="hljs-string">&quot;Once upon a time, in a land far, far away, &quot;</span>)

model_inputs = tokenizer([prompt], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=<span class="hljs-number">30</span>, do_sample=<span class="hljs-literal">False</span>)
tokenizer.batch_decode(generated_ids)[<span class="hljs-number">0</span>]`,wrap:!1}}),ue=new ae({props:{title:"Notes",local:"notes",headingTag:"h2"}}),fe=new ae({props:{title:"Resources",local:"resources",headingTag:"h2"}}),_e=new ae({props:{title:"OPTConfig",local:"transformers.OPTConfig",headingTag:"h2"}}),be=new H({props:{name:"class transformers.OPTConfig",anchor:"transformers.OPTConfig",parameters:[{name:"vocab_size",val:" = 50272"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"ffn_dim",val:" = 3072"},{name:"max_position_embeddings",val:" = 2048"},{name:"do_layer_norm_before",val:" = True"},{name:"_remove_final_layer_norm",val:" = False"},{name:"word_embed_proj_dim",val:" = None"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"num_attention_heads",val:" = 12"},{name:"activation_function",val:" = 'relu'"},{name:"layerdrop",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 2"},{name:"eos_token_id",val:" = 2"},{name:"enable_bias",val:" = True"},{name:"layer_norm_elementwise_affine",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OPTConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50272) &#x2014;
Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTModel">OPTModel</a>`,name:"vocab_size"},{anchor:"transformers.OPTConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.OPTConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"num_hidden_layers"},{anchor:"transformers.OPTConfig.ffn_dim",description:`<strong>ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"ffn_dim"},{anchor:"transformers.OPTConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.OPTConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.OPTConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.OPTConfig.do_layer_norm_before",description:`<strong>do_layer_norm_before</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to perform layer normalization before the attention block.`,name:"do_layer_norm_before"},{anchor:"transformers.OPTConfig.word_embed_proj_dim",description:`<strong>word_embed_proj_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
<code>word_embed_proj_dim</code> can be set to down-project word embeddings, <em>e.g.</em> <code>opt-350m</code>. Defaults to
<code>hidden_size</code>.`,name:"word_embed_proj_dim"},{anchor:"transformers.OPTConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.OPTConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.OPTConfig.layerdrop",description:`<strong>layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>) for more
details.`,name:"layerdrop"},{anchor:"transformers.OPTConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.OPTConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.OPTConfig.enable_bias",description:`<strong>enable_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not if the linear layers in the attention blocks should use the bias term.`,name:"enable_bias"},{anchor:"transformers.OPTConfig.layer_norm_elementwise_affine",description:`<strong>layer_norm_elementwise_affine</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not if the layer norms should have learnable parameters.`,name:"layer_norm_elementwise_affine"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/configuration_opt.py#L24"}}),E=new Ke({props:{anchor:"transformers.OPTConfig.example",$$slots:{default:[Gn]},$$scope:{ctx:v}}}),Te=new ae({props:{title:"OPTModel",local:"transformers.OPTModel",headingTag:"h2"}}),ye=new H({props:{name:"class transformers.OPTModel",anchor:"transformers.OPTModel",parameters:[{name:"config",val:": OPTConfig"}],parametersDescription:[{anchor:"transformers.OPTModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig">OPTConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py#L692"}}),Me=new H({props:{name:"forward",anchor:"transformers.OPTModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}],parametersDescription:[{anchor:"transformers.OPTModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.OPTModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.OPTModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.OPTModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.OPTModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.OPTModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.OPTModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OPTModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OPTModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.OPTModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.OPTModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py#L705",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig"
>OPTConfig</a>) and inputs.</p>
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
`}}),A=new De({props:{$$slots:{default:[Qn]},$$scope:{ctx:v}}}),we=new ae({props:{title:"OPTForCausalLM",local:"transformers.OPTForCausalLM",headingTag:"h2"}}),ve=new H({props:{name:"class transformers.OPTForCausalLM",anchor:"transformers.OPTForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py#L753"}}),ke=new H({props:{name:"forward",anchor:"transformers.OPTForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.OPTForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.OPTForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.OPTForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.OPTForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.OPTForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.OPTForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.OPTForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.OPTForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OPTForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OPTForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.OPTForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.OPTForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py#L778",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig"
>OPTConfig</a>) and inputs.</p>
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
`}}),D=new De({props:{$$slots:{default:[Rn]},$$scope:{ctx:v}}}),K=new Ke({props:{anchor:"transformers.OPTForCausalLM.forward.example",$$slots:{default:[Nn]},$$scope:{ctx:v}}}),$e=new ae({props:{title:"OPTForSequenceClassification",local:"transformers.OPTForSequenceClassification",headingTag:"h2"}}),Je=new H({props:{name:"class transformers.OPTForSequenceClassification",anchor:"transformers.OPTForSequenceClassification",parameters:[{name:"config",val:": OPTConfig"}],parametersDescription:[{anchor:"transformers.OPTForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig">OPTConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py#L877"}}),je=new H({props:{name:"forward",anchor:"transformers.OPTForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.OPTForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.OPTForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.OPTForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.OPTForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.OPTForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.OPTForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.OPTForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.OPTForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OPTForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OPTForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.OPTForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py#L887",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig"
>OPTConfig</a>) and inputs.</p>
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
`}}),ee=new De({props:{$$slots:{default:[Hn]},$$scope:{ctx:v}}}),te=new Ke({props:{anchor:"transformers.OPTForSequenceClassification.forward.example",$$slots:{default:[Yn]},$$scope:{ctx:v}}}),ne=new Ke({props:{anchor:"transformers.OPTForSequenceClassification.forward.example-2",$$slots:{default:[Sn]},$$scope:{ctx:v}}}),Ce=new ae({props:{title:"OPTForQuestionAnswering",local:"transformers.OPTForQuestionAnswering",headingTag:"h2"}}),Ue=new H({props:{name:"class transformers.OPTForQuestionAnswering",anchor:"transformers.OPTForQuestionAnswering",parameters:[{name:"config",val:": OPTConfig"}],parametersDescription:[{anchor:"transformers.OPTForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig">OPTConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py#L990"}}),Ze=new H({props:{name:"forward",anchor:"transformers.OPTForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.OPTForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.OPTForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.OPTForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.OPTForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.OPTForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.OPTForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.OPTForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.OPTForQuestionAnswering.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.OPTForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OPTForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OPTForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.OPTForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/opt/modeling_opt.py#L999",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/opt#transformers.OPTConfig"
>OPTConfig</a>) and inputs.</p>
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
`}}),oe=new De({props:{$$slots:{default:[Ln]},$$scope:{ctx:v}}}),se=new Ke({props:{anchor:"transformers.OPTForQuestionAnswering.forward.example",$$slots:{default:[En]},$$scope:{ctx:v}}}),xe=new Fn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/opt.md"}}),{c(){t=u("meta"),p=i(),n=u("p"),r=i(),f=u("p"),f.innerHTML=o,c=i(),k=u("div"),k.innerHTML=Ee,re=i(),g(G.$$.fragment),et=i(),ie=u("p"),ie.innerHTML=nn,tt=i(),le=u("p"),le.innerHTML=on,nt=i(),g(S.$$.fragment),ot=i(),de=u("p"),de.innerHTML=sn,st=i(),g(L.$$.fragment),at=i(),ce=u("p"),ce.innerHTML=an,rt=i(),pe=u("p"),pe.innerHTML=rn,it=i(),g(me.$$.fragment),lt=i(),g(ue.$$.fragment),dt=i(),he=u("ul"),he.innerHTML=ln,ct=i(),g(fe.$$.fragment),pt=i(),ge=u("ul"),ge.innerHTML=dn,mt=i(),g(_e.$$.fragment),ut=i(),Z=u("div"),g(be.$$.fragment),$t=i(),We=u("p"),We.innerHTML=cn,Jt=i(),ze=u("p"),ze.innerHTML=pn,jt=i(),g(E.$$.fragment),ht=i(),g(Te.$$.fragment),ft=i(),j=u("div"),g(ye.$$.fragment),Ct=i(),Fe=u("p"),Fe.textContent=mn,Ut=i(),Ie=u("p"),Ie.innerHTML=un,Zt=i(),Oe=u("p"),Oe.innerHTML=hn,xt=i(),Q=u("div"),g(Me.$$.fragment),Pt=i(),Be=u("p"),Be.innerHTML=fn,Wt=i(),g(A.$$.fragment),gt=i(),g(we.$$.fragment),_t=i(),R=u("div"),g(ve.$$.fragment),zt=i(),F=u("div"),g(ke.$$.fragment),Ft=i(),qe=u("p"),qe.innerHTML=gn,It=i(),g(D.$$.fragment),Ot=i(),g(K.$$.fragment),bt=i(),g($e.$$.fragment),Tt=i(),$=u("div"),g(Je.$$.fragment),Bt=i(),Ve=u("p"),Ve.textContent=_n,qt=i(),Xe=u("p"),Xe.innerHTML=bn,Vt=i(),Ge=u("p"),Ge.innerHTML=Tn,Xt=i(),Qe=u("p"),Qe.innerHTML=yn,Gt=i(),Re=u("p"),Re.innerHTML=Mn,Qt=i(),U=u("div"),g(je.$$.fragment),Rt=i(),Ne=u("p"),Ne.innerHTML=wn,Nt=i(),g(ee.$$.fragment),Ht=i(),g(te.$$.fragment),Yt=i(),g(ne.$$.fragment),yt=i(),g(Ce.$$.fragment),Mt=i(),C=u("div"),g(Ue.$$.fragment),St=i(),He=u("p"),He.innerHTML=vn,Lt=i(),Ye=u("p"),Ye.innerHTML=kn,Et=i(),Se=u("p"),Se.innerHTML=$n,At=i(),I=u("div"),g(Ze.$$.fragment),Dt=i(),Le=u("p"),Le.innerHTML=Jn,Kt=i(),g(oe.$$.fragment),en=i(),g(se.$$.fragment),wt=i(),g(xe.$$.fragment),vt=i(),Ae=u("p"),this.h()},l(e){const s=Wn("svelte-u9bgzb",document.head);t=h(s,"META",{name:!0,content:!0}),s.forEach(a),p=l(e),n=h(e,"P",{}),V(n).forEach(a),r=l(e),f=h(e,"P",{"data-svelte-h":!0}),w(f)!=="svelte-1h1ifur"&&(f.innerHTML=o),c=l(e),k=h(e,"DIV",{style:!0,"data-svelte-h":!0}),w(k)!=="svelte-1ou4yfx"&&(k.innerHTML=Ee),re=l(e),_(G.$$.fragment,e),et=l(e),ie=h(e,"P",{"data-svelte-h":!0}),w(ie)!=="svelte-1yxo72e"&&(ie.innerHTML=nn),tt=l(e),le=h(e,"P",{"data-svelte-h":!0}),w(le)!=="svelte-1bqyz8g"&&(le.innerHTML=on),nt=l(e),_(S.$$.fragment,e),ot=l(e),de=h(e,"P",{"data-svelte-h":!0}),w(de)!=="svelte-17pa8jt"&&(de.innerHTML=sn),st=l(e),_(L.$$.fragment,e),at=l(e),ce=h(e,"P",{"data-svelte-h":!0}),w(ce)!=="svelte-nf5ooi"&&(ce.innerHTML=an),rt=l(e),pe=h(e,"P",{"data-svelte-h":!0}),w(pe)!=="svelte-pbttlh"&&(pe.innerHTML=rn),it=l(e),_(me.$$.fragment,e),lt=l(e),_(ue.$$.fragment,e),dt=l(e),he=h(e,"UL",{"data-svelte-h":!0}),w(he)!=="svelte-1011rjd"&&(he.innerHTML=ln),ct=l(e),_(fe.$$.fragment,e),pt=l(e),ge=h(e,"UL",{"data-svelte-h":!0}),w(ge)!=="svelte-1itq7yk"&&(ge.innerHTML=dn),mt=l(e),_(_e.$$.fragment,e),ut=l(e),Z=h(e,"DIV",{class:!0});var O=V(Z);_(be.$$.fragment,O),$t=l(O),We=h(O,"P",{"data-svelte-h":!0}),w(We)!=="svelte-eg4n3s"&&(We.innerHTML=cn),Jt=l(O),ze=h(O,"P",{"data-svelte-h":!0}),w(ze)!=="svelte-1ek1ss9"&&(ze.innerHTML=pn),jt=l(O),_(E.$$.fragment,O),O.forEach(a),ht=l(e),_(Te.$$.fragment,e),ft=l(e),j=h(e,"DIV",{class:!0});var x=V(j);_(ye.$$.fragment,x),Ct=l(x),Fe=h(x,"P",{"data-svelte-h":!0}),w(Fe)!=="svelte-1txet49"&&(Fe.textContent=mn),Ut=l(x),Ie=h(x,"P",{"data-svelte-h":!0}),w(Ie)!=="svelte-q52n56"&&(Ie.innerHTML=un),Zt=l(x),Oe=h(x,"P",{"data-svelte-h":!0}),w(Oe)!=="svelte-hswkmf"&&(Oe.innerHTML=hn),xt=l(x),Q=h(x,"DIV",{class:!0});var N=V(Q);_(Me.$$.fragment,N),Pt=l(N),Be=h(N,"P",{"data-svelte-h":!0}),w(Be)!=="svelte-1ifkgho"&&(Be.innerHTML=fn),Wt=l(N),_(A.$$.fragment,N),N.forEach(a),x.forEach(a),gt=l(e),_(we.$$.fragment,e),_t=l(e),R=h(e,"DIV",{class:!0});var Pe=V(R);_(ve.$$.fragment,Pe),zt=l(Pe),F=h(Pe,"DIV",{class:!0});var B=V(F);_(ke.$$.fragment,B),Ft=l(B),qe=h(B,"P",{"data-svelte-h":!0}),w(qe)!=="svelte-93serw"&&(qe.innerHTML=gn),It=l(B),_(D.$$.fragment,B),Ot=l(B),_(K.$$.fragment,B),B.forEach(a),Pe.forEach(a),bt=l(e),_($e.$$.fragment,e),Tt=l(e),$=h(e,"DIV",{class:!0});var J=V($);_(Je.$$.fragment,J),Bt=l(J),Ve=h(J,"P",{"data-svelte-h":!0}),w(Ve)!=="svelte-1hcah69"&&(Ve.textContent=_n),qt=l(J),Xe=h(J,"P",{"data-svelte-h":!0}),w(Xe)!=="svelte-pplkhn"&&(Xe.innerHTML=bn),Vt=l(J),Ge=h(J,"P",{"data-svelte-h":!0}),w(Ge)!=="svelte-10ugs3m"&&(Ge.innerHTML=Tn),Xt=l(J),Qe=h(J,"P",{"data-svelte-h":!0}),w(Qe)!=="svelte-q52n56"&&(Qe.innerHTML=yn),Gt=l(J),Re=h(J,"P",{"data-svelte-h":!0}),w(Re)!=="svelte-hswkmf"&&(Re.innerHTML=Mn),Qt=l(J),U=h(J,"DIV",{class:!0});var P=V(U);_(je.$$.fragment,P),Rt=l(P),Ne=h(P,"P",{"data-svelte-h":!0}),w(Ne)!=="svelte-d6dfqm"&&(Ne.innerHTML=wn),Nt=l(P),_(ee.$$.fragment,P),Ht=l(P),_(te.$$.fragment,P),Yt=l(P),_(ne.$$.fragment,P),P.forEach(a),J.forEach(a),yt=l(e),_(Ce.$$.fragment,e),Mt=l(e),C=h(e,"DIV",{class:!0});var W=V(C);_(Ue.$$.fragment,W),St=l(W),He=h(W,"P",{"data-svelte-h":!0}),w(He)!=="svelte-8u21e4"&&(He.innerHTML=vn),Lt=l(W),Ye=h(W,"P",{"data-svelte-h":!0}),w(Ye)!=="svelte-q52n56"&&(Ye.innerHTML=kn),Et=l(W),Se=h(W,"P",{"data-svelte-h":!0}),w(Se)!=="svelte-hswkmf"&&(Se.innerHTML=$n),At=l(W),I=h(W,"DIV",{class:!0});var q=V(I);_(Ze.$$.fragment,q),Dt=l(q),Le=h(q,"P",{"data-svelte-h":!0}),w(Le)!=="svelte-1y9ssj4"&&(Le.innerHTML=Jn),Kt=l(q),_(oe.$$.fragment,q),en=l(q),_(se.$$.fragment,q),q.forEach(a),W.forEach(a),wt=l(e),_(xe.$$.fragment,e),vt=l(e),Ae=h(e,"P",{}),V(Ae).forEach(a),this.h()},h(){X(t,"name","hf:doc:metadata"),X(t,"content",Dn),zn(k,"float","right"),X(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){m(document.head,t),d(e,p,s),d(e,n,s),d(e,r,s),d(e,f,s),d(e,c,s),d(e,k,s),d(e,re,s),b(G,e,s),d(e,et,s),d(e,ie,s),d(e,tt,s),d(e,le,s),d(e,nt,s),b(S,e,s),d(e,ot,s),d(e,de,s),d(e,st,s),b(L,e,s),d(e,at,s),d(e,ce,s),d(e,rt,s),d(e,pe,s),d(e,it,s),b(me,e,s),d(e,lt,s),b(ue,e,s),d(e,dt,s),d(e,he,s),d(e,ct,s),b(fe,e,s),d(e,pt,s),d(e,ge,s),d(e,mt,s),b(_e,e,s),d(e,ut,s),d(e,Z,s),b(be,Z,null),m(Z,$t),m(Z,We),m(Z,Jt),m(Z,ze),m(Z,jt),b(E,Z,null),d(e,ht,s),b(Te,e,s),d(e,ft,s),d(e,j,s),b(ye,j,null),m(j,Ct),m(j,Fe),m(j,Ut),m(j,Ie),m(j,Zt),m(j,Oe),m(j,xt),m(j,Q),b(Me,Q,null),m(Q,Pt),m(Q,Be),m(Q,Wt),b(A,Q,null),d(e,gt,s),b(we,e,s),d(e,_t,s),d(e,R,s),b(ve,R,null),m(R,zt),m(R,F),b(ke,F,null),m(F,Ft),m(F,qe),m(F,It),b(D,F,null),m(F,Ot),b(K,F,null),d(e,bt,s),b($e,e,s),d(e,Tt,s),d(e,$,s),b(Je,$,null),m($,Bt),m($,Ve),m($,qt),m($,Xe),m($,Vt),m($,Ge),m($,Xt),m($,Qe),m($,Gt),m($,Re),m($,Qt),m($,U),b(je,U,null),m(U,Rt),m(U,Ne),m(U,Nt),b(ee,U,null),m(U,Ht),b(te,U,null),m(U,Yt),b(ne,U,null),d(e,yt,s),b(Ce,e,s),d(e,Mt,s),d(e,C,s),b(Ue,C,null),m(C,St),m(C,He),m(C,Lt),m(C,Ye),m(C,Et),m(C,Se),m(C,At),m(C,I),b(Ze,I,null),m(I,Dt),m(I,Le),m(I,Kt),b(oe,I,null),m(I,en),b(se,I,null),d(e,wt,s),b(xe,e,s),d(e,vt,s),d(e,Ae,s),kt=!0},p(e,[s]){const O={};s&2&&(O.$$scope={dirty:s,ctx:e}),S.$set(O);const x={};s&2&&(x.$$scope={dirty:s,ctx:e}),L.$set(x);const N={};s&2&&(N.$$scope={dirty:s,ctx:e}),E.$set(N);const Pe={};s&2&&(Pe.$$scope={dirty:s,ctx:e}),A.$set(Pe);const B={};s&2&&(B.$$scope={dirty:s,ctx:e}),D.$set(B);const J={};s&2&&(J.$$scope={dirty:s,ctx:e}),K.$set(J);const P={};s&2&&(P.$$scope={dirty:s,ctx:e}),ee.$set(P);const W={};s&2&&(W.$$scope={dirty:s,ctx:e}),te.$set(W);const q={};s&2&&(q.$$scope={dirty:s,ctx:e}),ne.$set(q);const jn={};s&2&&(jn.$$scope={dirty:s,ctx:e}),oe.$set(jn);const Cn={};s&2&&(Cn.$$scope={dirty:s,ctx:e}),se.$set(Cn)},i(e){kt||(T(G.$$.fragment,e),T(S.$$.fragment,e),T(L.$$.fragment,e),T(me.$$.fragment,e),T(ue.$$.fragment,e),T(fe.$$.fragment,e),T(_e.$$.fragment,e),T(be.$$.fragment,e),T(E.$$.fragment,e),T(Te.$$.fragment,e),T(ye.$$.fragment,e),T(Me.$$.fragment,e),T(A.$$.fragment,e),T(we.$$.fragment,e),T(ve.$$.fragment,e),T(ke.$$.fragment,e),T(D.$$.fragment,e),T(K.$$.fragment,e),T($e.$$.fragment,e),T(Je.$$.fragment,e),T(je.$$.fragment,e),T(ee.$$.fragment,e),T(te.$$.fragment,e),T(ne.$$.fragment,e),T(Ce.$$.fragment,e),T(Ue.$$.fragment,e),T(Ze.$$.fragment,e),T(oe.$$.fragment,e),T(se.$$.fragment,e),T(xe.$$.fragment,e),kt=!0)},o(e){y(G.$$.fragment,e),y(S.$$.fragment,e),y(L.$$.fragment,e),y(me.$$.fragment,e),y(ue.$$.fragment,e),y(fe.$$.fragment,e),y(_e.$$.fragment,e),y(be.$$.fragment,e),y(E.$$.fragment,e),y(Te.$$.fragment,e),y(ye.$$.fragment,e),y(Me.$$.fragment,e),y(A.$$.fragment,e),y(we.$$.fragment,e),y(ve.$$.fragment,e),y(ke.$$.fragment,e),y(D.$$.fragment,e),y(K.$$.fragment,e),y($e.$$.fragment,e),y(Je.$$.fragment,e),y(je.$$.fragment,e),y(ee.$$.fragment,e),y(te.$$.fragment,e),y(ne.$$.fragment,e),y(Ce.$$.fragment,e),y(Ue.$$.fragment,e),y(Ze.$$.fragment,e),y(oe.$$.fragment,e),y(se.$$.fragment,e),y(xe.$$.fragment,e),kt=!1},d(e){e&&(a(p),a(n),a(r),a(f),a(c),a(k),a(re),a(et),a(ie),a(tt),a(le),a(nt),a(ot),a(de),a(st),a(at),a(ce),a(rt),a(pe),a(it),a(lt),a(dt),a(he),a(ct),a(pt),a(ge),a(mt),a(ut),a(Z),a(ht),a(ft),a(j),a(gt),a(_t),a(R),a(bt),a(Tt),a($),a(yt),a(Mt),a(C),a(wt),a(vt),a(Ae)),a(t),M(G,e),M(S,e),M(L,e),M(me,e),M(ue,e),M(fe,e),M(_e,e),M(be),M(E),M(Te,e),M(ye),M(Me),M(A),M(we,e),M(ve),M(ke),M(D),M(K),M($e,e),M(Je),M(je),M(ee),M(te),M(ne),M(Ce,e),M(Ue),M(Ze),M(oe),M(se),M(xe,e)}}}const Dn='{"title":"OPT","local":"opt","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"OPTConfig","local":"transformers.OPTConfig","sections":[],"depth":2},{"title":"OPTModel","local":"transformers.OPTModel","sections":[],"depth":2},{"title":"OPTForCausalLM","local":"transformers.OPTForCausalLM","sections":[],"depth":2},{"title":"OPTForSequenceClassification","local":"transformers.OPTForSequenceClassification","sections":[],"depth":2},{"title":"OPTForQuestionAnswering","local":"transformers.OPTForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Kn(v){return Zn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class lo extends xn{constructor(t){super(),Pn(this,t,Kn,An,Un,{})}}export{lo as component};
