import{s as cs,o as ds,n as xe}from"../chunks/scheduler.18a86fab.js";import{S as ps,i as ms,g as c,s as a,r as y,A as Ms,h as d,f as n,c as l,j as C,x as h,u as g,k as Z,l as us,y as r,a as i,v as J,d as f,t as T,w as U}from"../chunks/index.98837b22.js";import{T as zt}from"../chunks/Tip.77304350.js";import{D as S}from"../chunks/Docstring.a1ef7999.js";import{C as Ne}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as hs}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Ge,E as ys}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as gs,a as is}from"../chunks/HfOption.6641485e.js";function Js(w){let t,u='This model was contributed by <a href="https://huggingface.co/saurabhdash" rel="nofollow">saurabhdash</a> and <a href="https://huggingface.co/yonigozlan" rel="nofollow">yonigozlan</a>.',o,M,p="Click on the Aya Vision models in the right sidebar for more examples of how to apply Aya Vision to different image-to-text tasks.";return{c(){t=c("p"),t.innerHTML=u,o=a(),M=c("p"),M.textContent=p},l(m){t=d(m,"P",{"data-svelte-h":!0}),h(t)!=="svelte-116ksa3"&&(t.innerHTML=u),o=l(m),M=d(m,"P",{"data-svelte-h":!0}),h(M)!=="svelte-wsdy0"&&(M.textContent=p)},m(m,j){i(m,t,j),i(m,o,j),i(m,M,j)},p:xe,d(m){m&&(n(t),n(o),n(M))}}}function fs(w){let t,u;return t=new Ne({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMHBpcGVsaW5lJTBBJTBBcGlwZSUyMCUzRCUyMHBpcGVsaW5lKG1vZGVsJTNEJTIyQ29oZXJlTGFicyUyRmF5YS12aXNpb24tOGIlMjIlMkMlMjB0YXNrJTNEJTIyaW1hZ2UtdGV4dC10by10ZXh0JTIyJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIpJTBBJTBBJTIzJTIwRm9ybWF0JTIwbWVzc2FnZSUyMHdpdGglMjB0aGUlMjBheWEtdmlzaW9uJTIwY2hhdCUyMHRlbXBsYXRlJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlMkMlMjAlMjJ1cmwlMjIlM0ElMjAlMjJodHRwcyUzQSUyRiUyRm1lZGlhLmlzdG9ja3Bob3RvLmNvbSUyRmlkJTJGNDU4MDEyMDU3JTJGcGhvdG8lMkZpc3RhbmJ1bC10dXJrZXkuanBnJTNGcyUzRDYxMng2MTIlMjZ3JTNEMCUyNmslM0QyMCUyNmMlM0Rxb2dBT1Z2a3BmVXlxTFVNcl9YSlF5cS1Ia0FDWHlZVVNaYktoQmxQcnhvJTNEJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyQnUlMjByZXNpbWRlJTIwaGFuZ2klMjBhbiVDNCVCMXQlMjBnJUMzJUI2c3RlcmlsbWVrdGVkaXIlM0YlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlNUQlN0QlMkMlMEElMjAlMjAlMjAlMjAlNUQlMEFvdXRwdXRzJTIwJTNEJTIwcGlwZSh0ZXh0JTNEbWVzc2FnZXMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDMwMCUyQyUyMHJldHVybl9mdWxsX3RleHQlM0RGYWxzZSklMEElMEFwcmludChvdXRwdXRzKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(model=<span class="hljs-string">&quot;CohereLabs/aya-vision-8b&quot;</span>, task=<span class="hljs-string">&quot;image-text-to-text&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>)

<span class="hljs-comment"># Format message with the aya-vision chat template</span>
messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
     <span class="hljs-string">&quot;content&quot;</span>: [
       {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&amp;w=0&amp;k=20&amp;c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo=&quot;</span>},
        {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;Bu resimde hangi anıt gösterilmektedir?&quot;</span>},
    ]},
    ]
outputs = pipe(text=messages, max_new_tokens=<span class="hljs-number">300</span>, return_full_text=<span class="hljs-literal">False</span>)

<span class="hljs-built_in">print</span>(outputs)`,wrap:!1}}),{c(){y(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,M){J(t,o,M),u=!0},p:xe,i(o){u||(f(t.$$.fragment,o),u=!0)},o(o){T(t.$$.fragment,o),u=!1},d(o){U(t,o)}}}function Ts(w){let t,u;return t=new Ne({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMCdnaXQlMkJodHRwcyUzQSUyRiUyRmdpdGh1Yi5jb20lMkZodWdnaW5nZmFjZSUyRnRyYW5zZm9ybWVycy5naXQlNDB2NC40OS4wLUF5YSUyMFZpc2lvbiclMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBBdXRvUHJvY2Vzc29yJTJDJTIwQXV0b01vZGVsRm9ySW1hZ2VUZXh0VG9UZXh0JTBBJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJDb2hlcmVMYWJzJTJGYXlhLXZpc2lvbi04YiUyMiUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9ySW1hZ2VUZXh0VG9UZXh0LmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjBtb2RlbF9pZCUyQyUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTBBKSUwQSUwQSUyMyUyMEZvcm1hdCUyMG1lc3NhZ2UlMjB3aXRoJTIwdGhlJTIwYXlhLXZpc2lvbiUyMGNoYXQlMjB0ZW1wbGF0ZSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTJDJTIwJTIydXJsJTIyJTNBJTIwJTIyaHR0cHMlM0ElMkYlMkZwYnMudHdpbWcuY29tJTJGbWVkaWElMkZGeDdZdmZRV1lBSXA2clolM0Zmb3JtYXQlM0RqcGclMjZuYW1lJTNEbWVkaXVtJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyJUUwJUE0JTlBJUUwJUE0JUJGJUUwJUE0JUE0JUUwJUE1JThEJUUwJUE0JUIwJTIwJUUwJUE0JUFFJUUwJUE1JTg3JUUwJUE0JTgyJTIwJUUwJUE0JUIyJUUwJUE0JUJGJUUwJUE0JTk2JUUwJUE0JUJFJTIwJUUwJUE0JUFBJUUwJUE0JUJFJUUwJUE0JUEwJTIwJUUwJUE0JTk1JUUwJUE1JThEJUUwJUE0JUFGJUUwJUE0JUJFJTIwJUUwJUE0JTk1JUUwJUE0JUI5JUUwJUE0JUE0JUUwJUE0JUJFJTIwJUUwJUE0JUI5JUUwJUE1JTg4JTNGJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTVEJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTVEJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMjBwYWRkaW5nJTNEVHJ1ZSUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMjB0b2tlbml6ZSUzRFRydWUlMkMlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBZ2VuX3Rva2VucyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCUwQSUyMCUyMCUyMCUyMCoqaW5wdXRzJTJDJTBBJTIwJTIwJTIwJTIwbWF4X25ld190b2tlbnMlM0QzMDAlMkMlMEElMjAlMjAlMjAlMjBkb19zYW1wbGUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwdGVtcGVyYXR1cmUlM0QwLjMlMkMlMEEpJTBBJTBBcHJpbnQocHJvY2Vzc29yLnRva2VuaXplci5kZWNvZGUoZ2VuX3Rva2VucyU1QjAlNUQlNUJpbnB1dHMuaW5wdXRfaWRzLnNoYXBlJTVCMSU1RCUzQSU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-comment"># pip install &#x27;git+https://github.com/huggingface/transformers.git@v4.49.0-Aya Vision&#x27;</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForImageTextToText

model_id = <span class="hljs-string">&quot;CohereLabs/aya-vision-8b&quot;</span>

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map=<span class="hljs-string">&quot;auto&quot;</span>, dtype=torch.float16
)

<span class="hljs-comment"># Format message with the aya-vision chat template</span>
messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
     <span class="hljs-string">&quot;content&quot;</span>: [
       {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://pbs.twimg.com/media/Fx7YvfQWYAIp6rZ?format=jpg&amp;name=medium&quot;</span>},
        {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;चित्र में लिखा पाठ क्या कहता है?&quot;</span>},
    ]},
    ]

inputs = processor.apply_chat_template(
    messages, padding=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
).to(model.device)

gen_tokens = model.generate(
    **inputs,
    max_new_tokens=<span class="hljs-number">300</span>,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.3</span>,
)

<span class="hljs-built_in">print</span>(processor.tokenizer.decode(gen_tokens[<span class="hljs-number">0</span>][inputs.input_ids.shape[<span class="hljs-number">1</span>]:], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){y(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,M){J(t,o,M),u=!0},p:xe,i(o){u||(f(t.$$.fragment,o),u=!0)},o(o){T(t.$$.fragment,o),u=!1},d(o){U(t,o)}}}function Us(w){let t,u,o,M;return t=new is({props:{id:"usage",option:"Pipeline",$$slots:{default:[fs]},$$scope:{ctx:w}}}),o=new is({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ts]},$$scope:{ctx:w}}}),{c(){y(t.$$.fragment),u=a(),y(o.$$.fragment)},l(p){g(t.$$.fragment,p),u=l(p),g(o.$$.fragment,p)},m(p,m){J(t,p,m),i(p,u,m),J(o,p,m),M=!0},p(p,m){const j={};m&2&&(j.$$scope={dirty:m,ctx:p}),t.$set(j);const k={};m&2&&(k.$$scope={dirty:m,ctx:p}),o.$set(k)},i(p){M||(f(t.$$.fragment,p),f(o.$$.fragment,p),M=!0)},o(p){T(t.$$.fragment,p),T(o.$$.fragment,p),M=!1},d(p){p&&n(u),U(t,p),U(o,p)}}}function _s(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=d(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,M){i(o,t,M)},p:xe,d(o){o&&n(t)}}}function ws(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=d(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,M){i(o,t,M)},p:xe,d(o){o&&n(t)}}}function js(w){let t,u="Example:",o,M,p;return M=new Ne({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBBeWFWaXNpb25Gb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRvcmNoX2RldmljZSUyMCUzRCUyMCUyMmN1ZGElM0EwJTIyJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyQ29oZXJlRm9yQUklMkZheWEtdmlzaW9uLThiJTIyJTJDJTIwdXNlX2Zhc3QlM0RUcnVlKSUwQW1vZGVsJTIwJTNEJTIwQXlhVmlzaW9uRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJDb2hlcmVGb3JBSSUyRmF5YS12aXNpb24tOGIlMjIlMkMlMjBkZXZpY2VfbWFwJTNEdG9yY2hfZGV2aWNlKSUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydXJsJTIyJTNBJTIwJTIyaHR0cHMlM0ElMkYlMkZwYnMudHdpbWcuY29tJTJGbWVkaWElMkZGeDdZdmZRV1lBSXA2clolM0Zmb3JtYXQlM0RqcGclMjZuYW1lJTNEbWVkaXVtJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyJUUwJUE0JTlBJUUwJUE0JUJGJUUwJUE0JUE0JUUwJUE1JThEJUUwJUE0JUIwJTIwJUUwJUE0JUFFJUUwJUE1JTg3JUUwJUE0JTgyJTIwJUUwJUE0JUIyJUUwJUE0JUJGJUUwJUE0JTk2JUUwJUE0JUJFJTIwJUUwJUE0JUFBJUUwJUE0JUJFJUUwJUE0JUEwJTIwJUUwJUE0JTk1JUUwJUE1JThEJUUwJUE0JUFGJUUwJUE0JUJFJTIwJUUwJUE0JTk1JUUwJUE0JUI5JUUwJUE0JUE0JUUwJUE0JUJFJTIwJUUwJUE0JUI5JUUwJUE1JTg4JTNGJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTBBJTVEJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMjBwYWRkaW5nJTNEVHJ1ZSUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMjB0b2tlbml6ZSUzRFRydWUlMkMlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwZGV2aWNlJTNEdG9yY2hfZGV2aWNlJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBZ2VuX3Rva2VucyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QzMDAlMkMlMjBkb19zYW1wbGUlM0RUcnVlJTJDJTIwdGVtcGVyYXR1cmUlM0QwLjMpJTBBcHJvY2Vzc29yLnRva2VuaXplci5kZWNvZGUoZ2VuX3Rva2VucyU1QjAlNUQlNUJpbnB1dHMuaW5wdXRfaWRzLnNoYXBlJTVCMSU1RCUzQSU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AyaVisionForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>torch_device = <span class="hljs-string">&quot;cuda:0&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;CohereForAI/aya-vision-8b&quot;</span>, use_fast=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AyaVisionForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;CohereForAI/aya-vision-8b&quot;</span>, device_map=torch_device)

<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [
<span class="hljs-meta">... </span>    {
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;content&quot;</span>: [
<span class="hljs-meta">... </span>            {
<span class="hljs-meta">... </span>                <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>,
<span class="hljs-meta">... </span>                <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://pbs.twimg.com/media/Fx7YvfQWYAIp6rZ?format=jpg&amp;name=medium&quot;</span>,
<span class="hljs-meta">... </span>            },
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;चित्र में लिखा पाठ क्या कहता है?&quot;</span>},
<span class="hljs-meta">... </span>        ],
<span class="hljs-meta">... </span>    }
<span class="hljs-meta">... </span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor.apply_chat_template(
<span class="hljs-meta">... </span>    messages, padding=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, device=torch_device
<span class="hljs-meta">... </span>).to(model.device)

<span class="hljs-meta">&gt;&gt;&gt; </span>gen_tokens = model.generate(**inputs, max_new_tokens=<span class="hljs-number">300</span>, do_sample=<span class="hljs-literal">True</span>, temperature=<span class="hljs-number">0.3</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.tokenizer.decode(gen_tokens[<span class="hljs-number">0</span>][inputs.input_ids.shape[<span class="hljs-number">1</span>]:], skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=a(),y(M.$$.fragment)},l(m){t=d(m,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=u),o=l(m),g(M.$$.fragment,m)},m(m,j){i(m,t,j),i(m,o,j),J(M,m,j),p=!0},p:xe,i(m){p||(f(M.$$.fragment,m),p=!0)},o(m){T(M.$$.fragment,m),p=!1},d(m){m&&(n(t),n(o)),U(M,m)}}}function bs(w){let t,u,o,M,p,m="<em>This model was released on 2025-05-13 and added to Hugging Face Transformers on 2025-03-04.</em>",j,k,Gt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Xe,H,Qe,Y,Nt='<a href="https://huggingface.co/papers/2505.08751" rel="nofollow">Aya Vision</a> is a family of open-weight multimodal vision-language models from Cohere Labs. It is trained with a synthetic annotation framework that generates high-quality multilingual image captions, improving Aya Vision’s generated responses. In addition, a cross-modal model merging technique is used to prevent the model from losing its text capabilities after adding vision capabilities. The model combines a CommandR-7B language model with a SigLIP vision encoder.',We,L,xt='You can find all the original Aya Vision checkpoints under the <a href="https://huggingface.co/collections/CohereLabs/cohere-labs-aya-vision-67c4ccd395ca064308ee1484" rel="nofollow">Aya Vision</a> collection.',Se,z,He,P,Ft='The example below demonstrates how to generate text based on an image with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',Ye,G,Le,D,Xt='Quantization reduces the memory footprint of large models by representing weights at lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for supported backends.',Pe,O,Qt='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to 4-bits.',De,K,Oe,ee,Ke,v,Ue,Wt="<p>Images are represented with the <code>&lt;image&gt;</code> tag in the chat template.</p>",Mt,_e,St='<p>Use the <a href="/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template">apply_chat_template()</a> method to correctly format inputs.</p>',ut,te,we,Ht="The example below demonstrates inference with multiple images.",ht,se,yt,ne,je,Yt="The example below demonstrates inference with batched inputs.",gt,oe,et,ae,tt,R,le,Jt,be,Lt=`Constructs a AyaVision processor which wraps a <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor">AutoImageProcessor</a> and
<code>PretrainedTokenizerFast</code> tokenizer into a single processor that inherits both the image processor and
tokenizer functionalities. See the <code>__call__()</code> and <a href="/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode">decode()</a> for more information.`,st,re,nt,V,ie,ft,Ie,Pt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionForConditionalGeneration">AyaVisionForConditionalGeneration</a>. It is used to instantiate an
AyaVision model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of AyaVision.
e.g. <a href="https://huggingface.co/CohereForAI/aya-vision-8b" rel="nofollow">CohereForAI/aya-vision-8b</a>`,Tt,Ce,Dt=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,ot,ce,at,_,de,Ut,ve,Ot="The AyaVision model which consists of a vision backbone and a language model, without a language modeling head.",_t,Ae,Kt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,wt,ke,es=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,jt,E,pe,bt,Ve,ts='The <a href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionModel">AyaVisionModel</a> forward method, overrides the <code>__call__</code> special method.',It,N,Ct,x,me,vt,Be,ss="Obtains image last hidden states from the vision tower and apply multimodal projection.",At,F,Me,kt,Ee,ns=`Obtains multimodal placeholder mask from <code>input_ids</code> or <code>inputs_embeds</code>, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.`,lt,ue,rt,b,he,Vt,qe,os="The AYA_VISION model which consists of a vision backbone and a language model.",Bt,Ze,as=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Et,Re,ls=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,qt,A,ye,Zt,$e,rs='The <a href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionForConditionalGeneration">AyaVisionForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Rt,X,$t,Q,it,ge,ct,Fe,dt;return H=new Ge({props:{title:"Aya Vision",local:"aya-vision",headingTag:"h1"}}),z=new zt({props:{warning:!1,$$slots:{default:[Js]},$$scope:{ctx:w}}}),G=new gs({props:{id:"usage",options:["Pipeline","AutoModel"],$$slots:{default:[Us]},$$scope:{ctx:w}}}),K=new Ne({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwKCUwQSUyMCUyMCUyMCUyMEF1dG9Qcm9jZXNzb3IlMkMlMEElMjAlMjAlMjAlMjBBdXRvTW9kZWxGb3JJbWFnZVRleHRUb1RleHQlMkMlMEElMjAlMjAlMjAlMjBCaXRzQW5kQnl0ZXNDb25maWclMEEpJTBBJTBBYm5iX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlJTBBKSUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMkNvaGVyZUxhYnMlMkZheWEtdmlzaW9uLTMyYiUyMiUyQyUyMHVzZV9mYXN0JTNEVHJ1ZSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dC5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyQ29oZXJlTGFicyUyRmF5YS12aXNpb24tMzJiJTIyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRGJuYl9jb25maWclMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUwQSklMEElMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlMkMlMjAlMjJ1cmwlMjIlM0ElMjAlMjJodHRwcyUzQSUyRiUyRmh1Z2dpbmdmYWNlLmNvJTJGcm9zY2htaWQlMkZkb2ctcmFjZXMlMkZyZXNvbHZlJTJGbWFpbiUyRmltYWdlcyUyRkJvcmRlcl9Db2xsaWUuanBnJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjAlMjJ0ZXh0JTIyJTNBJTIyRGVzY3JpYmUlMjB3aGF0JTIweW91JTIwc2VlLiUyMiU3RCUwQSUyMCUyMCUyMCUyMCU1RCU3RCUwQSUyMCUyMCUyMCUyMCU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhZGRpbmclM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLnRvKG1vZGVsLmRldmljZSklMEElMEFnZW5lcmF0ZWQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNENTApJTBBcHJpbnQocHJvY2Vzc29yLnRva2VuaXplci5kZWNvZGUoZ2VuZXJhdGVkJTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>
)

processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;CohereLabs/aya-vision-32b&quot;</span>, use_fast=<span class="hljs-literal">True</span>)
model = AutoModelForImageTextToText.from_pretrained(
    <span class="hljs-string">&quot;CohereLabs/aya-vision-32b&quot;</span>,
    quantization_config=bnb_config,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)

inputs = processor.apply_chat_template(
    [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [
        {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://huggingface.co/roschmid/dog-races/resolve/main/images/Border_Collie.jpg&quot;</span>},
        {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>,  <span class="hljs-string">&quot;text&quot;</span>:<span class="hljs-string">&quot;Describe what you see.&quot;</span>}
    ]}
    ],
    padding=<span class="hljs-literal">True</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>,
    tokenize=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=<span class="hljs-number">50</span>)
<span class="hljs-built_in">print</span>(processor.tokenizer.decode(generated[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),ee=new Ge({props:{title:"Notes",local:"notes",headingTag:"h2"}}),se=new Ne({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dCUwQSUyMCUyMCUyMCUyMCUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMkNvaGVyZUZvckFJJTJGYXlhLXZpc2lvbi04YiUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dC5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyQ29oZXJlRm9yQUklMkZheWEtdmlzaW9uLThiJTIyJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMEEpJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ1cmwlMjIlM0ElMjAlMjJodHRwcyUzQSUyRiUyRmNkbi5icml0YW5uaWNhLmNvbSUyRjYxJTJGOTMwNjEtMDUwLTk5MTQ3RENFJTJGU3RhdHVlLW9mLUxpYmVydHktSXNsYW5kLU5ldy1Zb3JrLUJheS5qcGclMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ1cmwlMjIlM0ElMjAlMjJodHRwcyUzQSUyRiUyRnRodW1icy5kcmVhbXN0aW1lLmNvbSUyRmIlMkZnb2xkZW4tZ2F0ZS1icmlkZ2Utc2FuLWZyYW5jaXNjby1wdXJwbGUtZmxvd2Vycy1jYWxpZm9ybmlhLWVjaGl1bS1jYW5kaWNhbnMtMzY4MDU5NDcuanBnJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyVGhlc2UlMjBpbWFnZXMlMjBkZXBpY3QlMjB0d28lMjBkaWZmZXJlbnQlMjBsYW5kbWFya3MuJTIwQ2FuJTIweW91JTIwaWRlbnRpZnklMjB0aGVtJTNGJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTVEJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMjBwYWRkaW5nJTNEVHJ1ZSUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMjB0b2tlbml6ZSUzRFRydWUlMkMlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBZ2VuX3Rva2VucyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCUwQSUyMCUyMCUyMCUyMCoqaW5wdXRzJTJDJTIwJTBBJTIwJTIwJTIwJTIwbWF4X25ld190b2tlbnMlM0QzMDAlMkMlMjAlMEElMjAlMjAlMjAlMjBkb19zYW1wbGUlM0RUcnVlJTJDJTIwJTBBJTIwJTIwJTIwJTIwdGVtcGVyYXR1cmUlM0QwLjMlMkMlMEEpJTBBJTBBZ2VuX3RleHQlMjAlM0QlMjBwcm9jZXNzb3IudG9rZW5pemVyLmRlY29kZShnZW5fdG9rZW5zJTVCMCU1RCU1QmlucHV0cy5pbnB1dF9pZHMuc2hhcGUlNUIxJTVEJTNBJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBcHJpbnQoZ2VuX3RleHQp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForImageTextToText
    
processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;CohereForAI/aya-vision-8b&quot;</span>)
model = AutoModelForImageTextToText.from_pretrained(
    <span class="hljs-string">&quot;CohereForAI/aya-vision-8b&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>, dtype=torch.float16
)

messages = [
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>: [
            {
                <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>,
                <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg&quot;</span>,
            },
            {
                <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>,
                <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg&quot;</span>,
            },
            {
                <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>,
                <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;These images depict two different landmarks. Can you identify them?&quot;</span>,
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages, padding=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
).to(model.device)

gen_tokens = model.generate(
    **inputs, 
    max_new_tokens=<span class="hljs-number">300</span>, 
    do_sample=<span class="hljs-literal">True</span>, 
    temperature=<span class="hljs-number">0.3</span>,
)

gen_text = processor.tokenizer.decode(gen_tokens[<span class="hljs-number">0</span>][inputs.input_ids.shape[<span class="hljs-number">1</span>]:], skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(gen_text)`,wrap:!1}}),oe=new Ne({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dCUwQSUyMCUyMCUyMCUyMCUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9ySW1hZ2VUZXh0VG9UZXh0LmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJDb2hlcmVGb3JBSSUyRmF5YS12aXNpb24tOGIlMjIlMkMlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUwQSklMEElMEFiYXRjaF9tZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUyMCUyMnVybCUyMiUzQSUyMCUyMmh0dHBzJTNBJTJGJTJGbGxhdmEtdmwuZ2l0aHViLmlvJTJGc3RhdGljJTJGaW1hZ2VzJTJGdmlldy5qcGclMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjAlMjJXcml0ZSUyMGElMjBoYWlrdSUyMGZvciUyMHRoaXMlMjBpbWFnZSUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU1RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3RCUyQyUwQSUyMCUyMCUyMCUyMCU1RCUyQyUwQSUyMCUyMCUyMCUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnVybCUyMiUzQSUyMCUyMmh0dHBzJTNBJTJGJTJGY2RuLmJyaXRhbm5pY2EuY29tJTJGNjElMkY5MzA2MS0wNTAtOTkxNDdEQ0UlMkZTdGF0dWUtb2YtTGliZXJ0eS1Jc2xhbmQtTmV3LVlvcmstQmF5LmpwZyUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnVybCUyMiUzQSUyMCUyMmh0dHBzJTNBJTJGJTJGdGh1bWJzLmRyZWFtc3RpbWUuY29tJTJGYiUyRmdvbGRlbi1nYXRlLWJyaWRnZS1zYW4tZnJhbmNpc2NvLXB1cnBsZS1mbG93ZXJzLWNhbGlmb3JuaWEtZWNoaXVtLWNhbmRpY2Fucy0zNjgwNTk0Ny5qcGclMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnRleHQlMjIlM0ElMjAlMjJUaGVzZSUyMGltYWdlcyUyMGRlcGljdCUyMHR3byUyMGRpZmZlcmVudCUyMGxhbmRtYXJrcy4lMjBDYW4lMjB5b3UlMjBpZGVudGlmeSUyMHRoZW0lM0YlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlNUQlMkMlMEElNUQlMEElMEFiYXRjaF9pbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjBiYXRjaF9tZXNzYWdlcyUyQyUyMCUwQSUyMCUyMCUyMCUyMHBhZGRpbmclM0RUcnVlJTJDJTIwJTBBJTIwJTIwJTIwJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUyQyUyMCUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNEVHJ1ZSUyQyUyMCUwQSUyMCUyMCUyMCUyMHJldHVybl9kaWN0JTNEVHJ1ZSUyQyUyMCUwQSUyMCUyMCUyMCUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLnRvKG1vZGVsLmRldmljZSklMEElMEFiYXRjaF9vdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoJTBBJTIwJTIwJTIwJTIwKipiYXRjaF9pbnB1dHMlMkMlMEElMjAlMjAlMjAlMjBtYXhfbmV3X3Rva2VucyUzRDMwMCUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuMyUyQyUwQSklMEElMEFmb3IlMjBpJTJDJTIwb3V0cHV0JTIwaW4lMjBlbnVtZXJhdGUoYmF0Y2hfb3V0cHV0cyklM0ElMEElMjAlMjAlMjAlMjByZXNwb25zZSUyMCUzRCUyMHByb2Nlc3Nvci50b2tlbml6ZXIuZGVjb2RlKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMG91dHB1dCU1QmJhdGNoX2lucHV0cy5pbnB1dF9pZHMuc2hhcGUlNUIxJTVEJTNBJTVEJTJDJTIwJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMEElMjAlMjAlMjAlMjApJTBBJTIwJTIwJTIwJTIwcHJpbnQoZiUyMlJlc3BvbnNlJTIwJTdCaSUyQjElN0QlM0ElNUNuJTdCcmVzcG9uc2UlN0QlNUNuJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForImageTextToText
    
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    <span class="hljs-string">&quot;CohereForAI/aya-vision-8b&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>, dtype=torch.float16
)

batch_messages = [
    [
        {
            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
            <span class="hljs-string">&quot;content&quot;</span>: [
                {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://llava-vl.github.io/static/images/view.jpg&quot;</span>},
                {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;Write a haiku for this image&quot;</span>},
            ],
        },
    ],
    [
        {
            <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
            <span class="hljs-string">&quot;content&quot;</span>: [
                {
                    <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>,
                    <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg&quot;</span>,
                },
                {
                    <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>,
                    <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg&quot;</span>,
                },
                {
                    <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>,
                    <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;These images depict two different landmarks. Can you identify them?&quot;</span>,
                },
            ],
        },
    ],
]

batch_inputs = processor.apply_chat_template(
    batch_messages, 
    padding=<span class="hljs-literal">True</span>, 
    add_generation_prompt=<span class="hljs-literal">True</span>, 
    tokenize=<span class="hljs-literal">True</span>, 
    return_dict=<span class="hljs-literal">True</span>, 
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
).to(model.device)

batch_outputs = model.generate(
    **batch_inputs,
    max_new_tokens=<span class="hljs-number">300</span>,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.3</span>,
)

<span class="hljs-keyword">for</span> i, output <span class="hljs-keyword">in</span> <span class="hljs-built_in">enumerate</span>(batch_outputs):
    response = processor.tokenizer.decode(
        output[batch_inputs.input_ids.shape[<span class="hljs-number">1</span>]:], 
        skip_special_tokens=<span class="hljs-literal">True</span>
    )
    <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Response <span class="hljs-subst">{i+<span class="hljs-number">1</span>}</span>:\\n<span class="hljs-subst">{response}</span>\\n&quot;</span>)`,wrap:!1}}),ae=new Ge({props:{title:"AyaVisionProcessor",local:"transformers.AyaVisionProcessor",headingTag:"h2"}}),le=new S({props:{name:"class transformers.AyaVisionProcessor",anchor:"transformers.AyaVisionProcessor",parameters:[{name:"image_processor",val:" = None"},{name:"tokenizer",val:" = None"},{name:"patch_size",val:": int = 28"},{name:"img_size",val:": int = 364"},{name:"image_token",val:" = '<image>'"},{name:"downsample_factor",val:": int = 1"},{name:"start_of_img_token",val:" = '<|START_OF_IMG|>'"},{name:"end_of_img_token",val:" = '<|END_OF_IMG|>'"},{name:"img_patch_token",val:" = '<|IMG_PATCH|>'"},{name:"img_line_break_token",val:" = '<|IMG_LINE_BREAK|>'"},{name:"tile_token",val:" = 'TILE'"},{name:"tile_global_token",val:" = 'TILE_GLOBAL'"},{name:"chat_template",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.AyaVisionProcessor.image_processor",description:`<strong>image_processor</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor">AutoImageProcessor</a>, <em>optional</em>) &#x2014;
The image processor is a required input.`,name:"image_processor"},{anchor:"transformers.AyaVisionProcessor.tokenizer",description:`<strong>tokenizer</strong> ([<code>PreTrainedTokenizer</code>, <code>PreTrainedTokenizerFast</code>], <em>optional</em>) &#x2014;
The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.AyaVisionProcessor.patch_size",description:`<strong>patch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 28) &#x2014;
The size of image patches for tokenization.`,name:"patch_size"},{anchor:"transformers.AyaVisionProcessor.img_size",description:`<strong>img_size</strong> (<code>int</code>, <em>optional</em>, defaults to 364) &#x2014;
The size of the image to be tokenized. This should correspond to the size given to the image processor.`,name:"img_size"},{anchor:"transformers.AyaVisionProcessor.image_token",description:`<strong>image_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;image&gt;&quot;</code>) &#x2014;
The token to be used to represent an image in the text.`,name:"image_token"},{anchor:"transformers.AyaVisionProcessor.downsample_factor",description:`<strong>downsample_factor</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The factor by which to scale the patch size.`,name:"downsample_factor"},{anchor:"transformers.AyaVisionProcessor.start_of_img_token",description:`<strong>start_of_img_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|START_OF_IMG|&gt;&quot;</code>) &#x2014;
The token to be used to represent the start of an image in the text.`,name:"start_of_img_token"},{anchor:"transformers.AyaVisionProcessor.end_of_img_token",description:`<strong>end_of_img_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|END_OF_IMG|&gt;&quot;</code>) &#x2014;
The token to be used to represent the end of an image in the text.`,name:"end_of_img_token"},{anchor:"transformers.AyaVisionProcessor.img_patch_token",description:`<strong>img_patch_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|IMG_PATCH|&gt;&quot;</code>) &#x2014;
The token to be used to represent an image patch in the text.`,name:"img_patch_token"},{anchor:"transformers.AyaVisionProcessor.img_line_break_token",description:`<strong>img_line_break_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|IMG_LINE_BREAK|&gt;&quot;</code>) &#x2014;
The token to be used to represent a line break in the text.`,name:"img_line_break_token"},{anchor:"transformers.AyaVisionProcessor.tile_token",description:`<strong>tile_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;TILE&quot;</code>) &#x2014;
The token to be used to represent an image patch in the text.`,name:"tile_token"},{anchor:"transformers.AyaVisionProcessor.tile_global_token",description:`<strong>tile_global_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;TILE_GLOBAL&quot;</code>) &#x2014;
The token to be used to represent the cover image in the text.`,name:"tile_global_token"},{anchor:"transformers.AyaVisionProcessor.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014; A Jinja template which will be used to convert lists of messages
in a chat into a tokenizable string.`,name:"chat_template"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/processing_aya_vision.py#L46"}}),re=new Ge({props:{title:"AyaVisionConfig",local:"transformers.AyaVisionConfig",headingTag:"h2"}}),ie=new S({props:{name:"class transformers.AyaVisionConfig",anchor:"transformers.AyaVisionConfig",parameters:[{name:"vision_config",val:" = None"},{name:"text_config",val:" = None"},{name:"vision_feature_select_strategy",val:" = 'full'"},{name:"vision_feature_layer",val:" = -1"},{name:"downsample_factor",val:" = 2"},{name:"adapter_layer_norm_eps",val:" = 1e-06"},{name:"image_token_index",val:" = 255036"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.AyaVisionConfig.vision_config",description:`<strong>vision_config</strong> (<code>Union[AutoConfig, dict]</code>,  <em>optional</em>, defaults to <code>SiglipVisionConfig</code>) &#x2014;
The config object or dictionary of the vision backbone.`,name:"vision_config"},{anchor:"transformers.AyaVisionConfig.text_config",description:`<strong>text_config</strong> (<code>Union[AutoConfig, dict]</code>, <em>optional</em>, defaults to <code>Cohere2Config</code>) &#x2014;
The config object or dictionary of the text backbone.`,name:"text_config"},{anchor:"transformers.AyaVisionConfig.vision_feature_select_strategy",description:`<strong>vision_feature_select_strategy</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;full&quot;</code>) &#x2014;
The feature selection strategy used to select the vision feature from the vision backbone.
Can be one of <code>&quot;default&quot;</code> or <code>&quot;full&quot;</code>. If <code>&quot;default&quot;</code>, the CLS token is removed from the vision features.
If <code>&quot;full&quot;</code>, the full vision features are used.`,name:"vision_feature_select_strategy"},{anchor:"transformers.AyaVisionConfig.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
The index of the layer to select the vision feature.`,name:"vision_feature_layer"},{anchor:"transformers.AyaVisionConfig.downsample_factor",description:`<strong>downsample_factor</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The downsample factor to apply to the vision features.`,name:"downsample_factor"},{anchor:"transformers.AyaVisionConfig.adapter_layer_norm_eps",description:`<strong>adapter_layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon value used for layer normalization in the adapter.`,name:"adapter_layer_norm_eps"},{anchor:"transformers.AyaVisionConfig.image_token_index",description:`<strong>image_token_index</strong> (<code>int</code>, <em>optional</em>, defaults to 255036) &#x2014;
The image token index to encode the image prompt.`,name:"image_token_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/configuration_aya_vision.py#L25"}}),ce=new Ge({props:{title:"AyaVisionModel",local:"transformers.AyaVisionModel",headingTag:"h2"}}),de=new S({props:{name:"class transformers.AyaVisionModel",anchor:"transformers.AyaVisionModel",parameters:[{name:"config",val:": AyaVisionConfig"}],parametersDescription:[{anchor:"transformers.AyaVisionModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig">AyaVisionConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L167"}}),pe=new S({props:{name:"forward",anchor:"transformers.AyaVisionModel.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"pixel_values",val:": FloatTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"vision_feature_layer",val:": typing.Union[int, list[int], NoneType] = None"},{name:"vision_feature_select_strategy",val:": typing.Optional[str] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}],parametersDescription:[{anchor:"transformers.AyaVisionModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.AyaVisionModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<code>image_processor_class</code>. See <code>image_processor_class.__call__</code> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionProcessor">AyaVisionProcessor</a> uses
<code>image_processor_class</code> for processing images).`,name:"pixel_values"},{anchor:"transformers.AyaVisionModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.AyaVisionModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.AyaVisionModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.AyaVisionModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.AyaVisionModel.forward.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>Union[int, list[int], NoneType]</code>) &#x2014;
The index of the layer to select the vision feature. If multiple indices are provided,
the vision feature of the corresponding indices will be concatenated to form the
vision features.`,name:"vision_feature_layer"},{anchor:"transformers.AyaVisionModel.forward.vision_feature_select_strategy",description:`<strong>vision_feature_select_strategy</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The feature selection strategy used to select the vision feature from the vision backbone.
Can be one of <code>&quot;default&quot;</code> or <code>&quot;full&quot;</code>.`,name:"vision_feature_select_strategy"},{anchor:"transformers.AyaVisionModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.AyaVisionModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L269",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.aya_vision.modeling_aya_vision.AyaVisionModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig"
>AyaVisionConfig</a>) and inputs.</p>
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


<p><code>transformers.models.aya_vision.modeling_aya_vision.AyaVisionModelOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),N=new zt({props:{$$slots:{default:[_s]},$$scope:{ctx:w}}}),me=new S({props:{name:"get_image_features",anchor:"transformers.AyaVisionModel.get_image_features",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"vision_feature_layer",val:": typing.Union[int, list[int], NoneType] = None"},{name:"vision_feature_select_strategy",val:": typing.Optional[str] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.AyaVisionModel.get_image_features.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor]</code> of shape <code>(batch_size, channels, height, width)</code>) &#x2014;
The tensors corresponding to the input images.`,name:"pixel_values"},{anchor:"transformers.AyaVisionModel.get_image_features.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>Union[int, list[int]]</code>, <em>optional</em>) &#x2014;
The index of the layer to select the vision feature. If multiple indices are provided,
the vision feature of the corresponding indices will be concatenated to form the
vision features.`,name:"vision_feature_layer"},{anchor:"transformers.AyaVisionModel.get_image_features.vision_feature_select_strategy",description:`<strong>vision_feature_select_strategy</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The feature selection strategy used to select the vision feature from the vision backbone.
Can be one of <code>&quot;default&quot;</code> or <code>&quot;full&quot;</code>`,name:"vision_feature_select_strategy"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L190",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Image feature tensor of shape <code>(num_images, image_length, embed_dim)</code>).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>image_features (<code>torch.Tensor</code>)</p>
`}}),Me=new S({props:{name:"get_placeholder_mask",anchor:"transformers.AyaVisionModel.get_placeholder_mask",parameters:[{name:"input_ids",val:": LongTensor"},{name:"inputs_embeds",val:": FloatTensor"},{name:"image_features",val:": FloatTensor"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L245"}}),ue=new Ge({props:{title:"AyaVisionForConditionalGeneration",local:"transformers.AyaVisionForConditionalGeneration",headingTag:"h2"}}),he=new S({props:{name:"class transformers.AyaVisionForConditionalGeneration",anchor:"transformers.AyaVisionForConditionalGeneration",parameters:[{name:"config",val:": AyaVisionConfig"}],parametersDescription:[{anchor:"transformers.AyaVisionForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig">AyaVisionConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L336"}}),ye=new S({props:{name:"forward",anchor:"transformers.AyaVisionForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"vision_feature_layer",val:": typing.Union[int, list[int], NoneType] = None"},{name:"vision_feature_select_strategy",val:": typing.Optional[str] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"image_sizes",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.AyaVisionForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<code>image_processor_class</code>. See <code>image_processor_class.__call__</code> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionProcessor">AyaVisionProcessor</a> uses
<code>image_processor_class</code> for processing images).`,name:"pixel_values"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>Union[int, list[int], NoneType]</code>) &#x2014;
The index of the layer to select the vision feature. If multiple indices are provided,
the vision feature of the corresponding indices will be concatenated to form the
vision features.`,name:"vision_feature_layer"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.vision_feature_select_strategy",description:`<strong>vision_feature_select_strategy</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The feature selection strategy used to select the vision feature from the vision backbone.
Can be one of <code>&quot;default&quot;</code> or <code>&quot;full&quot;</code>.`,name:"vision_feature_select_strategy"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"},{anchor:"transformers.AyaVisionForConditionalGeneration.forward.image_sizes",description:`<strong>image_sizes</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, 2)</code>, <em>optional</em>) &#x2014;
The sizes of the images in the batch, being (height, width) for each image.`,name:"image_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aya_vision/modeling_aya_vision.py#L393",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.aya_vision.modeling_aya_vision.AyaVisionCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/aya_vision#transformers.AyaVisionConfig"
>AyaVisionConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
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
image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.aya_vision.modeling_aya_vision.AyaVisionCausalLMOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),X=new zt({props:{$$slots:{default:[ws]},$$scope:{ctx:w}}}),Q=new hs({props:{anchor:"transformers.AyaVisionForConditionalGeneration.forward.example",$$slots:{default:[js]},$$scope:{ctx:w}}}),ge=new ys({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/aya_vision.md"}}),{c(){t=c("meta"),u=a(),o=c("p"),M=a(),p=c("p"),p.innerHTML=m,j=a(),k=c("div"),k.innerHTML=Gt,Xe=a(),y(H.$$.fragment),Qe=a(),Y=c("p"),Y.innerHTML=Nt,We=a(),L=c("p"),L.innerHTML=xt,Se=a(),y(z.$$.fragment),He=a(),P=c("p"),P.innerHTML=Ft,Ye=a(),y(G.$$.fragment),Le=a(),D=c("p"),D.innerHTML=Xt,Pe=a(),O=c("p"),O.innerHTML=Qt,De=a(),y(K.$$.fragment),Oe=a(),y(ee.$$.fragment),Ke=a(),v=c("ul"),Ue=c("li"),Ue.innerHTML=Wt,Mt=a(),_e=c("li"),_e.innerHTML=St,ut=a(),te=c("li"),we=c("p"),we.textContent=Ht,ht=a(),y(se.$$.fragment),yt=a(),ne=c("li"),je=c("p"),je.textContent=Yt,gt=a(),y(oe.$$.fragment),et=a(),y(ae.$$.fragment),tt=a(),R=c("div"),y(le.$$.fragment),Jt=a(),be=c("p"),be.innerHTML=Lt,st=a(),y(re.$$.fragment),nt=a(),V=c("div"),y(ie.$$.fragment),ft=a(),Ie=c("p"),Ie.innerHTML=Pt,Tt=a(),Ce=c("p"),Ce.innerHTML=Dt,ot=a(),y(ce.$$.fragment),at=a(),_=c("div"),y(de.$$.fragment),Ut=a(),ve=c("p"),ve.textContent=Ot,_t=a(),Ae=c("p"),Ae.innerHTML=Kt,wt=a(),ke=c("p"),ke.innerHTML=es,jt=a(),E=c("div"),y(pe.$$.fragment),bt=a(),Ve=c("p"),Ve.innerHTML=ts,It=a(),y(N.$$.fragment),Ct=a(),x=c("div"),y(me.$$.fragment),vt=a(),Be=c("p"),Be.textContent=ss,At=a(),F=c("div"),y(Me.$$.fragment),kt=a(),Ee=c("p"),Ee.innerHTML=ns,lt=a(),y(ue.$$.fragment),rt=a(),b=c("div"),y(he.$$.fragment),Vt=a(),qe=c("p"),qe.textContent=os,Bt=a(),Ze=c("p"),Ze.innerHTML=as,Et=a(),Re=c("p"),Re.innerHTML=ls,qt=a(),A=c("div"),y(ye.$$.fragment),Zt=a(),$e=c("p"),$e.innerHTML=rs,Rt=a(),y(X.$$.fragment),$t=a(),y(Q.$$.fragment),it=a(),y(ge.$$.fragment),ct=a(),Fe=c("p"),this.h()},l(e){const s=Ms("svelte-u9bgzb",document.head);t=d(s,"META",{name:!0,content:!0}),s.forEach(n),u=l(e),o=d(e,"P",{}),C(o).forEach(n),M=l(e),p=d(e,"P",{"data-svelte-h":!0}),h(p)!=="svelte-2p1ouc"&&(p.innerHTML=m),j=l(e),k=d(e,"DIV",{style:!0,"data-svelte-h":!0}),h(k)!=="svelte-wa5t4p"&&(k.innerHTML=Gt),Xe=l(e),g(H.$$.fragment,e),Qe=l(e),Y=d(e,"P",{"data-svelte-h":!0}),h(Y)!=="svelte-6jf6qx"&&(Y.innerHTML=Nt),We=l(e),L=d(e,"P",{"data-svelte-h":!0}),h(L)!=="svelte-zf46cv"&&(L.innerHTML=xt),Se=l(e),g(z.$$.fragment,e),He=l(e),P=d(e,"P",{"data-svelte-h":!0}),h(P)!=="svelte-2n7mbe"&&(P.innerHTML=Ft),Ye=l(e),g(G.$$.fragment,e),Le=l(e),D=d(e,"P",{"data-svelte-h":!0}),h(D)!=="svelte-14ckvxi"&&(D.innerHTML=Xt),Pe=l(e),O=d(e,"P",{"data-svelte-h":!0}),h(O)!=="svelte-60nsd0"&&(O.innerHTML=Qt),De=l(e),g(K.$$.fragment,e),Oe=l(e),g(ee.$$.fragment,e),Ke=l(e),v=d(e,"UL",{});var B=C(v);Ue=d(B,"LI",{"data-svelte-h":!0}),h(Ue)!=="svelte-vurmho"&&(Ue.innerHTML=Wt),Mt=l(B),_e=d(B,"LI",{"data-svelte-h":!0}),h(_e)!=="svelte-1modl3q"&&(_e.innerHTML=St),ut=l(B),te=d(B,"LI",{});var Je=C(te);we=d(Je,"P",{"data-svelte-h":!0}),h(we)!=="svelte-1nt84ok"&&(we.textContent=Ht),ht=l(Je),g(se.$$.fragment,Je),Je.forEach(n),yt=l(B),ne=d(B,"LI",{});var fe=C(ne);je=d(fe,"P",{"data-svelte-h":!0}),h(je)!=="svelte-grg408"&&(je.textContent=Yt),gt=l(fe),g(oe.$$.fragment,fe),fe.forEach(n),B.forEach(n),et=l(e),g(ae.$$.fragment,e),tt=l(e),R=d(e,"DIV",{class:!0});var Te=C(R);g(le.$$.fragment,Te),Jt=l(Te),be=d(Te,"P",{"data-svelte-h":!0}),h(be)!=="svelte-zo5vg0"&&(be.innerHTML=Lt),Te.forEach(n),st=l(e),g(re.$$.fragment,e),nt=l(e),V=d(e,"DIV",{class:!0});var $=C(V);g(ie.$$.fragment,$),ft=l($),Ie=d($,"P",{"data-svelte-h":!0}),h(Ie)!=="svelte-xosbtp"&&(Ie.innerHTML=Pt),Tt=l($),Ce=d($,"P",{"data-svelte-h":!0}),h(Ce)!=="svelte-1ek1ss9"&&(Ce.innerHTML=Dt),$.forEach(n),ot=l(e),g(ce.$$.fragment,e),at=l(e),_=d(e,"DIV",{class:!0});var I=C(_);g(de.$$.fragment,I),Ut=l(I),ve=d(I,"P",{"data-svelte-h":!0}),h(ve)!=="svelte-2tuuv0"&&(ve.textContent=Ot),_t=l(I),Ae=d(I,"P",{"data-svelte-h":!0}),h(Ae)!=="svelte-q52n56"&&(Ae.innerHTML=Kt),wt=l(I),ke=d(I,"P",{"data-svelte-h":!0}),h(ke)!=="svelte-hswkmf"&&(ke.innerHTML=es),jt=l(I),E=d(I,"DIV",{class:!0});var ze=C(E);g(pe.$$.fragment,ze),bt=l(ze),Ve=d(ze,"P",{"data-svelte-h":!0}),h(Ve)!=="svelte-10bx3yp"&&(Ve.innerHTML=ts),It=l(ze),g(N.$$.fragment,ze),ze.forEach(n),Ct=l(I),x=d(I,"DIV",{class:!0});var pt=C(x);g(me.$$.fragment,pt),vt=l(pt),Be=d(pt,"P",{"data-svelte-h":!0}),h(Be)!=="svelte-1vzo9k5"&&(Be.textContent=ss),pt.forEach(n),At=l(I),F=d(I,"DIV",{class:!0});var mt=C(F);g(Me.$$.fragment,mt),kt=l(mt),Ee=d(mt,"P",{"data-svelte-h":!0}),h(Ee)!=="svelte-3ue1dv"&&(Ee.innerHTML=ns),mt.forEach(n),I.forEach(n),lt=l(e),g(ue.$$.fragment,e),rt=l(e),b=d(e,"DIV",{class:!0});var q=C(b);g(he.$$.fragment,q),Vt=l(q),qe=d(q,"P",{"data-svelte-h":!0}),h(qe)!=="svelte-bpugxb"&&(qe.textContent=os),Bt=l(q),Ze=d(q,"P",{"data-svelte-h":!0}),h(Ze)!=="svelte-q52n56"&&(Ze.innerHTML=as),Et=l(q),Re=d(q,"P",{"data-svelte-h":!0}),h(Re)!=="svelte-hswkmf"&&(Re.innerHTML=ls),qt=l(q),A=d(q,"DIV",{class:!0});var W=C(A);g(ye.$$.fragment,W),Zt=l(W),$e=d(W,"P",{"data-svelte-h":!0}),h($e)!=="svelte-idqqwp"&&($e.innerHTML=rs),Rt=l(W),g(X.$$.fragment,W),$t=l(W),g(Q.$$.fragment,W),W.forEach(n),q.forEach(n),it=l(e),g(ge.$$.fragment,e),ct=l(e),Fe=d(e,"P",{}),C(Fe).forEach(n),this.h()},h(){Z(t,"name","hf:doc:metadata"),Z(t,"content",Is),us(k,"float","right"),Z(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(_,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(b,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){r(document.head,t),i(e,u,s),i(e,o,s),i(e,M,s),i(e,p,s),i(e,j,s),i(e,k,s),i(e,Xe,s),J(H,e,s),i(e,Qe,s),i(e,Y,s),i(e,We,s),i(e,L,s),i(e,Se,s),J(z,e,s),i(e,He,s),i(e,P,s),i(e,Ye,s),J(G,e,s),i(e,Le,s),i(e,D,s),i(e,Pe,s),i(e,O,s),i(e,De,s),J(K,e,s),i(e,Oe,s),J(ee,e,s),i(e,Ke,s),i(e,v,s),r(v,Ue),r(v,Mt),r(v,_e),r(v,ut),r(v,te),r(te,we),r(te,ht),J(se,te,null),r(v,yt),r(v,ne),r(ne,je),r(ne,gt),J(oe,ne,null),i(e,et,s),J(ae,e,s),i(e,tt,s),i(e,R,s),J(le,R,null),r(R,Jt),r(R,be),i(e,st,s),J(re,e,s),i(e,nt,s),i(e,V,s),J(ie,V,null),r(V,ft),r(V,Ie),r(V,Tt),r(V,Ce),i(e,ot,s),J(ce,e,s),i(e,at,s),i(e,_,s),J(de,_,null),r(_,Ut),r(_,ve),r(_,_t),r(_,Ae),r(_,wt),r(_,ke),r(_,jt),r(_,E),J(pe,E,null),r(E,bt),r(E,Ve),r(E,It),J(N,E,null),r(_,Ct),r(_,x),J(me,x,null),r(x,vt),r(x,Be),r(_,At),r(_,F),J(Me,F,null),r(F,kt),r(F,Ee),i(e,lt,s),J(ue,e,s),i(e,rt,s),i(e,b,s),J(he,b,null),r(b,Vt),r(b,qe),r(b,Bt),r(b,Ze),r(b,Et),r(b,Re),r(b,qt),r(b,A),J(ye,A,null),r(A,Zt),r(A,$e),r(A,Rt),J(X,A,null),r(A,$t),J(Q,A,null),i(e,it,s),J(ge,e,s),i(e,ct,s),i(e,Fe,s),dt=!0},p(e,[s]){const B={};s&2&&(B.$$scope={dirty:s,ctx:e}),z.$set(B);const Je={};s&2&&(Je.$$scope={dirty:s,ctx:e}),G.$set(Je);const fe={};s&2&&(fe.$$scope={dirty:s,ctx:e}),N.$set(fe);const Te={};s&2&&(Te.$$scope={dirty:s,ctx:e}),X.$set(Te);const $={};s&2&&($.$$scope={dirty:s,ctx:e}),Q.$set($)},i(e){dt||(f(H.$$.fragment,e),f(z.$$.fragment,e),f(G.$$.fragment,e),f(K.$$.fragment,e),f(ee.$$.fragment,e),f(se.$$.fragment,e),f(oe.$$.fragment,e),f(ae.$$.fragment,e),f(le.$$.fragment,e),f(re.$$.fragment,e),f(ie.$$.fragment,e),f(ce.$$.fragment,e),f(de.$$.fragment,e),f(pe.$$.fragment,e),f(N.$$.fragment,e),f(me.$$.fragment,e),f(Me.$$.fragment,e),f(ue.$$.fragment,e),f(he.$$.fragment,e),f(ye.$$.fragment,e),f(X.$$.fragment,e),f(Q.$$.fragment,e),f(ge.$$.fragment,e),dt=!0)},o(e){T(H.$$.fragment,e),T(z.$$.fragment,e),T(G.$$.fragment,e),T(K.$$.fragment,e),T(ee.$$.fragment,e),T(se.$$.fragment,e),T(oe.$$.fragment,e),T(ae.$$.fragment,e),T(le.$$.fragment,e),T(re.$$.fragment,e),T(ie.$$.fragment,e),T(ce.$$.fragment,e),T(de.$$.fragment,e),T(pe.$$.fragment,e),T(N.$$.fragment,e),T(me.$$.fragment,e),T(Me.$$.fragment,e),T(ue.$$.fragment,e),T(he.$$.fragment,e),T(ye.$$.fragment,e),T(X.$$.fragment,e),T(Q.$$.fragment,e),T(ge.$$.fragment,e),dt=!1},d(e){e&&(n(u),n(o),n(M),n(p),n(j),n(k),n(Xe),n(Qe),n(Y),n(We),n(L),n(Se),n(He),n(P),n(Ye),n(Le),n(D),n(Pe),n(O),n(De),n(Oe),n(Ke),n(v),n(et),n(tt),n(R),n(st),n(nt),n(V),n(ot),n(at),n(_),n(lt),n(rt),n(b),n(it),n(ct),n(Fe)),n(t),U(H,e),U(z,e),U(G,e),U(K,e),U(ee,e),U(se),U(oe),U(ae,e),U(le),U(re,e),U(ie),U(ce,e),U(de),U(pe),U(N),U(me),U(Me),U(ue,e),U(he),U(ye),U(X),U(Q),U(ge,e)}}}const Is='{"title":"Aya Vision","local":"aya-vision","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"AyaVisionProcessor","local":"transformers.AyaVisionProcessor","sections":[],"depth":2},{"title":"AyaVisionConfig","local":"transformers.AyaVisionConfig","sections":[],"depth":2},{"title":"AyaVisionModel","local":"transformers.AyaVisionModel","sections":[],"depth":2},{"title":"AyaVisionForConditionalGeneration","local":"transformers.AyaVisionForConditionalGeneration","sections":[],"depth":2}],"depth":1}';function Cs(w){return ds(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Rs extends ps{constructor(t){super(),ms(this,t,Cs,bs,cs,{})}}export{Rs as component};
