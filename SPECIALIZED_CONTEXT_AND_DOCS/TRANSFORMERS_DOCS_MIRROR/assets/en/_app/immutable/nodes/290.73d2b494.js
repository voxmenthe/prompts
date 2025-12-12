import{s as qs,o as Qs,n as E}from"../chunks/scheduler.18a86fab.js";import{S as Rs,i as Fs,g as a,s as o,r as f,A as Es,h as r,f as i,c as n,j as v,x as m,u as g,k,l as Hs,y as t,a as u,v as M,d as _,t as T,w as y}from"../chunks/index.98837b22.js";import{T as zo}from"../chunks/Tip.77304350.js";import{D as C}from"../chunks/Docstring.a1ef7999.js";import{C as F}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as En}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Yt,E as As}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Ss,a as Gs}from"../chunks/HfOption.6641485e.js";function Xs(U){let s,J=`This model was contributed by <a href="https://huggingface.co/cyrilvallez" rel="nofollow">cyrilvallez</a> and <a href="https://huggingface.co/yonigozlan" rel="nofollow">yonigozlan</a>.
Click on the Mistral3 models in the right sidebar for more examples of how to apply Mistral3 to different tasks.`;return{c(){s=a("p"),s.innerHTML=J},l(d){s=r(d,"P",{"data-svelte-h":!0}),m(s)!=="svelte-vd0pcm"&&(s.innerHTML=J)},m(d,w){u(d,s,w)},p:E,d(d){d&&i(s)}}}function Ds(U){let s,J;return s=new F({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyY29udGVudCUyMiUzQSU1QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmltYWdlJTIyJTNBJTIwJTIyaHR0cHMlM0ElMkYlMkZodWdnaW5nZmFjZS5jbyUyRmRhdGFzZXRzJTJGaHVnZ2luZ2ZhY2UlMkZkb2N1bWVudGF0aW9uLWltYWdlcyUyRnJlc29sdmUlMkZtYWluJTJGYmVlLmpwZyUyMiUyQyU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMkRlc2NyaWJlJTIwdGhpcyUyMGltYWdlLiUyMiU3RCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyQyU1RCUwQSUyMCUyMCUyMCUyMCUyQyU3RCUwQSUyQyU1RCUwQSUwQXBpcGVsaW5lJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMmltYWdlLXRleHQtdG8tdGV4dCUyMiUyQyUyMCUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIybWlzdHJhbGFpJTJGTWlzdHJhbC1TbWFsbC0zLjEtMjRCLUluc3RydWN0LTI1MDMlMjIlMkMlMjAlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUwQSklMEFvdXRwdXRzJTIwJTNEJTIwcGlwZWxpbmUodGV4dCUzRG1lc3NhZ2VzJTJDJTIwbWF4X25ld190b2tlbnMlM0Q1MCUyQyUyMHJldHVybl9mdWxsX3RleHQlM0RGYWxzZSklMEElMEFvdXRwdXRzJTVCMCU1RCU1QiUyMmdlbmVyYXRlZF90ZXh0JTIyJTVEJTBBJ1RoZSUyMGltYWdlJTIwZGVwaWN0cyUyMGElMjB2aWJyYW50JTIwYW5kJTIwbHVzaCUyMGdhcmRlbiUyMHNjZW5lJTIwZmVhdHVyaW5nJTIwYSUyMHZhcmlldHklMjBvZiUyMHdpbGRmbG93ZXJzJTIwYW5kJTIwcGxhbnRzLiUyMFRoZSUyMGNlbnRyYWwlMjBmb2N1cyUyMGlzJTIwb24lMjBhJTIwbGFyZ2UlMkMlMjBwaW5raXNoLXB1cnBsZSUyMGZsb3dlciUyQyUyMGxpa2VseSUyMGElMjBHcmVhdGVyJTIwQ2VsYW5kaW5lJTIwKENoZWxpZG9uaXVtJTIwbWFqdXMpJTJDJTIwd2l0aCUyMGEn",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>:[
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>,
            <span class="hljs-string">&quot;image&quot;</span>: <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg&quot;</span>,},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;Describe this image.&quot;</span>}
        ,]
    ,}
,]

pipeline = pipeline(
    task=<span class="hljs-string">&quot;image-text-to-text&quot;</span>, 
    model=<span class="hljs-string">&quot;mistralai/Mistral-Small-3.1-24B-Instruct-2503&quot;</span>, 
    dtype=torch.bfloat16,
    device=<span class="hljs-number">0</span>
)
outputs = pipeline(text=messages, max_new_tokens=<span class="hljs-number">50</span>, return_full_text=<span class="hljs-literal">False</span>)

outputs[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;generated_text&quot;</span>]
<span class="hljs-string">&#x27;The image depicts a vibrant and lush garden scene featuring a variety of wildflowers and plants. The central focus is on a large, pinkish-purple flower, likely a Greater Celandine (Chelidonium majus), with a&#x27;</span>`,wrap:!1}}),{c(){f(s.$$.fragment)},l(d){g(s.$$.fragment,d)},m(d,w){M(s,d,w),J=!0},p:E,i(d){J||(_(s.$$.fragment,d),J=!0)},o(d){T(s.$$.fragment,d),J=!1},d(d){y(s,d)}}}function Ys(U){let s,J;return s=new F({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dCUyQyUyMGluZmVyX2RldmljZSUyMCUwQSUwQXRvcmNoX2RldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWxfY2hlY2twb2ludCUyMCUzRCUyMCUyMm1pc3RyYWxhaSUyRk1pc3RyYWwtU21hbGwtMy4xLTI0Qi1JbnN0cnVjdC0yNTAzJTIyJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQobW9kZWxfY2hlY2twb2ludCklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dC5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwbW9kZWxfY2hlY2twb2ludCUyQyUyMCUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0R0b3JjaF9kZXZpY2UlMkMlMjAlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTBBKSUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJjb250ZW50JTIyJTNBJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyaW1hZ2UlMjIlM0ElMjAlMjJodHRwcyUzQSUyRiUyRmh1Z2dpbmdmYWNlLmNvJTJGZGF0YXNldHMlMkZodWdnaW5nZmFjZSUyRmRvY3VtZW50YXRpb24taW1hZ2VzJTJGcmVzb2x2ZSUyRm1haW4lMkZiZWUuanBnJTIyJTJDJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyRGVzY3JpYmUlMjB0aGlzJTIwaW1hZ2UuJTIyJTdEJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTJDJTVEJTBBJTIwJTIwJTIwJTIwJTJDJTdEJTBBJTJDJTVEJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMjAlMEElMjAlMjAlMjAlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTJDJTIwJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTIwcmV0dXJuX2RpY3QlM0RUcnVlJTJDJTIwJTBBJTIwJTIwJTIwJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlJTJDJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiklMEElMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMjApJTBBZGVjb2RlZF9vdXRwdXQlMjAlM0QlMjBwcm9jZXNzb3IuZGVjb2RlKGdlbmVyYXRlX2lkcyU1QjAlMkMlMjBpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQuc2hhcGUlNUIxJTVEJTIwJTNBJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBZGVjb2RlZF9vdXRwdXQlMEEnVGhlJTIwaW1hZ2UlMjBkZXBpY3RzJTIwYSUyMHZpYnJhbnQlMjBhbmQlMjBsdXNoJTIwZ2FyZGVuJTIwc2NlbmUlMjBmZWF0dXJpbmclMjBhJTIwdmFyaWV0eSUyMG9mJTIwd2lsZGZsb3dlcnMlMjBhbmQlMjBwbGFudHMuJTIwVGhlJTIwY2VudHJhbCUyMGZvY3VzJTIwaXMlMjBvbiUyMGElMjBsYXJnZSUyQyUyMHBpbmtpc2gtcHVycGxlJTIwZmxvd2VyJTJDJTIwbGlrZWx5JTIwYSUyMEdyZWF0ZXIlMjBDZWxhbmRpbmUlMjAoQ2hlbGlkb25pdW0lMjBtYWp1cyklMkMlMjB3aXRoJTIwYSc=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForImageTextToText, infer_device 

torch_device = infer_device()
model_checkpoint = <span class="hljs-string">&quot;mistralai/Mistral-Small-3.1-24B-Instruct-2503&quot;</span>
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(
    model_checkpoint, 
    device_map=torch_device, 
    dtype=torch.bfloat16
)

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
        <span class="hljs-string">&quot;content&quot;</span>:[
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>,
            <span class="hljs-string">&quot;image&quot;</span>: <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg&quot;</span>,},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;Describe this image.&quot;</span>}
        ,]
    ,}
,]

inputs = processor.apply_chat_template(
    messages, 
    add_generation_prompt=<span class="hljs-literal">True</span>, 
    tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, 
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=<span class="hljs-number">20</span>)
decoded_output = processor.decode(generate_ids[<span class="hljs-number">0</span>, inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape[<span class="hljs-number">1</span>] :], skip_special_tokens=<span class="hljs-literal">True</span>)

decoded_output
<span class="hljs-string">&#x27;The image depicts a vibrant and lush garden scene featuring a variety of wildflowers and plants. The central focus is on a large, pinkish-purple flower, likely a Greater Celandine (Chelidonium majus), with a&#x27;</span>`,wrap:!1}}),{c(){f(s.$$.fragment)},l(d){g(s.$$.fragment,d)},m(d,w){M(s,d,w),J=!0},p:E,i(d){J||(_(s.$$.fragment,d),J=!0)},o(d){T(s.$$.fragment,d),J=!1},d(d){y(s,d)}}}function Ls(U){let s,J,d,w;return s=new Gs({props:{id:"usage",option:"Pipeline",$$slots:{default:[Ds]},$$scope:{ctx:U}}}),d=new Gs({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ys]},$$scope:{ctx:U}}}),{c(){f(s.$$.fragment),J=o(),f(d.$$.fragment)},l(b){g(s.$$.fragment,b),J=n(b),g(d.$$.fragment,b)},m(b,c){M(s,b,c),u(b,J,c),M(d,b,c),w=!0},p(b,c){const I={};c&2&&(I.$$scope={dirty:c,ctx:b}),s.$set(I);const Z={};c&2&&(Z.$$scope={dirty:c,ctx:b}),d.$set(Z)},i(b){w||(_(s.$$.fragment,b),_(d.$$.fragment,b),w=!0)},o(b){T(s.$$.fragment,b),T(d.$$.fragment,b),w=!1},d(b){b&&i(J),y(s,b),y(d,b)}}}function Ps(U){let s,J="Example:",d,w,b;return w=new F({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1pc3RyYWwzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTJDJTIwTWlzdHJhbDNDb25maWclMkMlMjBQaXh0cmFsVmlzaW9uQ29uZmlnJTJDJTIwTWlzdHJhbENvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBQaXh0cmFsLXZpc2lvbiUyMGNvbmZpZyUwQXZpc2lvbl9jb25maWclMjAlM0QlMjBQaXh0cmFsVmlzaW9uQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTWlzdHJhbCUyMGNvbmZpZyUwQXRleHRfY29uZmlnJTIwJTNEJTIwTWlzdHJhbENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyME1pc3RyYWwzJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBNaXN0cmFsM0NvbmZpZyh2aXNpb25fY29uZmlnJTJDJTIwdGV4dF9jb25maWcpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMG1pc3RyYWwzLjElMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBNaXN0cmFsM0ZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbihjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Mistral3ForConditionalGeneration, Mistral3Config, PixtralVisionConfig, MistralConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Pixtral-vision config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>vision_config = PixtralVisionConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Mistral config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>text_config = MistralConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Mistral3 configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Mistral3Config(vision_config, text_config)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the mistral3.1 configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Mistral3ForConditionalGeneration(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){s=a("p"),s.textContent=J,d=o(),f(w.$$.fragment)},l(c){s=r(c,"P",{"data-svelte-h":!0}),m(s)!=="svelte-11lpom8"&&(s.textContent=J),d=n(c),g(w.$$.fragment,c)},m(c,I){u(c,s,I),u(c,d,I),M(w,c,I),b=!0},p:E,i(c){b||(_(w.$$.fragment,c),b=!0)},o(c){T(w.$$.fragment,c),b=!1},d(c){c&&(i(s),i(d)),y(w,c)}}}function Os(U){let s,J="<code>mistral-common</code> is the official tokenizer library for Mistral AI models. To use it, you need to install it with:",d,w,b;return w=new F({props:{code:"cGlwJTIwaW5zdGFsbCUyMHRyYW5zZm9ybWVycyU1Qm1pc3RyYWwtY29tbW9uJTVE",highlighted:"pip install transformers[mistral-common]",wrap:!1}}),{c(){s=a("p"),s.innerHTML=J,d=o(),f(w.$$.fragment)},l(c){s=r(c,"P",{"data-svelte-h":!0}),m(s)!=="svelte-m7z88d"&&(s.innerHTML=J),d=n(c),g(w.$$.fragment,c)},m(c,I){u(c,s,I),u(c,d,I),M(w,c,I),b=!0},p:E,i(c){b||(_(w.$$.fragment,c),b=!0)},o(c){T(w.$$.fragment,c),b=!1},d(c){c&&(i(s),i(d)),y(w,c)}}}function Ks(U){let s,J=`If the <code>encoded_inputs</code> passed are dictionary of numpy arrays, PyTorch tensors, the
result will use the same type unless you provide a different tensor type with <code>return_tensors</code>. In the case of
PyTorch tensors, you will lose the specific device of your tensors however.`;return{c(){s=a("p"),s.innerHTML=J},l(d){s=r(d,"P",{"data-svelte-h":!0}),m(s)!=="svelte-mer66"&&(s.innerHTML=J)},m(d,w){u(d,s,w)},p:E,d(d){d&&i(s)}}}function ea(U){let s,J=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=a("p"),s.innerHTML=J},l(d){s=r(d,"P",{"data-svelte-h":!0}),m(s)!=="svelte-fincs2"&&(s.innerHTML=J)},m(d,w){u(d,s,w)},p:E,d(d){d&&i(s)}}}function ta(U){let s,J=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=a("p"),s.innerHTML=J},l(d){s=r(d,"P",{"data-svelte-h":!0}),m(s)!=="svelte-fincs2"&&(s.innerHTML=J)},m(d,w){u(d,s,w)},p:E,d(d){d&&i(s)}}}function oa(U){let s,J="Example:",d,w,b;return w=new F({props:{code:"ZnJvbSUyMFBJTCUyMGltcG9ydCUyMEltYWdlJTBBaW1wb3J0JTIwcmVxdWVzdHMlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyME1pc3RyYWwzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBbW9kZWwlMjAlM0QlMjBNaXN0cmFsM0ZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybWlzdHJhbGFpJTJGTWlzdHJhbC1TbWFsbC0zLjEtMjRCLUluc3RydWN0LTI1MDMlMjIpJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIybWlzdHJhbGFpJTJGTWlzdHJhbC1TbWFsbC0zLjEtMjRCLUluc3RydWN0LTI1MDMlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIyJTNDcyUzRSU1QklOU1QlNUQlNUJJTUclNURXaGF0JTIwaXMlMjB0aGUlMjBpbWFnZSUzRiU1QiUyRklOU1QlNUQlMjIlMEF1cmwlMjAlM0QlMjAlMjJodHRwJTNBJTJGJTJGaW1hZ2VzLmNvY29kYXRhc2V0Lm9yZyUyRnZhbDIwMTclMkYwMDAwMDAwMzk3NjkuanBnJTIyJTBBaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKGltYWdlcyUzRGltYWdlJTJDJTIwdGV4dCUzRHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMTUpJTBBcHJvY2Vzc29yLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, Mistral3ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Mistral3ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;mistralai/Mistral-Small-3.1-24B-Instruct-2503&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;mistralai/Mistral-Small-3.1-24B-Instruct-2503&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;&lt;s&gt;[INST][IMG]What is the image?[/INST]&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(images=image, text=prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(**inputs, max_new_tokens=<span class="hljs-number">15</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;What is the image?The image depicts two cats lying on a pink blanket.&quot;</span>`,wrap:!1}}),{c(){s=a("p"),s.textContent=J,d=o(),f(w.$$.fragment)},l(c){s=r(c,"P",{"data-svelte-h":!0}),m(s)!=="svelte-11lpom8"&&(s.textContent=J),d=n(c),g(w.$$.fragment,c)},m(c,I){u(c,s,I),u(c,d,I),M(w,c,I),b=!0},p:E,i(c){b||(_(w.$$.fragment,c),b=!0)},o(c){T(w.$$.fragment,c),b=!1},d(c){c&&(i(s),i(d)),y(w,c)}}}function na(U){let s,J,d,w,b,c="<em>This model was released on 2025-01-30 and added to Hugging Face Transformers on 2025-03-18.</em>",I,Z,Hn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Pt,ue,Ot,he,An='<a href="https://mistral.ai/news/mistral-small-3" rel="nofollow">Mistral 3</a> is a latency optimized model with a lot fewer layers to reduce the time per forward pass. This model adds vision understanding and supports long context lengths of up to 128K tokens without compromising performance.',Kt,fe,Sn='You can find the original Mistral 3 checkpoints under the <a href="https://huggingface.co/mistralai/models?search=mistral-small-3" rel="nofollow">Mistral AI</a> organization.',eo,H,to,ge,Xn='The example below demonstrates how to generate text for an image with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> and the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',oo,A,no,Me,so,_e,Dn="<li>Mistral 3 supports text-only generation.</li>",ao,Te,ro,ye,Yn=`/_/\\
( o.o )`,io,S,at,Ln="^ <",Vo,we,lo,be,Pn="<li>Mistral 3 accepts batched image and text inputs.</li>",co,Je,mo,ve,On="<li>Mistral 3 also supported batched image and text inputs with a different number of images for each text. The example below quantizes the model with bitsandbytes.</li>",po,ke,uo,Ue,ho,V,Ce,$o,rt,Kn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration">Mistral3ForConditionalGeneration</a>. It is used to instantiate an
Mistral3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
<a href="https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503" rel="nofollow">mistralai/Mistral-Small-3.1-24B-Instruct-2503</a>`,Bo,it,es=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Zo,X,fo,Ie,go,p,xe,No,lt,ts="Class to wrap <code>mistral-common</code> tokenizers.",Wo,D,Go,dt,os="Otherwise the tokenizer falls back to the Transformers implementation of the tokenizer.",qo,ct,ns='For more info on <code>mistral-common</code>, see <a href="https://github.com/mistralai/mistral-common" rel="nofollow">mistral-common</a>.',Qo,mt,ss=`This class is a wrapper around a <code>mistral_common.tokens.tokenizers.mistral.MistralTokenizer</code>.
It provides a Hugging Face compatible interface to tokenize using the official mistral-common tokenizer.`,Ro,pt,as="Supports the following methods from the <code>PreTrainedTokenizerBase</code> class:",Fo,ut,rs='<li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.get_vocab">get_vocab()</a>: Returns the vocabulary as a dictionary of token to index.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.encode">encode()</a>: Encode a string to a list of integers.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.decode">decode()</a>: Decode a list of integers to a string.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.batch_decode">batch_decode()</a>: Decode a batch of list of integers to a list of strings.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.convert_tokens_to_ids">convert_tokens_to_ids()</a>: Convert a list of tokens to a list of integers.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.convert_ids_to_tokens">convert_ids_to_tokens()</a>: Convert a list of integers to a list of tokens.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.tokenize">tokenize()</a>: Tokenize a string.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.get_special_tokens_mask">get_special_tokens_mask()</a>: Get the special tokens mask for a list of tokens.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.prepare_for_model">prepare_for_model()</a>: Prepare a list of inputs for the model.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.pad">pad()</a>: Pad a list of inputs to the same length.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.truncate_sequences">truncate_sequences()</a>: Truncate a list of sequences to the same length.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.apply_chat_template">apply_chat_template()</a>: Apply a chat template to a list of messages.</li> <li><code>__call__()</code>: Tokenize a string or a list of strings.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.from_pretrained">from_pretrained()</a>: Download and cache a pretrained tokenizer from the Hugging Face model hub or local directory.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.save_pretrained">save_pretrained()</a>: Save a tokenizer to a directory, so it can be reloaded using the <code>from_pretrained</code> class method.</li> <li><a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub">push_to_hub()</a>: Upload tokenizer to the Hugging Face model hub.</li>',Eo,ht,is="Here are the key differences with the <code>PreTrainedTokenizerBase</code> class:",Ho,ft,ls='<li>Pair of sequences are not supported. The signature have been kept for compatibility but all arguments related to pair of sequences are ignored. The return values of pairs are returned as <code>None</code>.</li> <li>The <code>is_split_into_words</code> argument is not supported.</li> <li>The <code>return_token_type_ids</code> argument is not supported.</li> <li>It is not possible to add new tokens to the tokenizer. Also the special tokens are handled differently from Transformers. In <code>mistral-common</code>, special tokens are never encoded directly. This means that: <code>tokenizer.encode(&quot;&lt;s&gt;&quot;)</code> will not return the ID of the <code>&lt;s&gt;</code> token. Instead, it will return a list of IDs corresponding to the tokenization of the string <code>&quot;&lt;s&gt;&quot;</code>. For more information, see the <a href="https://mistralai.github.io/mistral-common/usage/tokenizers/#special-tokens" rel="nofollow">mistral-common documentation</a>.</li>',Ao,gt,ds='If you have suggestions to improve this class, please open an issue on the <a href="https://github.com/mistralai/mistral-common/issues" rel="nofollow">mistral-common GitHub repository</a> if it is related to the tokenizer or on the <a href="https://github.com/huggingface/transformers/issues" rel="nofollow">Transformers GitHub repository</a> if it is related to the Hugging Face interface.',So,Y,je,Xo,Mt,cs=`Converts a list of dictionaries with <code>&quot;role&quot;</code> and <code>&quot;content&quot;</code> keys to a list of token
ids.`,Do,L,ze,Yo,_t,ms="Convert a list of lists of token ids into a list of strings by calling decode.",Lo,P,Ve,Po,Tt,ps=`Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
added tokens.`,Oo,O,$e,Ko,yt,us=`Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
vocabulary.`,en,K,Be,tn,wt,hs=`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`,on,ee,Ze,nn,bt,fs="Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.",sn,te,Ne,an,Jt,gs=`Instantiate a <code>MistralCommonTokenizer</code> from a predefined
tokenizer.`,rn,oe,We,ln,vt,Ms=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> or <code>encode_plus</code> methods.`,dn,W,Ge,cn,kt,_s="Returns the vocabulary as a dictionary of token to index.",mn,Ut,Ts=`This is a lossy conversion. There may be multiple token ids that decode to the same
string due to partial UTF-8 byte sequences being converted to �.`,pn,$,qe,un,Ct,ys=`Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
in the batch.`,hn,It,ws=`Padding side (left/right) padding token ids are defined at the tokenizer level (with <code>self.padding_side</code>,
<code>self.pad_token_id</code>).`,fn,ne,gn,se,Qe,Mn,xt,bs=`Prepares a sequence of input id so that it can be used by the model. It
adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
manages a moving window (with user defined stride) for overflowing tokens.`,_n,G,Re,Tn,jt,Js="Save the full tokenizer state.",yn,zt,vs=`This method make sure the full tokenizer can then be re-loaded using the
<code>~MistralCommonTokenizer.tokenization_mistral_common.from_pretrained</code> class method.`,wn,q,Fe,bn,Vt,ks="Converts a string into a sequence of tokens, using the tokenizer.",Jn,$t,Us="Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.",vn,ae,Ee,kn,Bt,Cs="Truncates a sequence pair in-place following the strategy.",Mo,He,_o,x,Ae,Un,Zt,Is="The Mistral3 model which consists of a vision backbone and a language model, without a language modeling head.",Cn,Nt,xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,In,Wt,js=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xn,Q,Se,jn,Gt,zs='The <a href="/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Model">Mistral3Model</a> forward method, overrides the <code>__call__</code> special method.',zn,re,Vn,ie,Xe,$n,qt,Vs="Obtains image last hidden states from the vision tower and apply multimodal projection.",Bn,le,De,Zn,Qt,$s=`Obtains multimodal placeholder mask from <code>input_ids</code> or <code>inputs_embeds</code>, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.`,To,Ye,yo,j,Le,Nn,Rt,Bs="The MISTRAL3 model which consists of a vision backbone and a language model.",Wn,Ft,Zs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Gn,Et,Ns=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,qn,B,Pe,Qn,Ht,Ws='The <a href="/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3ForConditionalGeneration">Mistral3ForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Rn,de,Fn,ce,wo,Oe,bo,Lt,Jo;return ue=new Yt({props:{title:"Mistral 3",local:"mistral-3",headingTag:"h1"}}),H=new zo({props:{warning:!1,$$slots:{default:[Xs]},$$scope:{ctx:U}}}),A=new Ss({props:{id:"usage",options:["Pipeline","AutoModel"],$$slots:{default:[Ls]},$$scope:{ctx:U}}}),Me=new Yt({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Te=new F({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dCUyQyUyMGluZmVyX2RldmljZSUwQSUwQXRvcmNoX2RldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWxfY2hlY2twb2ludCUyMCUzRCUyMCUyMi5taXN0cmFsYWklMkZNaXN0cmFsLVNtYWxsLTMuMS0yNEItSW5zdHJ1Y3QtMjUwMyUyMiUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2NoZWNrcG9pbnQpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JJbWFnZVRleHRUb1RleHQuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2NoZWNrcG9pbnQlMkMlMjBkZXZpY2VfbWFwJTNEdG9yY2hfZGV2aWNlJTJDJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiklMEElMEFTWVNURU1fUFJPTVBUJTIwJTNEJTIwJTIyWW91JTIwYXJlJTIwYSUyMGNvbnZlcnNhdGlvbmFsJTIwYWdlbnQlMjB0aGF0JTIwYWx3YXlzJTIwYW5zd2VycyUyMHN0cmFpZ2h0JTIwdG8lMjB0aGUlMjBwb2ludCUyQyUyMGFsd2F5cyUyMGVuZCUyMHlvdXIlMjBhY2N1cmF0ZSUyMHJlc3BvbnNlJTIwd2l0aCUyMGFuJTIwQVNDSUklMjBkcmF3aW5nJTIwb2YlMjBhJTIwY2F0LiUyMiUwQXVzZXJfcHJvbXB0JTIwJTNEJTIwJTIyR2l2ZSUyMG1lJTIwNSUyMG5vbi1mb3JtYWwlMjB3YXlzJTIwdG8lMjBzYXklMjAnU2VlJTIweW91JTIwbGF0ZXInJTIwaW4lMjBGcmVuY2guJTIyJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIyc3lzdGVtJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMFNZU1RFTV9QUk9NUFQlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjB1c2VyX3Byb21wdCU3RCUyQyUwQSU1RCUwQSUwQXRleHQlMjAlM0QlMjBwcm9jZXNzb3IuYXBwbHlfY2hhdF90ZW1wbGF0ZShtZXNzYWdlcyUyQyUyMHRva2VuaXplJTNERmFsc2UlMkMlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlKSUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3Nvcih0ZXh0JTNEdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKDAlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYpJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDUwJTJDJTIwZG9fc2FtcGxlJTNERmFsc2UpJTBBZGVjb2RlZF9vdXRwdXQlMjAlM0QlMjBwcm9jZXNzb3IuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyU1QiUzQSUyQyUyMGlucHV0cyU1QiUyMmlucHV0X2lkcyUyMiU1RC5zaGFwZSU1QjElNUQlMjAlM0ElNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklNUIwJTVEJTBBJTBBcHJpbnQoZGVjb2RlZF9vdXRwdXQpJTBBJTIyMS4lMjAlQzMlODAlMjBwbHVzJTIwdGFyZCElMEElMjAyLiUyMFNhbHV0JTJDJTIwJUMzJUEwJTIwcGx1cyElMEElMjAzLiUyMCVDMyU4MCUyMHRvdXRlISUwQSUyMDQuJTIwJUMzJTgwJTIwbGElMjBwcm9jaGFpbmUhJTBBJTIwNS4lMjBKZSUyMG1lJTIwY2Fzc2UlMkMlMjAlQzMlQTAlMjBwbHVzISUwQQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForImageTextToText, infer_device

torch_device = infer_device()
model_checkpoint = <span class="hljs-string">&quot;.mistralai/Mistral-Small-3.1-24B-Instruct-2503&quot;</span>
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, dtype=torch.bfloat16)

SYSTEM_PROMPT = <span class="hljs-string">&quot;You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat.&quot;</span>
user_prompt = <span class="hljs-string">&quot;Give me 5 non-formal ways to say &#x27;See you later&#x27; in French.&quot;</span>

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: SYSTEM_PROMPT},
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: user_prompt},
]

text = processor.apply_chat_template(messages, tokenize=<span class="hljs-literal">False</span>, add_generation_prompt=<span class="hljs-literal">True</span>)
inputs = processor(text=text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(<span class="hljs-number">0</span>, dtype=torch.float16)
generate_ids = model.generate(**inputs, max_new_tokens=<span class="hljs-number">50</span>, do_sample=<span class="hljs-literal">False</span>)
decoded_output = processor.batch_decode(generate_ids[:, inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape[<span class="hljs-number">1</span>] :], skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-built_in">print</span>(decoded_output)
<span class="hljs-string">&quot;1. À plus tard!
 2. Salut, à plus!
 3. À toute!
 4. À la prochaine!
 5. Je me casse, à plus!
</span>`,wrap:!1}}),we=new F({props:{code:"",highlighted:"",wrap:!1}}),Je=new F({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dCUyQyUyMGluZmVyX2RldmljZSUwQSUwQXRvcmNoX2RldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWxfY2hlY2twb2ludCUyMCUzRCUyMCUyMm1pc3RyYWxhaSUyRk1pc3RyYWwtU21hbGwtMy4xLTI0Qi1JbnN0cnVjdC0yNTAzJTIyJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQobW9kZWxfY2hlY2twb2ludCklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dC5mcm9tX3ByZXRyYWluZWQobW9kZWxfY2hlY2twb2ludCUyQyUyMGRldmljZV9tYXAlM0R0b3JjaF9kZXZpY2UlMkMlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2KSUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTJDJTIwJTIydXJsJTIyJTNBJTIwJTIyaHR0cHMlM0ElMkYlMkZsbGF2YS12bC5naXRodWIuaW8lMkZzdGF0aWMlMkZpbWFnZXMlMkZ2aWV3LmpwZyUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMldyaXRlJTIwYSUyMGhhaWt1JTIwZm9yJTIwdGhpcyUyMGltYWdlJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJjb250ZW50JTIyJTNBJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTJDJTIwJTIydXJsJTIyJTNBJTIwJTIyaHR0cHMlM0ElMkYlMkZ3d3cuaWxhbmtlbG1hbi5vcmclMkZzdG9wc2lnbnMlMkZhdXN0cmFsaWEuanBnJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyRGVzY3JpYmUlMjB0aGlzJTIwaW1hZ2UlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlNUQlMkMlMEElMjAlNUQlMEElMEElMEElMjBpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYXBwbHlfY2hhdF90ZW1wbGF0ZShtZXNzYWdlcyUyQyUyMHBhZGRpbmclM0RUcnVlJTJDJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUyQyUyMHRva2VuaXplJTNEVHJ1ZSUyQyUyMHJldHVybl9kaWN0JTNEVHJ1ZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSUyQyUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYpJTBBJTBBJTIwb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDI1KSUwQSUwQSUyMGRlY29kZWRfb3V0cHV0cyUyMCUzRCUyMHByb2Nlc3Nvci5iYXRjaF9kZWNvZGUob3V0cHV0JTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTIwZGVjb2RlZF9vdXRwdXRzJTBBJTVCJTIyV3JpdGUlMjBhJTIwaGFpa3UlMjBmb3IlMjB0aGlzJTIwaW1hZ2VDYWxtJTIwd2F0ZXJzJTIwcmVmbGVjdCU1Q25XaGlzcGVycyUyMG9mJTIwdGhlJTIwZm9yZXN0J3MlMjBicmVhdGglNUNuUGVhY2UlMjBvbiUyMHdvb2RlbiUyMHBhdGglMjIlMEElMkMlMjAlMjJEZXNjcmliZSUyMHRoaXMlMjBpbWFnZVRoZSUyMGltYWdlJTIwZGVwaWN0cyUyMGElMjB2aWJyYW50JTIwc3RyZWV0JTIwc2NlbmUlMjBpbiUyMHdoYXQlMjBhcHBlYXJzJTIwdG8lMjBiZSUyMGElMjBDaGluYXRvd24lMjBkaXN0cmljdC4lMjBUaGUlMjBmb2NhbCUyMHBvaW50JTIwaXMlMjBhJTIwdHJhZGl0aW9uYWwlMjBDaGluZXNlJTIyJTVE",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForImageTextToText, infer_device

torch_device = infer_device()
model_checkpoint = <span class="hljs-string">&quot;mistralai/Mistral-Small-3.1-24B-Instruct-2503&quot;</span>
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, dtype=torch.bfloat16)

messages = [
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
                 {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://www.ilankelman.org/stopsigns/australia.jpg&quot;</span>},
                 {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;Describe this image&quot;</span>},
             ],
         },
     ],
 ]


 inputs = processor.apply_chat_template(messages, padding=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device, dtype=torch.bfloat16)

 output = model.generate(**inputs, max_new_tokens=<span class="hljs-number">25</span>)

 decoded_outputs = processor.batch_decode(output, skip_special_tokens=<span class="hljs-literal">True</span>)
 decoded_outputs
[<span class="hljs-string">&quot;Write a haiku for this imageCalm waters reflect\\nWhispers of the forest&#x27;s breath\\nPeace on wooden path&quot;</span>
, <span class="hljs-string">&quot;Describe this imageThe image depicts a vibrant street scene in what appears to be a Chinatown district. The focal point is a traditional Chinese&quot;</span>]`,wrap:!1}}),ke=new F({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbEZvckltYWdlVGV4dFRvVGV4dCUyQyUyMEJpdHNBbmRCeXRlc0NvbmZpZyUyQyUyMGluZmVyX2RldmljZSUwQSUwQXRvcmNoX2RldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWxfY2hlY2twb2ludCUyMCUzRCUyMCUyMm1pc3RyYWxhaSUyRk1pc3RyYWwtU21hbGwtMy4xLTI0Qi1JbnN0cnVjdC0yNTAzJTIyJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQobW9kZWxfY2hlY2twb2ludCklMEFxdWFudGl6YXRpb25fY29uZmlnJTIwJTNEJTIwQml0c0FuZEJ5dGVzQ29uZmlnKGxvYWRfaW5fNGJpdCUzRFRydWUpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JJbWFnZVRleHRUb1RleHQuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMG1vZGVsX2NoZWNrcG9pbnQlMkMlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEcXVhbnRpemF0aW9uX2NvbmZpZyUwQSUyMCklMEElMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCU1QiUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCU3QiUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTBBJTIwJUMyJUEwJTIwJUMyJUEwJTIwJUMyJUEwJTIwJUMyJUEwJTIwJUMyJUEwJTIwJUMyJUEwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUyMCUyMnVybCUyMiUzQSUyMCUyMmh0dHBzJTNBJTJGJTJGbGxhdmEtdmwuZ2l0aHViLmlvJTJGc3RhdGljJTJGaW1hZ2VzJTJGdmlldy5qcGclMjIlN0QlMkMlMEElMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjAlMjJXcml0ZSUyMGElMjBoYWlrdSUyMGZvciUyMHRoaXMlMjBpbWFnZSUyMiU3RCUyQyUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCU1RCUyQyUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCU3RCUyQyUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCU1RCUyQyUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCU1QiUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCU3QiUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTBBJTIwJUMyJUEwJTIwJUMyJUEwJTIwJUMyJUEwJTIwJUMyJUEwJTIwJUMyJUEwJTIwJUMyJUEwJTIwJTIyY29udGVudCUyMiUzQSUyMCU1QiUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUyMCUyMnVybCUyMiUzQSUyMCUyMmh0dHBzJTNBJTJGJTJGY2RuLmJyaXRhbm5pY2EuY29tJTJGNjElMkY5MzA2MS0wNTAtOTkxNDdEQ0UlMkZTdGF0dWUtb2YtTGliZXJ0eS1Jc2xhbmQtTmV3LVlvcmstQmF5LmpwZyUyMiU3RCUyQyUwQSUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCVDMiVBMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJpbWFnZSUyMiUyQyUyMCUyMnVybCUyMiUzQSUyMCUyMmh0dHBzJTNBJTJGJTJGdGh1bWJzLmRyZWFtc3RpbWUuY29tJTJGYiUyRmdvbGRlbi1nYXRlLWJyaWRnZS1zYW4tZnJhbmNpc2NvLXB1cnBsZS1mbG93ZXJzLWNhbGlmb3JuaWEtZWNoaXVtLWNhbmRpY2Fucy0zNjgwNTk0Ny5qcGclMjIlN0QlMkMlMEElMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjAlMjJUaGVzZSUyMGltYWdlcyUyMGRlcGljdCUyMHR3byUyMGRpZmZlcmVudCUyMGxhbmRtYXJrcy4lMjBDYW4lMjB5b3UlMjBpZGVudGlmeSUyMHRoZW0lM0YlMjIlN0QlMkMlMEElMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlNUQlMkMlMEElMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlQzIlQTAlMjAlN0QlMkMlMEElMjAlQzIlQTAlMjAlQzIlQTAlMjAlNUQlMkMlMEElMjAlNUQlMEElMEElMjBpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYXBwbHlfY2hhdF90ZW1wbGF0ZShtZXNzYWdlcyUyQyUyMHBhZGRpbmclM0RUcnVlJTJDJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUyQyUyMHRva2VuaXplJTNEVHJ1ZSUyQyUyMHJldHVybl9kaWN0JTNEVHJ1ZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSUyQyUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYpJTBBJTBBJTIwb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDI1KSUwQSUwQSUyMGRlY29kZWRfb3V0cHV0cyUyMCUzRCUyMHByb2Nlc3Nvci5iYXRjaF9kZWNvZGUob3V0cHV0JTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTIwZGVjb2RlZF9vdXRwdXRzJTBBJTVCJTIyV3JpdGUlMjBhJTIwaGFpa3UlMjBmb3IlMjB0aGlzJTIwaW1hZ2VTdXJlJTJDJTIwaGVyZSUyMGlzJTIwYSUyMGhhaWt1JTIwaW5zcGlyZWQlMjBieSUyMHRoZSUyMGltYWdlJTNBJTVDbiU1Q25DYWxtJTIwbGFrZSdzJTIwd29vZGVuJTIwcGF0aCU1Q25TaWxlbnQlMjBmb3Jlc3QlMjBzdGFuZHMlMjBndWFyZCU1Q24lMjIlMkMlMjAlMjJUaGVzZSUyMGltYWdlcyUyMGRlcGljdCUyMHR3byUyMGRpZmZlcmVudCUyMGxhbmRtYXJrcy4lMjBDYW4lMjB5b3UlMjBpZGVudGlmeSUyMHRoZW0lM0YlMjBDZXJ0YWlubHkhJTIwVGhlJTIwaW1hZ2VzJTIwZGVwaWN0JTIwdHdvJTIwaWNvbmljJTIwbGFuZG1hcmtzJTNBJTVDbiU1Q24xLiUyMFRoZSUyMGZpcnN0JTIwaW1hZ2UlMjBzaG93cyUyMHRoZSUyMFN0YXR1ZSUyMG9mJTIwTGliZXJ0eSUyMGluJTIwTmV3JTIwWW9yayUyMENpdHkuJTIyJTVE",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, infer_device

torch_device = infer_device()
model_checkpoint = <span class="hljs-string">&quot;mistralai/Mistral-Small-3.1-24B-Instruct-2503&quot;</span>
processor = AutoProcessor.from_pretrained(model_checkpoint)
quantization_config = BitsAndBytesConfig(load_in_4bit=<span class="hljs-literal">True</span>)
model = AutoModelForImageTextToText.from_pretrained(
     model_checkpoint, quantization_config=quantization_config
 )

messages = [
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
                 {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg&quot;</span>},
                 {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg&quot;</span>},
                 {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;These images depict two different landmarks. Can you identify them?&quot;</span>},
             ],
         },
     ],
 ]

 inputs = processor.apply_chat_template(messages, padding=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device, dtype=torch.bfloat16)

 output = model.generate(**inputs, max_new_tokens=<span class="hljs-number">25</span>)

 decoded_outputs = processor.batch_decode(output, skip_special_tokens=<span class="hljs-literal">True</span>)
 decoded_outputs
[<span class="hljs-string">&quot;Write a haiku for this imageSure, here is a haiku inspired by the image:\\n\\nCalm lake&#x27;s wooden path\\nSilent forest stands guard\\n&quot;</span>, <span class="hljs-string">&quot;These images depict two different landmarks. Can you identify them? Certainly! The images depict two iconic landmarks:\\n\\n1. The first image shows the Statue of Liberty in New York City.&quot;</span>]`,wrap:!1}}),Ue=new Yt({props:{title:"Mistral3Config",local:"transformers.Mistral3Config",headingTag:"h2"}}),Ce=new C({props:{name:"class transformers.Mistral3Config",anchor:"transformers.Mistral3Config",parameters:[{name:"vision_config",val:" = None"},{name:"text_config",val:" = None"},{name:"image_token_index",val:" = 10"},{name:"projector_hidden_act",val:" = 'gelu'"},{name:"vision_feature_layer",val:" = -1"},{name:"multimodal_projector_bias",val:" = False"},{name:"spatial_merge_size",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Mistral3Config.vision_config",description:`<strong>vision_config</strong> (<code>Union[AutoConfig, dict]</code>,  <em>optional</em>, defaults to <code>PixtralVisionConfig</code>) &#x2014;
The config object or dictionary of the vision backbone.`,name:"vision_config"},{anchor:"transformers.Mistral3Config.text_config",description:`<strong>text_config</strong> (<code>Union[AutoConfig, dict]</code>, <em>optional</em>, defaults to <code>MistralConfig</code>) &#x2014;
The config object or dictionary of the text backbone.`,name:"text_config"},{anchor:"transformers.Mistral3Config.image_token_index",description:`<strong>image_token_index</strong> (<code>int</code>, <em>optional</em>, defaults to 10) &#x2014;
The image token index to encode the image prompt.`,name:"image_token_index"},{anchor:"transformers.Mistral3Config.projector_hidden_act",description:`<strong>projector_hidden_act</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The activation function used by the multimodal projector.`,name:"projector_hidden_act"},{anchor:"transformers.Mistral3Config.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>Union[int, list[int]]</code>, <em>optional</em>, defaults to -1) &#x2014;
The index of the layer to select the vision feature. If multiple indices are provided,
the vision feature of the corresponding indices will be concatenated to form the
vision features.`,name:"vision_feature_layer"},{anchor:"transformers.Mistral3Config.multimodal_projector_bias",description:`<strong>multimodal_projector_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use bias in the multimodal projector.`,name:"multimodal_projector_bias"},{anchor:"transformers.Mistral3Config.spatial_merge_size",description:`<strong>spatial_merge_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The downsampling factor for the spatial merge operation.`,name:"spatial_merge_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral3/configuration_mistral3.py#L21"}}),X=new En({props:{anchor:"transformers.Mistral3Config.example",$$slots:{default:[Ps]},$$scope:{ctx:U}}}),Ie=new Yt({props:{title:"MistralCommonTokenizer",local:"transformers.MistralCommonTokenizer",headingTag:"h2"}}),xe=new C({props:{name:"class transformers.MistralCommonTokenizer",anchor:"transformers.MistralCommonTokenizer",parameters:[{name:"tokenizer_path",val:": typing.Union[str, os.PathLike, pathlib.Path]"},{name:"mode",val:": ValidationMode = <ValidationMode.test: 'test'>"},{name:"model_max_length",val:": int = 1000000000000000019884624838656"},{name:"padding_side",val:": str = 'left'"},{name:"truncation_side",val:": str = 'right'"},{name:"model_input_names",val:": typing.Optional[list[str]] = None"},{name:"clean_up_tokenization_spaces",val:": bool = False"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L158"}}),D=new En({props:{anchor:"transformers.MistralCommonTokenizer.example",$$slots:{default:[Os]},$$scope:{ctx:U}}}),je=new C({props:{name:"apply_chat_template",anchor:"transformers.MistralCommonTokenizer.apply_chat_template",parameters:[{name:"conversation",val:": typing.Union[list[dict[str, str]], list[list[dict[str, str]]]]"},{name:"tools",val:": typing.Optional[list[typing.Union[dict, typing.Callable]]] = None"},{name:"continue_final_message",val:": bool = False"},{name:"tokenize",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": bool = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_dict",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.conversation",description:`<strong>conversation</strong> (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) &#x2014; A list of dicts
with &#x201C;role&#x201D; and &#x201C;content&#x201D; keys, representing the chat history so far.`,name:"conversation"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.tools",description:`<strong>tools</strong> (<code>List[Union[Dict, Callable]]</code>, <em>optional</em>) &#x2014;
A list of tools (callable functions) that will be accessible to the model. If the template does not
support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
giving the name, description and argument types for the tool. See our
<a href="https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use" rel="nofollow">chat templating guide</a>
for more information.`,name:"tools"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.continue_final_message",description:`<strong>continue_final_message</strong> (bool, <em>optional</em>) &#x2014;
If this is set, the chat will be formatted so that the final
message in the chat is open-ended, without any EOS tokens. The model will continue this message
rather than starting a new one. This allows you to &#x201C;prefill&#x201D; part of
the model&#x2019;s response for it. Cannot be used at the same time as <code>add_generation_prompt</code>.`,name:"continue_final_message"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.tokenize",description:`<strong>tokenize</strong> (<code>bool</code>, defaults to <code>True</code>) &#x2014;
Whether to tokenize the output. If <code>False</code>, the output will be a string.`,name:"tokenize"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.truncation",description:`<strong>truncation</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to truncate sequences at the maximum length. Has no effect if tokenize is <code>False</code>.`,name:"truncation"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is <code>False</code>. If
not specified, the tokenizer&#x2019;s <code>max_length</code> attribute will be used as a default.`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Has no effect if tokenize is <code>False</code>. Acceptable
values are:</p>
<ul>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to return a dictionary with named outputs. Has no effect if tokenize is <code>False</code>.
If at least one conversation contains an image, its pixel values will be returned in the <code>pixel_values</code> key.`,name:"return_dict"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.apply_chat_template</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1368",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of token ids representing the tokenized chat so far, including control
tokens. This output is ready to pass to the model, either directly or via methods like <code>generate()</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Union[str, List[int], List[str], List[List[int]], BatchEncoding]</code></p>
`}}),ze=new C({props:{name:"batch_decode",anchor:"transformers.MistralCommonTokenizer.batch_decode",parameters:[{name:"sequences",val:": typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.batch_decode.sequences",description:`<strong>sequences</strong> (<code>Union[List[int], List[List[int]], np.ndarray, torch.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"sequences"},{anchor:"transformers.MistralCommonTokenizer.batch_decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.MistralCommonTokenizer.batch_decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.MistralCommonTokenizer.batch_decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.batch_decode</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L476",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of decoded sentences.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[str]</code></p>
`}}),Ve=new C({props:{name:"convert_ids_to_tokens",anchor:"transformers.MistralCommonTokenizer.convert_ids_to_tokens",parameters:[{name:"ids",val:": typing.Union[int, list[int]]"},{name:"skip_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.convert_ids_to_tokens.ids",description:`<strong>ids</strong> (<code>int</code> or <code>List[int]</code>) &#x2014;
The token id (or token ids) to convert to tokens.`,name:"ids"},{anchor:"transformers.MistralCommonTokenizer.convert_ids_to_tokens.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L523",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded token(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code> or <code>List[str]</code></p>
`}}),$e=new C({props:{name:"convert_tokens_to_ids",anchor:"transformers.MistralCommonTokenizer.convert_tokens_to_ids",parameters:[{name:"tokens",val:": typing.Union[str, list[str]]"}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.convert_tokens_to_ids.tokens",description:"<strong>tokens</strong> (<code>str</code> or <code>List[str]</code>) &#x2014; One or several token(s) to convert to token id(s).",name:"tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L571",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token id or list of token ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code> or <code>List[int]</code></p>
`}}),Be=new C({props:{name:"decode",anchor:"transformers.MistralCommonTokenizer.decode",parameters:[{name:"token_ids",val:": typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.decode.token_ids",description:`<strong>token_ids</strong> (<code>Union[int, List[int], np.ndarray, torch.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"token_ids"},{anchor:"transformers.MistralCommonTokenizer.decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.MistralCommonTokenizer.decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.MistralCommonTokenizer.decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.decode</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L434",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded sentence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),Ze=new C({props:{name:"encode",anchor:"transformers.MistralCommonTokenizer.encode",parameters:[{name:"text",val:": typing.Union[str, list[int]]"},{name:"text_pair",val:": None = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.encode.text",description:`<strong>text</strong> (<code>str</code> or <code>List[int]</code>) &#x2014;
The first sequence to be encoded. This can be a string or a list of integers (tokenized string ids).`,name:"text"},{anchor:"transformers.MistralCommonTokenizer.encode.text_pair",description:`<strong>text_pair</strong> (<code>None</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.encode</code>. Kept to match <code>PreTrainedTokenizerBase.encode</code> signature.`,name:"text_pair"},{anchor:"transformers.MistralCommonTokenizer.encode.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.MistralCommonTokenizer.encode.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MistralCommonTokenizer.encode.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.MistralCommonTokenizer.encode.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.encode.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.MistralCommonTokenizer.encode.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MistralCommonTokenizer.encode.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MistralCommonTokenizer.encode.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MistralCommonTokenizer.encode.*kwargs",description:`*<strong>*kwargs</strong> &#x2014; Not supported by <code>MistralCommonTokenizer.encode</code>.
Will raise an error if used.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L367",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The tokenized ids of the text.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code>, <code>torch.Tensor</code></p>
`}}),Ne=new C({props:{name:"from_pretrained",anchor:"transformers.MistralCommonTokenizer.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:": typing.Union[str, os.PathLike]"},{name:"*init_inputs",val:""},{name:"mode",val:": ValidationMode = <ValidationMode.test: 'test'>"},{name:"cache_dir",val:": typing.Union[str, os.PathLike, NoneType] = None"},{name:"force_download",val:": bool = False"},{name:"local_files_only",val:": bool = False"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"revision",val:": str = 'main'"},{name:"model_max_length",val:": int = 1000000000000000019884624838656"},{name:"padding_side",val:": str = 'left'"},{name:"truncation_side",val:": str = 'right'"},{name:"model_input_names",val:": typing.Optional[list[str]] = None"},{name:"clean_up_tokenization_spaces",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.from_pretrained.pretrained_model_name_or_path",description:`<strong>pretrained_model_name_or_path</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
Can be either:</p>
<ul>
<li>A string, the <em>model id</em> of a predefined tokenizer hosted inside a model repo on huggingface.co.</li>
<li>A path to a <em>directory</em> containing the tokenizer config, for instance saved
using the <code>MistralCommonTokenizer.tokenization_mistral_common.save_pretrained</code> method, e.g.,
<code>./my_model_directory/</code>.</li>
</ul>`,name:"pretrained_model_name_or_path"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.mode",description:`<strong>mode</strong> (<code>ValidationMode</code>, <em>optional</em>, defaults to <code>ValidationMode.test</code>) &#x2014;
Validation mode for the <code>MistralTokenizer</code> tokenizer.`,name:"mode"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.cache_dir",description:`<strong>cache_dir</strong> (<code>str</code> or <code>os.PathLike</code>, <em>optional</em>) &#x2014;
Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
standard cache should not be used.`,name:"cache_dir"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.force_download",description:`<strong>force_download</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
exist.`,name:"force_download"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.token",description:`<strong>token</strong> (<code>str</code> or <em>bool</em>, <em>optional</em>) &#x2014;
The token to use as HTTP bearer authorization for remote files. If <code>True</code>, will use the token generated
when running <code>hf auth login</code> (stored in <code>~/.huggingface</code>).`,name:"token"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.local_files_only",description:`<strong>local_files_only</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to only rely on local files and not to attempt to download any files.`,name:"local_files_only"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.revision",description:`<strong>revision</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;main&quot;</code>) &#x2014;
The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
git-based system for storing models and other artifacts on huggingface.co, so <code>revision</code> can be any
identifier allowed by git.`,name:"revision"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;left&quot;</code>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.truncation_side",description:`<strong>truncation_side</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;right&quot;</code>) &#x2014;
The side on which the model should have truncation applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].`,name:"truncation_side"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.model_input_names",description:`<strong>model_input_names</strong> (<code>List[string]</code>, <em>optional</em>) &#x2014;
The list of inputs accepted by the forward pass of the model (like <code>&quot;token_type_ids&quot;</code> or
<code>&quot;attention_mask&quot;</code>). Default value is picked from the class attribute of the same name.`,name:"model_input_names"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the model should cleanup the spaces that were added when splitting the input text during the
tokenization process.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.from_pretrained</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1689"}}),We=new C({props:{name:"get_special_tokens_mask",anchor:"transformers.MistralCommonTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": None = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of ids of the sequence.`,name:"token_ids_0"},{anchor:"transformers.MistralCommonTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer</code>. Kept to match the interface of <code>PreTrainedTokenizerBase</code>.`,name:"token_ids_1"},{anchor:"transformers.MistralCommonTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L746",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]</p>
`}}),Ge=new C({props:{name:"get_vocab",anchor:"transformers.MistralCommonTokenizer.get_vocab",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L345",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Dict[str, int]</code></p>
`}}),qe=new C({props:{name:"pad",anchor:"transformers.MistralCommonTokenizer.pad",parameters:[{name:"encoded_inputs",val:": typing.Union[transformers.tokenization_utils_base.BatchEncoding, list[transformers.tokenization_utils_base.BatchEncoding], dict[str, list[int]], dict[str, list[list[int]]], list[dict[str, list[int]]]]"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"verbose",val:": bool = True"}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.pad.encoded_inputs",description:`<strong>encoded_inputs</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, list of <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, <code>Dict[str, List[int]]</code>, <code>Dict[str, List[List[int]]</code> or <code>List[Dict[str, List[int]]]</code>) &#x2014;
Tokenized inputs. Can represent one input (<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a> or <code>Dict[str, List[int]]</code>) or a batch of
tokenized inputs (list of <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, <em>Dict[str, List[List[int]]]</em> or <em>List[Dict[str,
List[int]]]</em>) so you can use this method during preprocessing as well as in a PyTorch Dataloader
collate function.</p>
<p>Instead of <code>List[int]</code> you can have tensors (numpy arrays, PyTorch tensors), see
the note above for the return type.`,name:"encoded_inputs"},{anchor:"transformers.MistralCommonTokenizer.pad.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code> (default): Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code>: No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MistralCommonTokenizer.pad.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length of the returned list and optionally padding length (see above).`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.pad.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value.</p>
<p>This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MistralCommonTokenizer.pad.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MistralCommonTokenizer.pad.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.MistralCommonTokenizer.pad.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MistralCommonTokenizer.pad.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1130"}}),ne=new zo({props:{$$slots:{default:[Ks]},$$scope:{ctx:U}}}),Qe=new C({props:{name:"prepare_for_model",anchor:"transformers.MistralCommonTokenizer.prepare_for_model",parameters:[{name:"ids",val:": list"},{name:"pair_ids",val:": None = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"prepend_batch_axis",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.ids",description:`<strong>ids</strong> (<code>List[int]</code>) &#x2014;
Tokenized input ids of the first sequence.`,name:"ids"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.pair_ids",description:`<strong>pair_ids</strong> (<code>None</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer</code>. Kept to match the interface of <code>PreTrainedTokenizerBase</code>.`,name:"pair_ids"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L842",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>overflowing_tokens</strong> — List of overflowing tokens sequences (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>num_truncated_tokens</strong> — Number of tokens truncated (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>special_tokens_mask</strong> — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when <code>add_special_tokens=True</code> and <code>return_special_tokens_mask=True</code>).</p>
</li>
<li>
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>)</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),Re=new C({props:{name:"save_pretrained",anchor:"transformers.MistralCommonTokenizer.save_pretrained",parameters:[{name:"save_directory",val:": typing.Union[str, os.PathLike, pathlib.Path]"},{name:"push_to_hub",val:": bool = False"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"commit_message",val:": typing.Optional[str] = None"},{name:"repo_id",val:": typing.Optional[str] = None"},{name:"private",val:": typing.Optional[bool] = None"},{name:"repo_url",val:": typing.Optional[str] = None"},{name:"organization",val:": typing.Optional[str] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.save_pretrained.save_directory",description:"<strong>save_directory</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014; The path to a directory where the tokenizer will be saved.",name:"save_directory"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.push_to_hub",description:`<strong>push_to_hub</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with <code>repo_id</code> (will default to the name of <code>save_directory</code> in your
namespace).`,name:"push_to_hub"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.token",description:`<strong>token</strong> (<code>str</code> or <em>bool</em>, <em>optional</em>, defaults to <code>None</code>) &#x2014;
The token to use to push to the model hub. If <code>True</code>, will use the token in the <code>HF_TOKEN</code> environment
variable.`,name:"token"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.commit_message",description:"<strong>commit_message</strong> (<code>str</code>, <em>optional</em>) &#x2014; The commit message to use when pushing to the hub.",name:"commit_message"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.repo_id",description:"<strong>repo_id</strong> (<code>str</code>, <em>optional</em>) &#x2014; The name of the repository to which push to the Hub.",name:"repo_id"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.private",description:"<strong>private</strong> (<code>bool</code>, <em>optional</em>) &#x2014; Whether the model repository is private or not.",name:"private"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.repo_url",description:"<strong>repo_url</strong> (<code>str</code>, <em>optional</em>) &#x2014; The URL to the Git repository to which push to the Hub.",name:"repo_url"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.organization",description:"<strong>organization</strong> (<code>str</code>, <em>optional</em>) &#x2014; The name of the organization in which you would like to push your model.",name:"organization"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.kwargs",description:`<strong>kwargs</strong> (<code>Dict[str, Any]</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.save_pretrained</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1816",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The files saved.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A tuple of <code>str</code></p>
`}}),Fe=new C({props:{name:"tokenize",anchor:"transformers.MistralCommonTokenizer.tokenize",parameters:[{name:"text",val:": str"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.tokenize.text",description:`<strong>text</strong> (<code>str</code>) &#x2014;
The sequence to be encoded.`,name:"text"},{anchor:"transformers.MistralCommonTokenizer.tokenize.*kwargs",description:`*<strong>*kwargs</strong> (additional keyword arguments) &#x2014;
Not supported by <code>MistralCommonTokenizer.tokenize</code>.
Will raise an error if used.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L606",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[str]</code></p>
`}}),Ee=new C({props:{name:"truncate_sequences",anchor:"transformers.MistralCommonTokenizer.truncate_sequences",parameters:[{name:"ids",val:": list"},{name:"pair_ids",val:": None = None"},{name:"num_tokens_to_remove",val:": int = 0"},{name:"truncation_strategy",val:": typing.Union[str, transformers.tokenization_utils_base.TruncationStrategy] = 'longest_first'"},{name:"stride",val:": int = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.ids",description:`<strong>ids</strong> (<code>List[int]</code>) &#x2014;
Tokenized input ids. Can be obtained from a string by chaining the <code>tokenize</code> and
<code>convert_tokens_to_ids</code> methods.`,name:"ids"},{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.pair_ids",description:`<strong>pair_ids</strong> (<code>None</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer</code>. Kept to match the signature of <code>PreTrainedTokenizerBase.truncate_sequences</code>.`,name:"pair_ids"},{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.num_tokens_to_remove",description:`<strong>num_tokens_to_remove</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Number of tokens to remove using the truncation strategy.`,name:"num_tokens_to_remove"},{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.truncation_strategy",description:`<strong>truncation_strategy</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>&apos;longest_first&apos;</code>) &#x2014;
The strategy to follow for truncation. Can be:</p>
<ul>
<li><code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided.</li>
<li><code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths greater
than the model maximum admissible input size).</li>
</ul>`,name:"truncation_strategy"},{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a positive number, the overflowing tokens returned will contain some tokens from the main
sequence returned. The value of this argument defines the number of additional tokens.`,name:"stride"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1293",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The truncated <code>ids</code> and the list of
overflowing tokens. <code>None</code> is returned to match Transformers signature.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Tuple[List[int], None, List[int]]</code></p>
`}}),He=new Yt({props:{title:"Mistral3Model",local:"transformers.Mistral3Model",headingTag:"h2"}}),Ae=new C({props:{name:"class transformers.Mistral3Model",anchor:"transformers.Mistral3Model",parameters:[{name:"config",val:": Mistral3Config"}],parametersDescription:[{anchor:"transformers.Mistral3Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Config">Mistral3Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral3/modeling_mistral3.py#L199"}}),Se=new C({props:{name:"forward",anchor:"transformers.Mistral3Model.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"pixel_values",val:": FloatTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"vision_feature_layer",val:": typing.Union[list[int], int, NoneType] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"image_sizes",val:": Tensor = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}],parametersDescription:[{anchor:"transformers.Mistral3Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Mistral3Model.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor">PixtralImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">PixtralImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralProcessor">PixtralProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor">PixtralImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.Mistral3Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Mistral3Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Mistral3Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Mistral3Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Mistral3Model.forward.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>Union[list[int], int, NoneType]</code>) &#x2014;
The index of the layer to select the vision feature. If multiple indices are provided,
the vision feature of the corresponding indices will be concatenated to form the
vision features.`,name:"vision_feature_layer"},{anchor:"transformers.Mistral3Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Mistral3Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Mistral3Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Mistral3Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Mistral3Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Mistral3Model.forward.image_sizes",description:`<strong>image_sizes</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, 2)</code>) &#x2014;
The sizes of the images in the batch, being (height, width) for each image.`,name:"image_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral3/modeling_mistral3.py#L289",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.mistral3.modeling_mistral3.Mistral3ModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Config"
>Mistral3Config</a>) and inputs.</p>
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


<p><code>transformers.models.mistral3.modeling_mistral3.Mistral3ModelOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),re=new zo({props:{$$slots:{default:[ea]},$$scope:{ctx:U}}}),Xe=new C({props:{name:"get_image_features",anchor:"transformers.Mistral3Model.get_image_features",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"image_sizes",val:": Tensor"},{name:"vision_feature_layer",val:": typing.Union[list[int], int, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Mistral3Model.get_image_features.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor]</code> of shape <code>(batch_size, channels, height, width)</code>) &#x2014;
The tensors corresponding to the input images.`,name:"pixel_values"},{anchor:"transformers.Mistral3Model.get_image_features.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>Union[int, list[int]]</code>, <em>optional</em>) &#x2014;
The index of the layer to select the vision feature. If multiple indices are provided,
the vision feature of the corresponding indices will be concatenated to form the
vision features.`,name:"vision_feature_layer"},{anchor:"transformers.Mistral3Model.get_image_features.image_sizes",description:`<strong>image_sizes</strong> (<code>torch.Tensor</code>, <em>optional</em>) &#x2014;
Tensor containing the image sizes as returned by the processor.`,name:"image_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral3/modeling_mistral3.py#L222",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Image feature tensor of shape <code>(num_images, image_length, embed_dim)</code>).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>image_features (<code>torch.Tensor</code>)</p>
`}}),De=new C({props:{name:"get_placeholder_mask",anchor:"transformers.Mistral3Model.get_placeholder_mask",parameters:[{name:"input_ids",val:": LongTensor"},{name:"inputs_embeds",val:": FloatTensor"},{name:"image_features",val:": FloatTensor"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral3/modeling_mistral3.py#L265"}}),Ye=new Yt({props:{title:"Mistral3ForConditionalGeneration",local:"transformers.Mistral3ForConditionalGeneration",headingTag:"h2"}}),Le=new C({props:{name:"class transformers.Mistral3ForConditionalGeneration",anchor:"transformers.Mistral3ForConditionalGeneration",parameters:[{name:"config",val:": Mistral3Config"}],parametersDescription:[{anchor:"transformers.Mistral3ForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Config">Mistral3Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral3/modeling_mistral3.py#L362"}}),Pe=new C({props:{name:"forward",anchor:"transformers.Mistral3ForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"pixel_values",val:": FloatTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"image_sizes",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Mistral3ForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor">PixtralImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">PixtralImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralProcessor">PixtralProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.PixtralImageProcessor">PixtralImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"},{anchor:"transformers.Mistral3ForConditionalGeneration.forward.image_sizes",description:`<strong>image_sizes</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, 2)</code>, <em>optional</em>) &#x2014;
The sizes of the images in the batch, being (height, width) for each image.`,name:"image_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral3/modeling_mistral3.py#L419",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.mistral3.modeling_mistral3.Mistral3CausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mistral3#transformers.Mistral3Config"
>Mistral3Config</a>) and inputs.</p>
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


<p><code>transformers.models.mistral3.modeling_mistral3.Mistral3CausalLMOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),de=new zo({props:{$$slots:{default:[ta]},$$scope:{ctx:U}}}),ce=new En({props:{anchor:"transformers.Mistral3ForConditionalGeneration.forward.example",$$slots:{default:[oa]},$$scope:{ctx:U}}}),Oe=new As({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mistral3.md"}}),{c(){s=a("meta"),J=o(),d=a("p"),w=o(),b=a("p"),b.innerHTML=c,I=o(),Z=a("div"),Z.innerHTML=Hn,Pt=o(),f(ue.$$.fragment),Ot=o(),he=a("p"),he.innerHTML=An,Kt=o(),fe=a("p"),fe.innerHTML=Sn,eo=o(),f(H.$$.fragment),to=o(),ge=a("p"),ge.innerHTML=Xn,oo=o(),f(A.$$.fragment),no=o(),f(Me.$$.fragment),so=o(),_e=a("ul"),_e.innerHTML=Dn,ao=o(),f(Te.$$.fragment),ro=o(),ye=a("p"),ye.textContent=Yn,io=o(),S=a("blockquote"),at=a("p"),at.textContent=Ln,Vo=o(),f(we.$$.fragment),lo=o(),be=a("ul"),be.innerHTML=Pn,co=o(),f(Je.$$.fragment),mo=o(),ve=a("ul"),ve.innerHTML=On,po=o(),f(ke.$$.fragment),uo=o(),f(Ue.$$.fragment),ho=o(),V=a("div"),f(Ce.$$.fragment),$o=o(),rt=a("p"),rt.innerHTML=Kn,Bo=o(),it=a("p"),it.innerHTML=es,Zo=o(),f(X.$$.fragment),fo=o(),f(Ie.$$.fragment),go=o(),p=a("div"),f(xe.$$.fragment),No=o(),lt=a("p"),lt.innerHTML=ts,Wo=o(),f(D.$$.fragment),Go=o(),dt=a("p"),dt.textContent=os,qo=o(),ct=a("p"),ct.innerHTML=ns,Qo=o(),mt=a("p"),mt.innerHTML=ss,Ro=o(),pt=a("p"),pt.innerHTML=as,Fo=o(),ut=a("ul"),ut.innerHTML=rs,Eo=o(),ht=a("p"),ht.innerHTML=is,Ho=o(),ft=a("ul"),ft.innerHTML=ls,Ao=o(),gt=a("p"),gt.innerHTML=ds,So=o(),Y=a("div"),f(je.$$.fragment),Xo=o(),Mt=a("p"),Mt.innerHTML=cs,Do=o(),L=a("div"),f(ze.$$.fragment),Yo=o(),_t=a("p"),_t.textContent=ms,Lo=o(),P=a("div"),f(Ve.$$.fragment),Po=o(),Tt=a("p"),Tt.textContent=ps,Oo=o(),O=a("div"),f($e.$$.fragment),Ko=o(),yt=a("p"),yt.textContent=us,en=o(),K=a("div"),f(Be.$$.fragment),tn=o(),wt=a("p"),wt.textContent=hs,on=o(),ee=a("div"),f(Ze.$$.fragment),nn=o(),bt=a("p"),bt.textContent=fs,sn=o(),te=a("div"),f(Ne.$$.fragment),an=o(),Jt=a("p"),Jt.innerHTML=gs,rn=o(),oe=a("div"),f(We.$$.fragment),ln=o(),vt=a("p"),vt.innerHTML=Ms,dn=o(),W=a("div"),f(Ge.$$.fragment),cn=o(),kt=a("p"),kt.textContent=_s,mn=o(),Ut=a("p"),Ut.textContent=Ts,pn=o(),$=a("div"),f(qe.$$.fragment),un=o(),Ct=a("p"),Ct.textContent=ys,hn=o(),It=a("p"),It.innerHTML=ws,fn=o(),f(ne.$$.fragment),gn=o(),se=a("div"),f(Qe.$$.fragment),Mn=o(),xt=a("p"),xt.textContent=bs,_n=o(),G=a("div"),f(Re.$$.fragment),Tn=o(),jt=a("p"),jt.textContent=Js,yn=o(),zt=a("p"),zt.innerHTML=vs,wn=o(),q=a("div"),f(Fe.$$.fragment),bn=o(),Vt=a("p"),Vt.textContent=ks,Jn=o(),$t=a("p"),$t.textContent=Us,vn=o(),ae=a("div"),f(Ee.$$.fragment),kn=o(),Bt=a("p"),Bt.textContent=Cs,Mo=o(),f(He.$$.fragment),_o=o(),x=a("div"),f(Ae.$$.fragment),Un=o(),Zt=a("p"),Zt.textContent=Is,Cn=o(),Nt=a("p"),Nt.innerHTML=xs,In=o(),Wt=a("p"),Wt.innerHTML=js,xn=o(),Q=a("div"),f(Se.$$.fragment),jn=o(),Gt=a("p"),Gt.innerHTML=zs,zn=o(),f(re.$$.fragment),Vn=o(),ie=a("div"),f(Xe.$$.fragment),$n=o(),qt=a("p"),qt.textContent=Vs,Bn=o(),le=a("div"),f(De.$$.fragment),Zn=o(),Qt=a("p"),Qt.innerHTML=$s,To=o(),f(Ye.$$.fragment),yo=o(),j=a("div"),f(Le.$$.fragment),Nn=o(),Rt=a("p"),Rt.textContent=Bs,Wn=o(),Ft=a("p"),Ft.innerHTML=Zs,Gn=o(),Et=a("p"),Et.innerHTML=Ns,qn=o(),B=a("div"),f(Pe.$$.fragment),Qn=o(),Ht=a("p"),Ht.innerHTML=Ws,Rn=o(),f(de.$$.fragment),Fn=o(),f(ce.$$.fragment),wo=o(),f(Oe.$$.fragment),bo=o(),Lt=a("p"),this.h()},l(e){const l=Es("svelte-u9bgzb",document.head);s=r(l,"META",{name:!0,content:!0}),l.forEach(i),J=n(e),d=r(e,"P",{}),v(d).forEach(i),w=n(e),b=r(e,"P",{"data-svelte-h":!0}),m(b)!=="svelte-1js7828"&&(b.innerHTML=c),I=n(e),Z=r(e,"DIV",{style:!0,"data-svelte-h":!0}),m(Z)!=="svelte-383xsf"&&(Z.innerHTML=Hn),Pt=n(e),g(ue.$$.fragment,e),Ot=n(e),he=r(e,"P",{"data-svelte-h":!0}),m(he)!=="svelte-fhm1l6"&&(he.innerHTML=An),Kt=n(e),fe=r(e,"P",{"data-svelte-h":!0}),m(fe)!=="svelte-bt03jg"&&(fe.innerHTML=Sn),eo=n(e),g(H.$$.fragment,e),to=n(e),ge=r(e,"P",{"data-svelte-h":!0}),m(ge)!=="svelte-1an17cp"&&(ge.innerHTML=Xn),oo=n(e),g(A.$$.fragment,e),no=n(e),g(Me.$$.fragment,e),so=n(e),_e=r(e,"UL",{"data-svelte-h":!0}),m(_e)!=="svelte-1e694z4"&&(_e.innerHTML=Dn),ao=n(e),g(Te.$$.fragment,e),ro=n(e),ye=r(e,"P",{"data-svelte-h":!0}),m(ye)!=="svelte-1lcujg2"&&(ye.textContent=Yn),io=n(e),S=r(e,"BLOCKQUOTE",{});var Ke=v(S);at=r(Ke,"P",{"data-svelte-h":!0}),m(at)!=="svelte-1olxqa4"&&(at.textContent=Ln),Vo=n(Ke),g(we.$$.fragment,Ke),Ke.forEach(i),lo=n(e),be=r(e,"UL",{"data-svelte-h":!0}),m(be)!=="svelte-cz098e"&&(be.innerHTML=Pn),co=n(e),g(Je.$$.fragment,e),mo=n(e),ve=r(e,"UL",{"data-svelte-h":!0}),m(ve)!=="svelte-xakft5"&&(ve.innerHTML=On),po=n(e),g(ke.$$.fragment,e),uo=n(e),g(Ue.$$.fragment,e),ho=n(e),V=r(e,"DIV",{class:!0});var N=v(V);g(Ce.$$.fragment,N),$o=n(N),rt=r(N,"P",{"data-svelte-h":!0}),m(rt)!=="svelte-115y9ah"&&(rt.innerHTML=Kn),Bo=n(N),it=r(N,"P",{"data-svelte-h":!0}),m(it)!=="svelte-1ek1ss9"&&(it.innerHTML=es),Zo=n(N),g(X.$$.fragment,N),N.forEach(i),fo=n(e),g(Ie.$$.fragment,e),go=n(e),p=r(e,"DIV",{class:!0});var h=v(p);g(xe.$$.fragment,h),No=n(h),lt=r(h,"P",{"data-svelte-h":!0}),m(lt)!=="svelte-iuk2y8"&&(lt.innerHTML=ts),Wo=n(h),g(D.$$.fragment,h),Go=n(h),dt=r(h,"P",{"data-svelte-h":!0}),m(dt)!=="svelte-kud278"&&(dt.textContent=os),qo=n(h),ct=r(h,"P",{"data-svelte-h":!0}),m(ct)!=="svelte-ifzpy9"&&(ct.innerHTML=ns),Qo=n(h),mt=r(h,"P",{"data-svelte-h":!0}),m(mt)!=="svelte-ktmcb2"&&(mt.innerHTML=ss),Ro=n(h),pt=r(h,"P",{"data-svelte-h":!0}),m(pt)!=="svelte-mzof2m"&&(pt.innerHTML=as),Fo=n(h),ut=r(h,"UL",{"data-svelte-h":!0}),m(ut)!=="svelte-1hlq74o"&&(ut.innerHTML=rs),Eo=n(h),ht=r(h,"P",{"data-svelte-h":!0}),m(ht)!=="svelte-k8piyc"&&(ht.innerHTML=is),Ho=n(h),ft=r(h,"UL",{"data-svelte-h":!0}),m(ft)!=="svelte-mjbefh"&&(ft.innerHTML=ls),Ao=n(h),gt=r(h,"P",{"data-svelte-h":!0}),m(gt)!=="svelte-18hne1"&&(gt.innerHTML=ds),So=n(h),Y=r(h,"DIV",{class:!0});var et=v(Y);g(je.$$.fragment,et),Xo=n(et),Mt=r(et,"P",{"data-svelte-h":!0}),m(Mt)!=="svelte-sr2voc"&&(Mt.innerHTML=cs),et.forEach(i),Do=n(h),L=r(h,"DIV",{class:!0});var tt=v(L);g(ze.$$.fragment,tt),Yo=n(tt),_t=r(tt,"P",{"data-svelte-h":!0}),m(_t)!=="svelte-1deng2j"&&(_t.textContent=ms),tt.forEach(i),Lo=n(h),P=r(h,"DIV",{class:!0});var ot=v(P);g(Ve.$$.fragment,ot),Po=n(ot),Tt=r(ot,"P",{"data-svelte-h":!0}),m(Tt)!=="svelte-cx157h"&&(Tt.textContent=ps),ot.forEach(i),Oo=n(h),O=r(h,"DIV",{class:!0});var nt=v(O);g($e.$$.fragment,nt),Ko=n(nt),yt=r(nt,"P",{"data-svelte-h":!0}),m(yt)!=="svelte-1urz5jj"&&(yt.textContent=us),nt.forEach(i),en=n(h),K=r(h,"DIV",{class:!0});var st=v(K);g(Be.$$.fragment,st),tn=n(st),wt=r(st,"P",{"data-svelte-h":!0}),m(wt)!=="svelte-vbfkpu"&&(wt.textContent=hs),st.forEach(i),on=n(h),ee=r(h,"DIV",{class:!0});var vo=v(ee);g(Ze.$$.fragment,vo),nn=n(vo),bt=r(vo,"P",{"data-svelte-h":!0}),m(bt)!=="svelte-12b8hzo"&&(bt.textContent=fs),vo.forEach(i),sn=n(h),te=r(h,"DIV",{class:!0});var ko=v(te);g(Ne.$$.fragment,ko),an=n(ko),Jt=r(ko,"P",{"data-svelte-h":!0}),m(Jt)!=="svelte-5j01oy"&&(Jt.innerHTML=gs),ko.forEach(i),rn=n(h),oe=r(h,"DIV",{class:!0});var Uo=v(oe);g(We.$$.fragment,Uo),ln=n(Uo),vt=r(Uo,"P",{"data-svelte-h":!0}),m(vt)!=="svelte-1wmjg8a"&&(vt.innerHTML=Ms),Uo.forEach(i),dn=n(h),W=r(h,"DIV",{class:!0});var At=v(W);g(Ge.$$.fragment,At),cn=n(At),kt=r(At,"P",{"data-svelte-h":!0}),m(kt)!=="svelte-1gbatu6"&&(kt.textContent=_s),mn=n(At),Ut=r(At,"P",{"data-svelte-h":!0}),m(Ut)!=="svelte-1d4v47d"&&(Ut.textContent=Ts),At.forEach(i),pn=n(h),$=r(h,"DIV",{class:!0});var me=v($);g(qe.$$.fragment,me),un=n(me),Ct=r(me,"P",{"data-svelte-h":!0}),m(Ct)!=="svelte-1n892mi"&&(Ct.textContent=ys),hn=n(me),It=r(me,"P",{"data-svelte-h":!0}),m(It)!=="svelte-954lq4"&&(It.innerHTML=ws),fn=n(me),g(ne.$$.fragment,me),me.forEach(i),gn=n(h),se=r(h,"DIV",{class:!0});var Co=v(se);g(Qe.$$.fragment,Co),Mn=n(Co),xt=r(Co,"P",{"data-svelte-h":!0}),m(xt)!=="svelte-15kr77e"&&(xt.textContent=bs),Co.forEach(i),_n=n(h),G=r(h,"DIV",{class:!0});var St=v(G);g(Re.$$.fragment,St),Tn=n(St),jt=r(St,"P",{"data-svelte-h":!0}),m(jt)!=="svelte-u73u19"&&(jt.textContent=Js),yn=n(St),zt=r(St,"P",{"data-svelte-h":!0}),m(zt)!=="svelte-oagoqu"&&(zt.innerHTML=vs),St.forEach(i),wn=n(h),q=r(h,"DIV",{class:!0});var Xt=v(q);g(Fe.$$.fragment,Xt),bn=n(Xt),Vt=r(Xt,"P",{"data-svelte-h":!0}),m(Vt)!=="svelte-sso1qb"&&(Vt.textContent=ks),Jn=n(Xt),$t=r(Xt,"P",{"data-svelte-h":!0}),m($t)!=="svelte-46tdba"&&($t.textContent=Us),Xt.forEach(i),vn=n(h),ae=r(h,"DIV",{class:!0});var Io=v(ae);g(Ee.$$.fragment,Io),kn=n(Io),Bt=r(Io,"P",{"data-svelte-h":!0}),m(Bt)!=="svelte-fkofn"&&(Bt.textContent=Cs),Io.forEach(i),h.forEach(i),Mo=n(e),g(He.$$.fragment,e),_o=n(e),x=r(e,"DIV",{class:!0});var z=v(x);g(Ae.$$.fragment,z),Un=n(z),Zt=r(z,"P",{"data-svelte-h":!0}),m(Zt)!=="svelte-1vonyvi"&&(Zt.textContent=Is),Cn=n(z),Nt=r(z,"P",{"data-svelte-h":!0}),m(Nt)!=="svelte-q52n56"&&(Nt.innerHTML=xs),In=n(z),Wt=r(z,"P",{"data-svelte-h":!0}),m(Wt)!=="svelte-hswkmf"&&(Wt.innerHTML=js),xn=n(z),Q=r(z,"DIV",{class:!0});var Dt=v(Q);g(Se.$$.fragment,Dt),jn=n(Dt),Gt=r(Dt,"P",{"data-svelte-h":!0}),m(Gt)!=="svelte-49ri8m"&&(Gt.innerHTML=zs),zn=n(Dt),g(re.$$.fragment,Dt),Dt.forEach(i),Vn=n(z),ie=r(z,"DIV",{class:!0});var xo=v(ie);g(Xe.$$.fragment,xo),$n=n(xo),qt=r(xo,"P",{"data-svelte-h":!0}),m(qt)!=="svelte-1vzo9k5"&&(qt.textContent=Vs),xo.forEach(i),Bn=n(z),le=r(z,"DIV",{class:!0});var jo=v(le);g(De.$$.fragment,jo),Zn=n(jo),Qt=r(jo,"P",{"data-svelte-h":!0}),m(Qt)!=="svelte-3ue1dv"&&(Qt.innerHTML=$s),jo.forEach(i),z.forEach(i),To=n(e),g(Ye.$$.fragment,e),yo=n(e),j=r(e,"DIV",{class:!0});var R=v(j);g(Le.$$.fragment,R),Nn=n(R),Rt=r(R,"P",{"data-svelte-h":!0}),m(Rt)!=="svelte-1n8yi42"&&(Rt.textContent=Bs),Wn=n(R),Ft=r(R,"P",{"data-svelte-h":!0}),m(Ft)!=="svelte-q52n56"&&(Ft.innerHTML=Zs),Gn=n(R),Et=r(R,"P",{"data-svelte-h":!0}),m(Et)!=="svelte-hswkmf"&&(Et.innerHTML=Ns),qn=n(R),B=r(R,"DIV",{class:!0});var pe=v(B);g(Pe.$$.fragment,pe),Qn=n(pe),Ht=r(pe,"P",{"data-svelte-h":!0}),m(Ht)!=="svelte-15r9yle"&&(Ht.innerHTML=Ws),Rn=n(pe),g(de.$$.fragment,pe),Fn=n(pe),g(ce.$$.fragment,pe),pe.forEach(i),R.forEach(i),wo=n(e),g(Oe.$$.fragment,e),bo=n(e),Lt=r(e,"P",{}),v(Lt).forEach(i),this.h()},h(){k(s,"name","hf:doc:metadata"),k(s,"content",sa),Hs(Z,"float","right"),k(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(p,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,l){t(document.head,s),u(e,J,l),u(e,d,l),u(e,w,l),u(e,b,l),u(e,I,l),u(e,Z,l),u(e,Pt,l),M(ue,e,l),u(e,Ot,l),u(e,he,l),u(e,Kt,l),u(e,fe,l),u(e,eo,l),M(H,e,l),u(e,to,l),u(e,ge,l),u(e,oo,l),M(A,e,l),u(e,no,l),M(Me,e,l),u(e,so,l),u(e,_e,l),u(e,ao,l),M(Te,e,l),u(e,ro,l),u(e,ye,l),u(e,io,l),u(e,S,l),t(S,at),t(S,Vo),M(we,S,null),u(e,lo,l),u(e,be,l),u(e,co,l),M(Je,e,l),u(e,mo,l),u(e,ve,l),u(e,po,l),M(ke,e,l),u(e,uo,l),M(Ue,e,l),u(e,ho,l),u(e,V,l),M(Ce,V,null),t(V,$o),t(V,rt),t(V,Bo),t(V,it),t(V,Zo),M(X,V,null),u(e,fo,l),M(Ie,e,l),u(e,go,l),u(e,p,l),M(xe,p,null),t(p,No),t(p,lt),t(p,Wo),M(D,p,null),t(p,Go),t(p,dt),t(p,qo),t(p,ct),t(p,Qo),t(p,mt),t(p,Ro),t(p,pt),t(p,Fo),t(p,ut),t(p,Eo),t(p,ht),t(p,Ho),t(p,ft),t(p,Ao),t(p,gt),t(p,So),t(p,Y),M(je,Y,null),t(Y,Xo),t(Y,Mt),t(p,Do),t(p,L),M(ze,L,null),t(L,Yo),t(L,_t),t(p,Lo),t(p,P),M(Ve,P,null),t(P,Po),t(P,Tt),t(p,Oo),t(p,O),M($e,O,null),t(O,Ko),t(O,yt),t(p,en),t(p,K),M(Be,K,null),t(K,tn),t(K,wt),t(p,on),t(p,ee),M(Ze,ee,null),t(ee,nn),t(ee,bt),t(p,sn),t(p,te),M(Ne,te,null),t(te,an),t(te,Jt),t(p,rn),t(p,oe),M(We,oe,null),t(oe,ln),t(oe,vt),t(p,dn),t(p,W),M(Ge,W,null),t(W,cn),t(W,kt),t(W,mn),t(W,Ut),t(p,pn),t(p,$),M(qe,$,null),t($,un),t($,Ct),t($,hn),t($,It),t($,fn),M(ne,$,null),t(p,gn),t(p,se),M(Qe,se,null),t(se,Mn),t(se,xt),t(p,_n),t(p,G),M(Re,G,null),t(G,Tn),t(G,jt),t(G,yn),t(G,zt),t(p,wn),t(p,q),M(Fe,q,null),t(q,bn),t(q,Vt),t(q,Jn),t(q,$t),t(p,vn),t(p,ae),M(Ee,ae,null),t(ae,kn),t(ae,Bt),u(e,Mo,l),M(He,e,l),u(e,_o,l),u(e,x,l),M(Ae,x,null),t(x,Un),t(x,Zt),t(x,Cn),t(x,Nt),t(x,In),t(x,Wt),t(x,xn),t(x,Q),M(Se,Q,null),t(Q,jn),t(Q,Gt),t(Q,zn),M(re,Q,null),t(x,Vn),t(x,ie),M(Xe,ie,null),t(ie,$n),t(ie,qt),t(x,Bn),t(x,le),M(De,le,null),t(le,Zn),t(le,Qt),u(e,To,l),M(Ye,e,l),u(e,yo,l),u(e,j,l),M(Le,j,null),t(j,Nn),t(j,Rt),t(j,Wn),t(j,Ft),t(j,Gn),t(j,Et),t(j,qn),t(j,B),M(Pe,B,null),t(B,Qn),t(B,Ht),t(B,Rn),M(de,B,null),t(B,Fn),M(ce,B,null),u(e,wo,l),M(Oe,e,l),u(e,bo,l),u(e,Lt,l),Jo=!0},p(e,[l]){const Ke={};l&2&&(Ke.$$scope={dirty:l,ctx:e}),H.$set(Ke);const N={};l&2&&(N.$$scope={dirty:l,ctx:e}),A.$set(N);const h={};l&2&&(h.$$scope={dirty:l,ctx:e}),X.$set(h);const et={};l&2&&(et.$$scope={dirty:l,ctx:e}),D.$set(et);const tt={};l&2&&(tt.$$scope={dirty:l,ctx:e}),ne.$set(tt);const ot={};l&2&&(ot.$$scope={dirty:l,ctx:e}),re.$set(ot);const nt={};l&2&&(nt.$$scope={dirty:l,ctx:e}),de.$set(nt);const st={};l&2&&(st.$$scope={dirty:l,ctx:e}),ce.$set(st)},i(e){Jo||(_(ue.$$.fragment,e),_(H.$$.fragment,e),_(A.$$.fragment,e),_(Me.$$.fragment,e),_(Te.$$.fragment,e),_(we.$$.fragment,e),_(Je.$$.fragment,e),_(ke.$$.fragment,e),_(Ue.$$.fragment,e),_(Ce.$$.fragment,e),_(X.$$.fragment,e),_(Ie.$$.fragment,e),_(xe.$$.fragment,e),_(D.$$.fragment,e),_(je.$$.fragment,e),_(ze.$$.fragment,e),_(Ve.$$.fragment,e),_($e.$$.fragment,e),_(Be.$$.fragment,e),_(Ze.$$.fragment,e),_(Ne.$$.fragment,e),_(We.$$.fragment,e),_(Ge.$$.fragment,e),_(qe.$$.fragment,e),_(ne.$$.fragment,e),_(Qe.$$.fragment,e),_(Re.$$.fragment,e),_(Fe.$$.fragment,e),_(Ee.$$.fragment,e),_(He.$$.fragment,e),_(Ae.$$.fragment,e),_(Se.$$.fragment,e),_(re.$$.fragment,e),_(Xe.$$.fragment,e),_(De.$$.fragment,e),_(Ye.$$.fragment,e),_(Le.$$.fragment,e),_(Pe.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(Oe.$$.fragment,e),Jo=!0)},o(e){T(ue.$$.fragment,e),T(H.$$.fragment,e),T(A.$$.fragment,e),T(Me.$$.fragment,e),T(Te.$$.fragment,e),T(we.$$.fragment,e),T(Je.$$.fragment,e),T(ke.$$.fragment,e),T(Ue.$$.fragment,e),T(Ce.$$.fragment,e),T(X.$$.fragment,e),T(Ie.$$.fragment,e),T(xe.$$.fragment,e),T(D.$$.fragment,e),T(je.$$.fragment,e),T(ze.$$.fragment,e),T(Ve.$$.fragment,e),T($e.$$.fragment,e),T(Be.$$.fragment,e),T(Ze.$$.fragment,e),T(Ne.$$.fragment,e),T(We.$$.fragment,e),T(Ge.$$.fragment,e),T(qe.$$.fragment,e),T(ne.$$.fragment,e),T(Qe.$$.fragment,e),T(Re.$$.fragment,e),T(Fe.$$.fragment,e),T(Ee.$$.fragment,e),T(He.$$.fragment,e),T(Ae.$$.fragment,e),T(Se.$$.fragment,e),T(re.$$.fragment,e),T(Xe.$$.fragment,e),T(De.$$.fragment,e),T(Ye.$$.fragment,e),T(Le.$$.fragment,e),T(Pe.$$.fragment,e),T(de.$$.fragment,e),T(ce.$$.fragment,e),T(Oe.$$.fragment,e),Jo=!1},d(e){e&&(i(J),i(d),i(w),i(b),i(I),i(Z),i(Pt),i(Ot),i(he),i(Kt),i(fe),i(eo),i(to),i(ge),i(oo),i(no),i(so),i(_e),i(ao),i(ro),i(ye),i(io),i(S),i(lo),i(be),i(co),i(mo),i(ve),i(po),i(uo),i(ho),i(V),i(fo),i(go),i(p),i(Mo),i(_o),i(x),i(To),i(yo),i(j),i(wo),i(bo),i(Lt)),i(s),y(ue,e),y(H,e),y(A,e),y(Me,e),y(Te,e),y(we),y(Je,e),y(ke,e),y(Ue,e),y(Ce),y(X),y(Ie,e),y(xe),y(D),y(je),y(ze),y(Ve),y($e),y(Be),y(Ze),y(Ne),y(We),y(Ge),y(qe),y(ne),y(Qe),y(Re),y(Fe),y(Ee),y(He,e),y(Ae),y(Se),y(re),y(Xe),y(De),y(Ye,e),y(Le),y(Pe),y(de),y(ce),y(Oe,e)}}}const sa='{"title":"Mistral 3","local":"mistral-3","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Mistral3Config","local":"transformers.Mistral3Config","sections":[],"depth":2},{"title":"MistralCommonTokenizer","local":"transformers.MistralCommonTokenizer","sections":[],"depth":2},{"title":"Mistral3Model","local":"transformers.Mistral3Model","sections":[],"depth":2},{"title":"Mistral3ForConditionalGeneration","local":"transformers.Mistral3ForConditionalGeneration","sections":[],"depth":2}],"depth":1}';function aa(U){return Qs(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ha extends Rs{constructor(s){super(),Fs(this,s,aa,na,qs,{})}}export{ha as component};
