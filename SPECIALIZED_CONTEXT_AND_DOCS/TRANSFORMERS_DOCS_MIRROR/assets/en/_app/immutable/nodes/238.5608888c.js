import{s as Ho,z as Vo,o as So,n as Ne}from"../chunks/scheduler.18a86fab.js";import{S as Lo,i as Eo,g as d,s as n,r as h,A as Xo,h as c,f as o,c as s,j as x,x as g,u,k as w,y as l,a as r,v as f,d as _,t as T,w as y}from"../chunks/index.98837b22.js";import{T as po}from"../chunks/Tip.77304350.js";import{D as q}from"../chunks/Docstring.a1ef7999.js";import{C as Jt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ft}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{P as Qo}from"../chunks/PipelineTag.7749150e.js";import{H as R,E as Oo}from"../chunks/getInferenceSnippets.06c2775f.js";function Ao(M){let a,v="Example:",m,p,b;return p=new Jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEltYWdlR1BUQ29uZmlnJTJDJTIwSW1hZ2VHUFRNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBJbWFnZUdQVCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwSW1hZ2VHUFRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwSW1hZ2VHUFRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ImageGPTConfig, ImageGPTModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a ImageGPT configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = ImageGPTConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ImageGPTModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){a=d("p"),a.textContent=v,m=n(),h(p.$$.fragment)},l(i){a=c(i,"P",{"data-svelte-h":!0}),g(a)!=="svelte-11lpom8"&&(a.textContent=v),m=s(i),u(p.$$.fragment,i)},m(i,I){r(i,a,I),r(i,m,I),f(p,i,I),b=!0},p:Ne,i(i){b||(_(p.$$.fragment,i),b=!0)},o(i){T(p.$$.fragment,i),b=!1},d(i){i&&(o(a),o(m)),y(p,i)}}}function Do(M){let a,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=d("p"),a.innerHTML=v},l(m){a=c(m,"P",{"data-svelte-h":!0}),g(a)!=="svelte-fincs2"&&(a.innerHTML=v)},m(m,p){r(m,a,p)},p:Ne,d(m){m&&o(a)}}}function Yo(M){let a,v="Examples:",m,p,b;return p=new Jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMEltYWdlR1BUTW9kZWwlMEFmcm9tJTIwUElMJTIwaW1wb3J0JTIwSW1hZ2UlMEFpbXBvcnQlMjByZXF1ZXN0cyUwQSUwQXVybCUyMCUzRCUyMCUyMmh0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEElMEFpbWFnZV9wcm9jZXNzb3IlMjAlM0QlMjBBdXRvSW1hZ2VQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haSUyRmltYWdlZ3B0LXNtYWxsJTIyKSUwQW1vZGVsJTIwJTNEJTIwSW1hZ2VHUFRNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpJTJGaW1hZ2VncHQtc21hbGwlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwaW1hZ2VfcHJvY2Vzc29yKGltYWdlcyUzRGltYWdlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBbGFzdF9oaWRkZW5fc3RhdGVzJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, ImageGPTModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;openai/imagegpt-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ImageGPTModel.from_pretrained(<span class="hljs-string">&quot;openai/imagegpt-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){a=d("p"),a.textContent=v,m=n(),h(p.$$.fragment)},l(i){a=c(i,"P",{"data-svelte-h":!0}),g(a)!=="svelte-kvfsh7"&&(a.textContent=v),m=s(i),u(p.$$.fragment,i)},m(i,I){r(i,a,I),r(i,m,I),f(p,i,I),b=!0},p:Ne,i(i){b||(_(p.$$.fragment,i),b=!0)},o(i){T(p.$$.fragment,i),b=!1},d(i){i&&(o(a),o(m)),y(p,i)}}}function Ko(M){let a,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=d("p"),a.innerHTML=v},l(m){a=c(m,"P",{"data-svelte-h":!0}),g(a)!=="svelte-fincs2"&&(a.innerHTML=v)},m(m,p){r(m,a,p)},p:Ne,d(m){m&&o(a)}}}function en(M){let a,v="Examples:",m,p,b;return p=new Jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMEltYWdlR1BURm9yQ2F1c2FsSW1hZ2VNb2RlbGluZyUwQWltcG9ydCUyMHRvcmNoJTBBaW1wb3J0JTIwbWF0cGxvdGxpYi5weXBsb3QlMjBhcyUyMHBsdCUwQWltcG9ydCUyMG51bXB5JTIwYXMlMjBucCUwQSUwQWltYWdlX3Byb2Nlc3NvciUyMCUzRCUyMEF1dG9JbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpJTJGaW1hZ2VncHQtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBJbWFnZUdQVEZvckNhdXNhbEltYWdlTW9kZWxpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haSUyRmltYWdlZ3B0LXNtYWxsJTIyKSUwQWRldmljZSUyMCUzRCUyMHRvcmNoLmRldmljZSglMjJjdWRhJTIyJTIwaWYlMjB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpJTIwZWxzZSUyMCUyMmNwdSUyMiklMEFtb2RlbC50byhkZXZpY2UpJTBBJTIzJTIwdW5jb25kaXRpb25hbCUyMGdlbmVyYXRpb24lMjBvZiUyMDglMjBpbWFnZXMlMEFiYXRjaF9zaXplJTIwJTNEJTIwNCUwQWNvbnRleHQlMjAlM0QlMjB0b3JjaC5mdWxsKChiYXRjaF9zaXplJTJDJTIwMSklMkMlMjBtb2RlbC5jb25maWcudm9jYWJfc2l6ZSUyMC0lMjAxKSUyMCUyMCUyMyUyMGluaXRpYWxpemUlMjB3aXRoJTIwU09TJTIwdG9rZW4lMEFjb250ZXh0JTIwJTNEJTIwY29udGV4dC50byhkZXZpY2UpJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoJTBBJTIwJTIwJTIwJTIwaW5wdXRfaWRzJTNEY29udGV4dCUyQyUyMG1heF9sZW5ndGglM0Rtb2RlbC5jb25maWcubl9wb3NpdGlvbnMlMjAlMkIlMjAxJTJDJTIwdGVtcGVyYXR1cmUlM0QxLjAlMkMlMjBkb19zYW1wbGUlM0RUcnVlJTJDJTIwdG9wX2slM0Q0MCUwQSklMEElMEFjbHVzdGVycyUyMCUzRCUyMGltYWdlX3Byb2Nlc3Nvci5jbHVzdGVycyUwQWhlaWdodCUyMCUzRCUyMGltYWdlX3Byb2Nlc3Nvci5zaXplJTVCJTIyaGVpZ2h0JTIyJTVEJTBBd2lkdGglMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3Iuc2l6ZSU1QiUyMndpZHRoJTIyJTVEJTBBJTBBc2FtcGxlcyUyMCUzRCUyMG91dHB1dCU1QiUzQSUyQyUyMDElM0ElNUQuZGV0YWNoKCkuY3B1KCkubnVtcHkoKSUwQXNhbXBsZXNfaW1nJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwbnAucmVzaGFwZShucC5yaW50KDEyNy41JTIwKiUyMChjbHVzdGVycyU1QnMlNUQlMjAlMkIlMjAxLjApKSUyQyUyMCU1QmhlaWdodCUyQyUyMHdpZHRoJTJDJTIwMyU1RCkuYXN0eXBlKG5wLnVpbnQ4KSUyMGZvciUyMHMlMjBpbiUyMHNhbXBsZXMlMEElNUQlMjAlMjAlMjMlMjBjb252ZXJ0JTIwY29sb3IlMjBjbHVzdGVyJTIwdG9rZW5zJTIwYmFjayUyMHRvJTIwcGl4ZWxzJTBBZiUyQyUyMGF4ZXMlMjAlM0QlMjBwbHQuc3VicGxvdHMoMSUyQyUyMGJhdGNoX3NpemUlMkMlMjBkcGklM0QzMDApJTBBJTBBZm9yJTIwaW1nJTJDJTIwYXglMjBpbiUyMHppcChzYW1wbGVzX2ltZyUyQyUyMGF4ZXMpJTNBJTBBJTIwJTIwJTIwJTIwYXguYXhpcyglMjJvZmYlMjIpJTBBJTIwJTIwJTIwJTIwYXguaW1zaG93KGltZyk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, ImageGPTForCausalImageModeling
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> matplotlib.pyplot <span class="hljs-keyword">as</span> plt
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;openai/imagegpt-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ImageGPTForCausalImageModeling.from_pretrained(<span class="hljs-string">&quot;openai/imagegpt-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>device = torch.device(<span class="hljs-string">&quot;cuda&quot;</span> <span class="hljs-keyword">if</span> torch.cuda.is_available() <span class="hljs-keyword">else</span> <span class="hljs-string">&quot;cpu&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model.to(device)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># unconditional generation of 8 images</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>batch_size = <span class="hljs-number">4</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>context = torch.full((batch_size, <span class="hljs-number">1</span>), model.config.vocab_size - <span class="hljs-number">1</span>)  <span class="hljs-comment"># initialize with SOS token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>context = context.to(device)
<span class="hljs-meta">&gt;&gt;&gt; </span>output = model.generate(
<span class="hljs-meta">... </span>    input_ids=context, max_length=model.config.n_positions + <span class="hljs-number">1</span>, temperature=<span class="hljs-number">1.0</span>, do_sample=<span class="hljs-literal">True</span>, top_k=<span class="hljs-number">40</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>clusters = image_processor.clusters
<span class="hljs-meta">&gt;&gt;&gt; </span>height = image_processor.size[<span class="hljs-string">&quot;height&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>width = image_processor.size[<span class="hljs-string">&quot;width&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>samples = output[:, <span class="hljs-number">1</span>:].detach().cpu().numpy()
<span class="hljs-meta">&gt;&gt;&gt; </span>samples_img = [
<span class="hljs-meta">... </span>    np.reshape(np.rint(<span class="hljs-number">127.5</span> * (clusters[s] + <span class="hljs-number">1.0</span>)), [height, width, <span class="hljs-number">3</span>]).astype(np.uint8) <span class="hljs-keyword">for</span> s <span class="hljs-keyword">in</span> samples
<span class="hljs-meta">... </span>]  <span class="hljs-comment"># convert color cluster tokens back to pixels</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>f, axes = plt.subplots(<span class="hljs-number">1</span>, batch_size, dpi=<span class="hljs-number">300</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> img, ax <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(samples_img, axes):
<span class="hljs-meta">... </span>    ax.axis(<span class="hljs-string">&quot;off&quot;</span>)
<span class="hljs-meta">... </span>    ax.imshow(img)`,wrap:!1}}),{c(){a=d("p"),a.textContent=v,m=n(),h(p.$$.fragment)},l(i){a=c(i,"P",{"data-svelte-h":!0}),g(a)!=="svelte-kvfsh7"&&(a.textContent=v),m=s(i),u(p.$$.fragment,i)},m(i,I){r(i,a,I),r(i,m,I),f(p,i,I),b=!0},p:Ne,i(i){b||(_(p.$$.fragment,i),b=!0)},o(i){T(p.$$.fragment,i),b=!1},d(i){i&&(o(a),o(m)),y(p,i)}}}function tn(M){let a,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=d("p"),a.innerHTML=v},l(m){a=c(m,"P",{"data-svelte-h":!0}),g(a)!=="svelte-fincs2"&&(a.innerHTML=v)},m(m,p){r(m,a,p)},p:Ne,d(m){m&&o(a)}}}function on(M){let a,v="Examples:",m,p,b;return p=new Jt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMEltYWdlR1BURm9ySW1hZ2VDbGFzc2lmaWNhdGlvbiUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHJlcXVlc3RzJTBBJTBBdXJsJTIwJTNEJTIwJTIyaHR0cCUzQSUyRiUyRmltYWdlcy5jb2NvZGF0YXNldC5vcmclMkZ2YWwyMDE3JTJGMDAwMDAwMDM5NzY5LmpwZyUyMiUwQWltYWdlJTIwJTNEJTIwSW1hZ2Uub3BlbihyZXF1ZXN0cy5nZXQodXJsJTJDJTIwc3RyZWFtJTNEVHJ1ZSkucmF3KSUwQSUwQWltYWdlX3Byb2Nlc3NvciUyMCUzRCUyMEF1dG9JbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyb3BlbmFpJTJGaW1hZ2VncHQtc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBJbWFnZUdQVEZvckltYWdlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haSUyRmltYWdlZ3B0LXNtYWxsJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMGltYWdlX3Byb2Nlc3NvcihpbWFnZXMlM0RpbWFnZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, ImageGPTForImageClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;openai/imagegpt-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ImageGPTForImageClassification.from_pretrained(<span class="hljs-string">&quot;openai/imagegpt-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){a=d("p"),a.textContent=v,m=n(),h(p.$$.fragment)},l(i){a=c(i,"P",{"data-svelte-h":!0}),g(a)!=="svelte-kvfsh7"&&(a.textContent=v),m=s(i),u(p.$$.fragment,i)},m(i,I){r(i,a,I),r(i,m,I),f(p,i,I),b=!0},p:Ne,i(i){b||(_(p.$$.fragment,i),b=!0)},o(i){T(p.$$.fragment,i),b=!1},d(i){i&&(o(a),o(m)),y(p,i)}}}function nn(M){let a,v,m,p,b,i="<em>This model was released on 2020-06-17 and added to Hugging Face Transformers on 2021-11-18.</em>",I,oe,nt,V,go='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',st,ne,at,se,ho=`The ImageGPT model was proposed in <a href="https://openai.com/blog/image-gpt" rel="nofollow">Generative Pretraining from Pixels</a> by Mark
Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever. ImageGPT (iGPT) is a GPT-2-like
model trained to predict the next pixel value, allowing for both unconditional and conditional image generation.`,rt,ae,uo='The abstract from the <a href="https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V1_ICML.pdf" rel="nofollow">paper</a> is the following:',it,re,fo=`<em>Inspired by progress in unsupervised representation learning for natural language, we examine whether similar models
can learn useful representations for images. We train a sequence Transformer to auto-regressively predict pixels,
without incorporating knowledge of the 2D input structure. Despite training on low-resolution ImageNet without labels,
we find that a GPT-2 scale model learns strong image representations as measured by linear probing, fine-tuning, and
low-data classification. On CIFAR-10, we achieve 96.3% accuracy with a linear probe, outperforming a supervised Wide
ResNet, and 99.0% accuracy with full fine-tuning, matching the top supervised pre-trained models. We are also
competitive with self-supervised benchmarks on ImageNet when substituting pixels for a VQVAE encoding, achieving 69.0%
top-1 accuracy on a linear probe of our features.</em>`,lt,S,_o,dt,ie,To="Summary of the approach. Taken from the [original paper](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf).",ct,le,yo=`This model was contributed by <a href="https://huggingface.co/nielsr" rel="nofollow">nielsr</a>, based on <a href="https://github.com/openai/image-gpt/issues/7" rel="nofollow">this issue</a>. The original code can be found
<a href="https://github.com/openai/image-gpt" rel="nofollow">here</a>.`,mt,de,pt,ce,bo=`<li>ImageGPT is almost exactly the same as <a href="gpt2">GPT-2</a>, with the exception that a different activation
function is used (namely “quick gelu”), and the layer normalization layers don’t mean center the inputs. ImageGPT
also doesn’t have tied input- and output embeddings.</li> <li>As the time- and memory requirements of the attention mechanism of Transformers scales quadratically in the sequence
length, the authors pre-trained ImageGPT on smaller input resolutions, such as 32x32 and 64x64. However, feeding a
sequence of 32x32x3=3072 tokens from 0..255 into a Transformer is still prohibitively large. Therefore, the authors
applied k-means clustering to the (R,G,B) pixel values with k=512. This way, we only have a 32*32 = 1024-long
sequence, but now of integers in the range 0..511. So we are shrinking the sequence length at the cost of a bigger
embedding matrix. In other words, the vocabulary size of ImageGPT is 512, + 1 for a special “start of sentence” (SOS)
token, used at the beginning of every sequence. One can use <a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTImageProcessor">ImageGPTImageProcessor</a> to prepare
images for the model.</li> <li>Despite being pre-trained entirely unsupervised (i.e. without the use of any labels), ImageGPT produces fairly
performant image features useful for downstream tasks, such as image classification. The authors showed that the
features in the middle of the network are the most performant, and can be used as-is to train a linear model (such as
a sklearn logistic regression model for example). This is also referred to as “linear probing”. Features can be
easily obtained by first forwarding the image through the model, then specifying <code>output_hidden_states=True</code>, and
then average-pool the hidden states at whatever layer you like.</li> <li>Alternatively, one can further fine-tune the entire model on a downstream dataset, similar to BERT. For this, you can
use <a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification">ImageGPTForImageClassification</a>.</li> <li>ImageGPT comes in different sizes: there’s ImageGPT-small, ImageGPT-medium and ImageGPT-large. The authors did also
train an XL variant, which they didn’t release. The differences in size are summarized in the following table:</li>`,gt,me,vo="<thead><tr><th><strong>Model variant</strong></th> <th><strong>Depths</strong></th> <th><strong>Hidden sizes</strong></th> <th><strong>Decoder hidden size</strong></th> <th><strong>Params (M)</strong></th> <th><strong>ImageNet-1k Top 1</strong></th></tr></thead> <tbody><tr><td>MiT-b0</td> <td>[2, 2, 2, 2]</td> <td>[32, 64, 160, 256]</td> <td>256</td> <td>3.7</td> <td>70.5</td></tr> <tr><td>MiT-b1</td> <td>[2, 2, 2, 2]</td> <td>[64, 128, 320, 512]</td> <td>256</td> <td>14.0</td> <td>78.7</td></tr> <tr><td>MiT-b2</td> <td>[3, 4, 6, 3]</td> <td>[64, 128, 320, 512]</td> <td>768</td> <td>25.4</td> <td>81.6</td></tr> <tr><td>MiT-b3</td> <td>[3, 4, 18, 3]</td> <td>[64, 128, 320, 512]</td> <td>768</td> <td>45.2</td> <td>83.1</td></tr> <tr><td>MiT-b4</td> <td>[3, 8, 27, 3]</td> <td>[64, 128, 320, 512]</td> <td>768</td> <td>62.6</td> <td>83.6</td></tr> <tr><td>MiT-b5</td> <td>[3, 6, 40, 3]</td> <td>[64, 128, 320, 512]</td> <td>768</td> <td>82.0</td> <td>83.8</td></tr></tbody>",ht,pe,ut,ge,Io="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ImageGPT.",ft,he,_t,ue,wo='<li>Demo notebooks for ImageGPT can be found <a href="https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ImageGPT" rel="nofollow">here</a>.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification">ImageGPTForImageClassification</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification" rel="nofollow">example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb" rel="nofollow">notebook</a>.</li> <li>See also: <a href="../tasks/image_classification">Image classification task guide</a></li>',Tt,fe,Mo="If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",yt,_e,bt,C,Te,Wt,Ze,ko=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTModel">ImageGPTModel</a> or a <code>TFImageGPTModel</code>. It is
used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the ImageGPT
<a href="https://huggingface.co/openai/imagegpt-small" rel="nofollow">openai/imagegpt-small</a> architecture.`,qt,Re,Po=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Nt,L,vt,ye,It,B,be,Zt,E,ve,Rt,Be,$o="Preprocess an image or a batch of images.",wt,Ie,Mt,F,we,Bt,He,Co=`Constructs a ImageGPT image processor. This image processor can be used to resize images to a smaller resolution
(such as 32x32 or 64x64), normalize them and finally color quantize them to obtain sequences of “pixel values”
(color clusters).`,Ht,X,Me,Vt,Ve,Go="Preprocess an image or batch of images.",kt,ke,Pt,k,Pe,St,Se,xo="The bare Imagegpt Model outputting raw hidden-states without any specific head on top.",Lt,Le,zo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Et,Ee,jo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xt,z,$e,Qt,Xe,Uo='The <a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTModel">ImageGPTModel</a> forward method, overrides the <code>__call__</code> special method.',Ot,Q,At,O,$t,Ce,Ct,P,Ge,Dt,Qe,Fo=`The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`,Yt,Oe,Jo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Kt,Ae,Wo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,eo,j,xe,to,De,qo='The <a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForCausalImageModeling">ImageGPTForCausalImageModeling</a> forward method, overrides the <code>__call__</code> special method.',oo,A,no,D,Gt,ze,xt,$,je,so,Ye,No=`The ImageGPT Model transformer with an image classification head on top (linear layer).
<a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification">ImageGPTForImageClassification</a> average-pools the hidden states in order to do the classification.`,ao,Ke,Zo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ro,et,Ro=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,io,U,Ue,lo,tt,Bo='The <a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification">ImageGPTForImageClassification</a> forward method, overrides the <code>__call__</code> special method.',co,Y,mo,K,zt,Fe,jt,ot,Ut;return oe=new R({props:{title:"ImageGPT",local:"imagegpt",headingTag:"h1"}}),ne=new R({props:{title:"Overview",local:"overview",headingTag:"h2"}}),de=new R({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),pe=new R({props:{title:"Resources",local:"resources",headingTag:"h2"}}),he=new Qo({props:{pipeline:"image-classification"}}),_e=new R({props:{title:"ImageGPTConfig",local:"transformers.ImageGPTConfig",headingTag:"h2"}}),Te=new q({props:{name:"class transformers.ImageGPTConfig",anchor:"transformers.ImageGPTConfig",parameters:[{name:"vocab_size",val:" = 513"},{name:"n_positions",val:" = 1024"},{name:"n_embd",val:" = 512"},{name:"n_layer",val:" = 24"},{name:"n_head",val:" = 8"},{name:"n_inner",val:" = None"},{name:"activation_function",val:" = 'quick_gelu'"},{name:"resid_pdrop",val:" = 0.1"},{name:"embd_pdrop",val:" = 0.1"},{name:"attn_pdrop",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-05"},{name:"initializer_range",val:" = 0.02"},{name:"scale_attn_weights",val:" = True"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"scale_attn_by_inverse_layer_idx",val:" = False"},{name:"reorder_and_upcast_attn",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ImageGPTConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTModel">ImageGPTModel</a> or <code>TFImageGPTModel</code>.`,name:"vocab_size"},{anchor:"transformers.ImageGPTConfig.n_positions",description:`<strong>n_positions</strong> (<code>int</code>, <em>optional</em>, defaults to 32*32) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"n_positions"},{anchor:"transformers.ImageGPTConfig.n_embd",description:`<strong>n_embd</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the embeddings and hidden states.`,name:"n_embd"},{anchor:"transformers.ImageGPTConfig.n_layer",description:`<strong>n_layer</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"n_layer"},{anchor:"transformers.ImageGPTConfig.n_head",description:`<strong>n_head</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"n_head"},{anchor:"transformers.ImageGPTConfig.n_inner",description:`<strong>n_inner</strong> (<code>int</code>, <em>optional</em>, defaults to None) &#x2014;
Dimensionality of the inner feed-forward layers. <code>None</code> will set it to 4 times n_embd`,name:"n_inner"},{anchor:"transformers.ImageGPTConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;quick_gelu&quot;</code>) &#x2014;
Activation function (can be one of the activation functions defined in src/transformers/activations.py).
Defaults to &#x201C;quick_gelu&#x201D;.`,name:"activation_function"},{anchor:"transformers.ImageGPTConfig.resid_pdrop",description:`<strong>resid_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"resid_pdrop"},{anchor:"transformers.ImageGPTConfig.embd_pdrop",description:`<strong>embd_pdrop</strong> (<code>int</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the embeddings.`,name:"embd_pdrop"},{anchor:"transformers.ImageGPTConfig.attn_pdrop",description:`<strong>attn_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention.`,name:"attn_pdrop"},{anchor:"transformers.ImageGPTConfig.layer_norm_epsilon",description:`<strong>layer_norm_epsilon</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon to use in the layer normalization layers.`,name:"layer_norm_epsilon"},{anchor:"transformers.ImageGPTConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.ImageGPTConfig.scale_attn_weights",description:`<strong>scale_attn_weights</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Scale attention weights by dividing by sqrt(hidden_size)..`,name:"scale_attn_weights"},{anchor:"transformers.ImageGPTConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.ImageGPTConfig.scale_attn_by_inverse_layer_idx",description:`<strong>scale_attn_by_inverse_layer_idx</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to additionally scale attention weights by <code>1 / layer_idx + 1</code>.`,name:"scale_attn_by_inverse_layer_idx"},{anchor:"transformers.ImageGPTConfig.reorder_and_upcast_attn",description:`<strong>reorder_and_upcast_attn</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
dot-product/softmax to float() when training with mixed precision.`,name:"reorder_and_upcast_attn"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/configuration_imagegpt.py#L32"}}),L=new Ft({props:{anchor:"transformers.ImageGPTConfig.example",$$slots:{default:[Ao]},$$scope:{ctx:M}}}),ye=new R({props:{title:"ImageGPTFeatureExtractor",local:"transformers.ImageGPTFeatureExtractor",headingTag:"h2"}}),be=new q({props:{name:"class transformers.ImageGPTFeatureExtractor",anchor:"transformers.ImageGPTFeatureExtractor",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/feature_extraction_imagegpt.py#L28"}}),ve=new q({props:{name:"__call__",anchor:"transformers.ImageGPTFeatureExtractor.__call__",parameters:[{name:"images",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49"}}),Ie=new R({props:{title:"ImageGPTImageProcessor",local:"transformers.ImageGPTImageProcessor",headingTag:"h2"}}),we=new q({props:{name:"class transformers.ImageGPTImageProcessor",anchor:"transformers.ImageGPTImageProcessor",parameters:[{name:"clusters",val:": typing.Union[list[list[int]], numpy.ndarray, NoneType] = None"},{name:"do_resize",val:": bool = True"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"resample",val:": Resampling = <Resampling.BILINEAR: 2>"},{name:"do_normalize",val:": bool = True"},{name:"do_color_quantize",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ImageGPTImageProcessor.clusters",description:`<strong>clusters</strong> (<code>np.ndarray</code> or <code>list[list[int]]</code>, <em>optional</em>) &#x2014;
The color clusters to use, of shape <code>(n_clusters, 3)</code> when color quantizing. Can be overridden by <code>clusters</code>
in <code>preprocess</code>.`,name:"clusters"},{anchor:"transformers.ImageGPTImageProcessor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the image&#x2019;s dimensions to <code>(size[&quot;height&quot;], size[&quot;width&quot;])</code>. Can be overridden by
<code>do_resize</code> in <code>preprocess</code>.`,name:"do_resize"},{anchor:"transformers.ImageGPTImageProcessor.size",description:`<strong>size</strong> (<code>dict[str, int]</code> <em>optional</em>, defaults to <code>{&quot;height&quot; -- 256, &quot;width&quot;: 256}</code>):
Size of the image after resizing. Can be overridden by <code>size</code> in <code>preprocess</code>.`,name:"size"},{anchor:"transformers.ImageGPTImageProcessor.resample",description:`<strong>resample</strong> (<code>PILImageResampling</code>, <em>optional</em>, defaults to <code>Resampling.BILINEAR</code>) &#x2014;
Resampling filter to use if resizing the image. Can be overridden by <code>resample</code> in <code>preprocess</code>.`,name:"resample"},{anchor:"transformers.ImageGPTImageProcessor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize the image pixel value to between [-1, 1]. Can be overridden by <code>do_normalize</code> in
<code>preprocess</code>.`,name:"do_normalize"},{anchor:"transformers.ImageGPTImageProcessor.do_color_quantize",description:`<strong>do_color_quantize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to color quantize the image. Can be overridden by <code>do_color_quantize</code> in <code>preprocess</code>.`,name:"do_color_quantize"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/image_processing_imagegpt.py#L61"}}),Me=new q({props:{name:"preprocess",anchor:"transformers.ImageGPTImageProcessor.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"do_resize",val:": typing.Optional[bool] = None"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"resample",val:": Resampling = None"},{name:"do_normalize",val:": typing.Optional[bool] = None"},{name:"do_color_quantize",val:": typing.Optional[bool] = None"},{name:"clusters",val:": typing.Union[list[list[int]], numpy.ndarray, NoneType] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = <ChannelDimension.FIRST: 'channels_first'>"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}],parametersDescription:[{anchor:"transformers.ImageGPTImageProcessor.preprocess.images",description:`<strong>images</strong> (<code>ImageInput</code>) &#x2014;
Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
passing in images with pixel values between 0 and 1, set <code>do_normalize=False</code>.`,name:"images"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_resize</code>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>, defaults to <code>self.size</code>) &#x2014;
Size of the image after resizing.`,name:"size"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.resample</code>) &#x2014;
Resampling filter to use if resizing the image. This can be one of the enum <code>PILImageResampling</code>, Only
has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_normalize</code>) &#x2014;
Whether to normalize the image`,name:"do_normalize"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.do_color_quantize",description:`<strong>do_color_quantize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_color_quantize</code>) &#x2014;
Whether to color quantize the image.`,name:"do_color_quantize"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.clusters",description:`<strong>clusters</strong> (<code>np.ndarray</code> or <code>list[list[int]]</code>, <em>optional</em>, defaults to <code>self.clusters</code>) &#x2014;
Clusters used to quantize the image of shape <code>(n_clusters, 3)</code>. Only has an effect if
<code>do_color_quantize</code> is set to <code>True</code>.`,name:"clusters"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <code>TensorType</code>, <em>optional</em>) &#x2014;
The type of tensors to return. Can be one of:<ul>
<li>Unset: Return a list of <code>np.ndarray</code>.</li>
<li><code>TensorType.TENSORFLOW</code> or <code>&apos;tf&apos;</code>: Return a batch of type <code>tf.Tensor</code>.</li>
<li><code>TensorType.PYTORCH</code> or <code>&apos;pt&apos;</code>: Return a batch of type <code>torch.Tensor</code>.</li>
<li><code>TensorType.NUMPY</code> or <code>&apos;np&apos;</code>: Return a batch of type <code>np.ndarray</code>.</li>
<li><code>TensorType.JAX</code> or <code>&apos;jax&apos;</code>: Return a batch of type <code>jax.numpy.ndarray</code>.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.data_format",description:`<strong>data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>, defaults to <code>ChannelDimension.FIRST</code>) &#x2014;
The channel dimension format for the output image. Can be one of:<ul>
<li><code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.
Only has an effect if <code>do_color_quantize</code> is set to <code>False</code>.</li>
</ul>`,name:"data_format"},{anchor:"transformers.ImageGPTImageProcessor.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/image_processing_imagegpt.py#L178"}}),ke=new R({props:{title:"ImageGPTModel",local:"transformers.ImageGPTModel",headingTag:"h2"}}),Pe=new q({props:{name:"class transformers.ImageGPTModel",anchor:"transformers.ImageGPTModel",parameters:[{name:"config",val:": ImageGPTConfig"}],parametersDescription:[{anchor:"transformers.ImageGPTModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig">ImageGPTConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L539"}}),$e=new q({props:{name:"forward",anchor:"transformers.ImageGPTModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing.Any"}],parametersDescription:[{anchor:"transformers.ImageGPTModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor">AutoImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ImageGPTImageProcessor.<strong>call</strong>()</a> for details.`,name:"input_ids"},{anchor:"transformers.ImageGPTModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ImageGPTModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ImageGPTModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ImageGPTModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ImageGPTModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ImageGPTModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ImageGPTModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ImageGPTModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ImageGPTModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ImageGPTModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ImageGPTModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ImageGPTModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ImageGPTModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L572",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig"
>ImageGPTConfig</a>) and inputs.</p>
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
`}}),Q=new po({props:{$$slots:{default:[Do]},$$scope:{ctx:M}}}),O=new Ft({props:{anchor:"transformers.ImageGPTModel.forward.example",$$slots:{default:[Yo]},$$scope:{ctx:M}}}),Ce=new R({props:{title:"ImageGPTForCausalImageModeling",local:"transformers.ImageGPTForCausalImageModeling",headingTag:"h2"}}),Ge=new q({props:{name:"class transformers.ImageGPTForCausalImageModeling",anchor:"transformers.ImageGPTForCausalImageModeling",parameters:[{name:"config",val:": ImageGPTConfig"}],parametersDescription:[{anchor:"transformers.ImageGPTForCausalImageModeling.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig">ImageGPTConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L785"}}),xe=new q({props:{name:"forward",anchor:"transformers.ImageGPTForCausalImageModeling.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing.Any"}],parametersDescription:[{anchor:"transformers.ImageGPTForCausalImageModeling.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor">AutoImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ImageGPTImageProcessor.<strong>call</strong>()</a> for details.`,name:"input_ids"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ImageGPTForCausalImageModeling.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L799",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig"
>ImageGPTConfig</a>) and inputs.</p>
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
`}}),A=new po({props:{$$slots:{default:[Ko]},$$scope:{ctx:M}}}),D=new Ft({props:{anchor:"transformers.ImageGPTForCausalImageModeling.forward.example",$$slots:{default:[en]},$$scope:{ctx:M}}}),ze=new R({props:{title:"ImageGPTForImageClassification",local:"transformers.ImageGPTForImageClassification",headingTag:"h2"}}),je=new q({props:{name:"class transformers.ImageGPTForImageClassification",anchor:"transformers.ImageGPTForImageClassification",parameters:[{name:"config",val:": ImageGPTConfig"}],parametersDescription:[{anchor:"transformers.ImageGPTForImageClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig">ImageGPTConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L921"}}),Ue=new q({props:{name:"forward",anchor:"transformers.ImageGPTForImageClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing.Any"}],parametersDescription:[{anchor:"transformers.ImageGPTForImageClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor">AutoImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ImageGPTImageProcessor.<strong>call</strong>()</a> for details.`,name:"input_ids"},{anchor:"transformers.ImageGPTForImageClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ImageGPTForImageClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ImageGPTForImageClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ImageGPTForImageClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ImageGPTForImageClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ImageGPTForImageClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ImageGPTForImageClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.ImageGPTForImageClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ImageGPTForImageClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ImageGPTForImageClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ImageGPTForImageClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/imagegpt/modeling_imagegpt.py#L931",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/imagegpt#transformers.ImageGPTConfig"
>ImageGPTConfig</a>) and inputs.</p>
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
`}}),Y=new po({props:{$$slots:{default:[tn]},$$scope:{ctx:M}}}),K=new Ft({props:{anchor:"transformers.ImageGPTForImageClassification.forward.example",$$slots:{default:[on]},$$scope:{ctx:M}}}),Fe=new Oo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/imagegpt.md"}}),{c(){a=d("meta"),v=n(),m=d("p"),p=n(),b=d("p"),b.innerHTML=i,I=n(),h(oe.$$.fragment),nt=n(),V=d("div"),V.innerHTML=go,st=n(),h(ne.$$.fragment),at=n(),se=d("p"),se.innerHTML=ho,rt=n(),ae=d("p"),ae.innerHTML=uo,it=n(),re=d("p"),re.innerHTML=fo,lt=n(),S=d("img"),dt=n(),ie=d("small"),ie.textContent=To,ct=n(),le=d("p"),le.innerHTML=yo,mt=n(),h(de.$$.fragment),pt=n(),ce=d("ul"),ce.innerHTML=bo,gt=n(),me=d("table"),me.innerHTML=vo,ht=n(),h(pe.$$.fragment),ut=n(),ge=d("p"),ge.textContent=Io,ft=n(),h(he.$$.fragment),_t=n(),ue=d("ul"),ue.innerHTML=wo,Tt=n(),fe=d("p"),fe.textContent=Mo,yt=n(),h(_e.$$.fragment),bt=n(),C=d("div"),h(Te.$$.fragment),Wt=n(),Ze=d("p"),Ze.innerHTML=ko,qt=n(),Re=d("p"),Re.innerHTML=Po,Nt=n(),h(L.$$.fragment),vt=n(),h(ye.$$.fragment),It=n(),B=d("div"),h(be.$$.fragment),Zt=n(),E=d("div"),h(ve.$$.fragment),Rt=n(),Be=d("p"),Be.textContent=$o,wt=n(),h(Ie.$$.fragment),Mt=n(),F=d("div"),h(we.$$.fragment),Bt=n(),He=d("p"),He.textContent=Co,Ht=n(),X=d("div"),h(Me.$$.fragment),Vt=n(),Ve=d("p"),Ve.textContent=Go,kt=n(),h(ke.$$.fragment),Pt=n(),k=d("div"),h(Pe.$$.fragment),St=n(),Se=d("p"),Se.textContent=xo,Lt=n(),Le=d("p"),Le.innerHTML=zo,Et=n(),Ee=d("p"),Ee.innerHTML=jo,Xt=n(),z=d("div"),h($e.$$.fragment),Qt=n(),Xe=d("p"),Xe.innerHTML=Uo,Ot=n(),h(Q.$$.fragment),At=n(),h(O.$$.fragment),$t=n(),h(Ce.$$.fragment),Ct=n(),P=d("div"),h(Ge.$$.fragment),Dt=n(),Qe=d("p"),Qe.textContent=Fo,Yt=n(),Oe=d("p"),Oe.innerHTML=Jo,Kt=n(),Ae=d("p"),Ae.innerHTML=Wo,eo=n(),j=d("div"),h(xe.$$.fragment),to=n(),De=d("p"),De.innerHTML=qo,oo=n(),h(A.$$.fragment),no=n(),h(D.$$.fragment),Gt=n(),h(ze.$$.fragment),xt=n(),$=d("div"),h(je.$$.fragment),so=n(),Ye=d("p"),Ye.innerHTML=No,ao=n(),Ke=d("p"),Ke.innerHTML=Zo,ro=n(),et=d("p"),et.innerHTML=Ro,io=n(),U=d("div"),h(Ue.$$.fragment),lo=n(),tt=d("p"),tt.innerHTML=Bo,co=n(),h(Y.$$.fragment),mo=n(),h(K.$$.fragment),zt=n(),h(Fe.$$.fragment),jt=n(),ot=d("p"),this.h()},l(e){const t=Xo("svelte-u9bgzb",document.head);a=c(t,"META",{name:!0,content:!0}),t.forEach(o),v=s(e),m=c(e,"P",{}),x(m).forEach(o),p=s(e),b=c(e,"P",{"data-svelte-h":!0}),g(b)!=="svelte-104gbkc"&&(b.innerHTML=i),I=s(e),u(oe.$$.fragment,e),nt=s(e),V=c(e,"DIV",{class:!0,"data-svelte-h":!0}),g(V)!=="svelte-13t8s2t"&&(V.innerHTML=go),st=s(e),u(ne.$$.fragment,e),at=s(e),se=c(e,"P",{"data-svelte-h":!0}),g(se)!=="svelte-thwva5"&&(se.innerHTML=ho),rt=s(e),ae=c(e,"P",{"data-svelte-h":!0}),g(ae)!=="svelte-ea76ky"&&(ae.innerHTML=uo),it=s(e),re=c(e,"P",{"data-svelte-h":!0}),g(re)!=="svelte-1nknq39"&&(re.innerHTML=fo),lt=s(e),S=c(e,"IMG",{src:!0,alt:!0,width:!0}),dt=s(e),ie=c(e,"SMALL",{"data-svelte-h":!0}),g(ie)!=="svelte-1yh5y0e"&&(ie.textContent=To),ct=s(e),le=c(e,"P",{"data-svelte-h":!0}),g(le)!=="svelte-1y5dk1"&&(le.innerHTML=yo),mt=s(e),u(de.$$.fragment,e),pt=s(e),ce=c(e,"UL",{"data-svelte-h":!0}),g(ce)!=="svelte-12nzu1l"&&(ce.innerHTML=bo),gt=s(e),me=c(e,"TABLE",{"data-svelte-h":!0}),g(me)!=="svelte-631n1h"&&(me.innerHTML=vo),ht=s(e),u(pe.$$.fragment,e),ut=s(e),ge=c(e,"P",{"data-svelte-h":!0}),g(ge)!=="svelte-34o9qd"&&(ge.textContent=Io),ft=s(e),u(he.$$.fragment,e),_t=s(e),ue=c(e,"UL",{"data-svelte-h":!0}),g(ue)!=="svelte-nt5eb"&&(ue.innerHTML=wo),Tt=s(e),fe=c(e,"P",{"data-svelte-h":!0}),g(fe)!=="svelte-1xesile"&&(fe.textContent=Mo),yt=s(e),u(_e.$$.fragment,e),bt=s(e),C=c(e,"DIV",{class:!0});var J=x(C);u(Te.$$.fragment,J),Wt=s(J),Ze=c(J,"P",{"data-svelte-h":!0}),g(Ze)!=="svelte-9npsyl"&&(Ze.innerHTML=ko),qt=s(J),Re=c(J,"P",{"data-svelte-h":!0}),g(Re)!=="svelte-1ek1ss9"&&(Re.innerHTML=Po),Nt=s(J),u(L.$$.fragment,J),J.forEach(o),vt=s(e),u(ye.$$.fragment,e),It=s(e),B=c(e,"DIV",{class:!0});var Je=x(B);u(be.$$.fragment,Je),Zt=s(Je),E=c(Je,"DIV",{class:!0});var We=x(E);u(ve.$$.fragment,We),Rt=s(We),Be=c(We,"P",{"data-svelte-h":!0}),g(Be)!=="svelte-khengj"&&(Be.textContent=$o),We.forEach(o),Je.forEach(o),wt=s(e),u(Ie.$$.fragment,e),Mt=s(e),F=c(e,"DIV",{class:!0});var H=x(F);u(we.$$.fragment,H),Bt=s(H),He=c(H,"P",{"data-svelte-h":!0}),g(He)!=="svelte-4cfp03"&&(He.textContent=Co),Ht=s(H),X=c(H,"DIV",{class:!0});var qe=x(X);u(Me.$$.fragment,qe),Vt=s(qe),Ve=c(qe,"P",{"data-svelte-h":!0}),g(Ve)!=="svelte-1x3yxsa"&&(Ve.textContent=Go),qe.forEach(o),H.forEach(o),kt=s(e),u(ke.$$.fragment,e),Pt=s(e),k=c(e,"DIV",{class:!0});var G=x(k);u(Pe.$$.fragment,G),St=s(G),Se=c(G,"P",{"data-svelte-h":!0}),g(Se)!=="svelte-m9fhq2"&&(Se.textContent=xo),Lt=s(G),Le=c(G,"P",{"data-svelte-h":!0}),g(Le)!=="svelte-q52n56"&&(Le.innerHTML=zo),Et=s(G),Ee=c(G,"P",{"data-svelte-h":!0}),g(Ee)!=="svelte-hswkmf"&&(Ee.innerHTML=jo),Xt=s(G),z=c(G,"DIV",{class:!0});var W=x(z);u($e.$$.fragment,W),Qt=s(W),Xe=c(W,"P",{"data-svelte-h":!0}),g(Xe)!=="svelte-1x2rhl1"&&(Xe.innerHTML=Uo),Ot=s(W),u(Q.$$.fragment,W),At=s(W),u(O.$$.fragment,W),W.forEach(o),G.forEach(o),$t=s(e),u(Ce.$$.fragment,e),Ct=s(e),P=c(e,"DIV",{class:!0});var N=x(P);u(Ge.$$.fragment,N),Dt=s(N),Qe=c(N,"P",{"data-svelte-h":!0}),g(Qe)!=="svelte-1jpvdp1"&&(Qe.textContent=Fo),Yt=s(N),Oe=c(N,"P",{"data-svelte-h":!0}),g(Oe)!=="svelte-q52n56"&&(Oe.innerHTML=Jo),Kt=s(N),Ae=c(N,"P",{"data-svelte-h":!0}),g(Ae)!=="svelte-hswkmf"&&(Ae.innerHTML=Wo),eo=s(N),j=c(N,"DIV",{class:!0});var ee=x(j);u(xe.$$.fragment,ee),to=s(ee),De=c(ee,"P",{"data-svelte-h":!0}),g(De)!=="svelte-4uuz6x"&&(De.innerHTML=qo),oo=s(ee),u(A.$$.fragment,ee),no=s(ee),u(D.$$.fragment,ee),ee.forEach(o),N.forEach(o),Gt=s(e),u(ze.$$.fragment,e),xt=s(e),$=c(e,"DIV",{class:!0});var Z=x($);u(je.$$.fragment,Z),so=s(Z),Ye=c(Z,"P",{"data-svelte-h":!0}),g(Ye)!=="svelte-1xr6udz"&&(Ye.innerHTML=No),ao=s(Z),Ke=c(Z,"P",{"data-svelte-h":!0}),g(Ke)!=="svelte-q52n56"&&(Ke.innerHTML=Zo),ro=s(Z),et=c(Z,"P",{"data-svelte-h":!0}),g(et)!=="svelte-hswkmf"&&(et.innerHTML=Ro),io=s(Z),U=c(Z,"DIV",{class:!0});var te=x(U);u(Ue.$$.fragment,te),lo=s(te),tt=c(te,"P",{"data-svelte-h":!0}),g(tt)!=="svelte-1iby34h"&&(tt.innerHTML=Bo),co=s(te),u(Y.$$.fragment,te),mo=s(te),u(K.$$.fragment,te),te.forEach(o),Z.forEach(o),zt=s(e),u(Fe.$$.fragment,e),jt=s(e),ot=c(e,"P",{}),x(ot).forEach(o),this.h()},h(){w(a,"name","hf:doc:metadata"),w(a,"content",sn),w(V,"class","flex flex-wrap space-x-1"),Vo(S.src,_o="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/imagegpt_architecture.png")||w(S,"src",_o),w(S,"alt","drawing"),w(S,"width","600"),w(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){l(document.head,a),r(e,v,t),r(e,m,t),r(e,p,t),r(e,b,t),r(e,I,t),f(oe,e,t),r(e,nt,t),r(e,V,t),r(e,st,t),f(ne,e,t),r(e,at,t),r(e,se,t),r(e,rt,t),r(e,ae,t),r(e,it,t),r(e,re,t),r(e,lt,t),r(e,S,t),r(e,dt,t),r(e,ie,t),r(e,ct,t),r(e,le,t),r(e,mt,t),f(de,e,t),r(e,pt,t),r(e,ce,t),r(e,gt,t),r(e,me,t),r(e,ht,t),f(pe,e,t),r(e,ut,t),r(e,ge,t),r(e,ft,t),f(he,e,t),r(e,_t,t),r(e,ue,t),r(e,Tt,t),r(e,fe,t),r(e,yt,t),f(_e,e,t),r(e,bt,t),r(e,C,t),f(Te,C,null),l(C,Wt),l(C,Ze),l(C,qt),l(C,Re),l(C,Nt),f(L,C,null),r(e,vt,t),f(ye,e,t),r(e,It,t),r(e,B,t),f(be,B,null),l(B,Zt),l(B,E),f(ve,E,null),l(E,Rt),l(E,Be),r(e,wt,t),f(Ie,e,t),r(e,Mt,t),r(e,F,t),f(we,F,null),l(F,Bt),l(F,He),l(F,Ht),l(F,X),f(Me,X,null),l(X,Vt),l(X,Ve),r(e,kt,t),f(ke,e,t),r(e,Pt,t),r(e,k,t),f(Pe,k,null),l(k,St),l(k,Se),l(k,Lt),l(k,Le),l(k,Et),l(k,Ee),l(k,Xt),l(k,z),f($e,z,null),l(z,Qt),l(z,Xe),l(z,Ot),f(Q,z,null),l(z,At),f(O,z,null),r(e,$t,t),f(Ce,e,t),r(e,Ct,t),r(e,P,t),f(Ge,P,null),l(P,Dt),l(P,Qe),l(P,Yt),l(P,Oe),l(P,Kt),l(P,Ae),l(P,eo),l(P,j),f(xe,j,null),l(j,to),l(j,De),l(j,oo),f(A,j,null),l(j,no),f(D,j,null),r(e,Gt,t),f(ze,e,t),r(e,xt,t),r(e,$,t),f(je,$,null),l($,so),l($,Ye),l($,ao),l($,Ke),l($,ro),l($,et),l($,io),l($,U),f(Ue,U,null),l(U,lo),l(U,tt),l(U,co),f(Y,U,null),l(U,mo),f(K,U,null),r(e,zt,t),f(Fe,e,t),r(e,jt,t),r(e,ot,t),Ut=!0},p(e,[t]){const J={};t&2&&(J.$$scope={dirty:t,ctx:e}),L.$set(J);const Je={};t&2&&(Je.$$scope={dirty:t,ctx:e}),Q.$set(Je);const We={};t&2&&(We.$$scope={dirty:t,ctx:e}),O.$set(We);const H={};t&2&&(H.$$scope={dirty:t,ctx:e}),A.$set(H);const qe={};t&2&&(qe.$$scope={dirty:t,ctx:e}),D.$set(qe);const G={};t&2&&(G.$$scope={dirty:t,ctx:e}),Y.$set(G);const W={};t&2&&(W.$$scope={dirty:t,ctx:e}),K.$set(W)},i(e){Ut||(_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(de.$$.fragment,e),_(pe.$$.fragment,e),_(he.$$.fragment,e),_(_e.$$.fragment,e),_(Te.$$.fragment,e),_(L.$$.fragment,e),_(ye.$$.fragment,e),_(be.$$.fragment,e),_(ve.$$.fragment,e),_(Ie.$$.fragment,e),_(we.$$.fragment,e),_(Me.$$.fragment,e),_(ke.$$.fragment,e),_(Pe.$$.fragment,e),_($e.$$.fragment,e),_(Q.$$.fragment,e),_(O.$$.fragment,e),_(Ce.$$.fragment,e),_(Ge.$$.fragment,e),_(xe.$$.fragment,e),_(A.$$.fragment,e),_(D.$$.fragment,e),_(ze.$$.fragment,e),_(je.$$.fragment,e),_(Ue.$$.fragment,e),_(Y.$$.fragment,e),_(K.$$.fragment,e),_(Fe.$$.fragment,e),Ut=!0)},o(e){T(oe.$$.fragment,e),T(ne.$$.fragment,e),T(de.$$.fragment,e),T(pe.$$.fragment,e),T(he.$$.fragment,e),T(_e.$$.fragment,e),T(Te.$$.fragment,e),T(L.$$.fragment,e),T(ye.$$.fragment,e),T(be.$$.fragment,e),T(ve.$$.fragment,e),T(Ie.$$.fragment,e),T(we.$$.fragment,e),T(Me.$$.fragment,e),T(ke.$$.fragment,e),T(Pe.$$.fragment,e),T($e.$$.fragment,e),T(Q.$$.fragment,e),T(O.$$.fragment,e),T(Ce.$$.fragment,e),T(Ge.$$.fragment,e),T(xe.$$.fragment,e),T(A.$$.fragment,e),T(D.$$.fragment,e),T(ze.$$.fragment,e),T(je.$$.fragment,e),T(Ue.$$.fragment,e),T(Y.$$.fragment,e),T(K.$$.fragment,e),T(Fe.$$.fragment,e),Ut=!1},d(e){e&&(o(v),o(m),o(p),o(b),o(I),o(nt),o(V),o(st),o(at),o(se),o(rt),o(ae),o(it),o(re),o(lt),o(S),o(dt),o(ie),o(ct),o(le),o(mt),o(pt),o(ce),o(gt),o(me),o(ht),o(ut),o(ge),o(ft),o(_t),o(ue),o(Tt),o(fe),o(yt),o(bt),o(C),o(vt),o(It),o(B),o(wt),o(Mt),o(F),o(kt),o(Pt),o(k),o($t),o(Ct),o(P),o(Gt),o(xt),o($),o(zt),o(jt),o(ot)),o(a),y(oe,e),y(ne,e),y(de,e),y(pe,e),y(he,e),y(_e,e),y(Te),y(L),y(ye,e),y(be),y(ve),y(Ie,e),y(we),y(Me),y(ke,e),y(Pe),y($e),y(Q),y(O),y(Ce,e),y(Ge),y(xe),y(A),y(D),y(ze,e),y(je),y(Ue),y(Y),y(K),y(Fe,e)}}}const sn='{"title":"ImageGPT","local":"imagegpt","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"ImageGPTConfig","local":"transformers.ImageGPTConfig","sections":[],"depth":2},{"title":"ImageGPTFeatureExtractor","local":"transformers.ImageGPTFeatureExtractor","sections":[],"depth":2},{"title":"ImageGPTImageProcessor","local":"transformers.ImageGPTImageProcessor","sections":[],"depth":2},{"title":"ImageGPTModel","local":"transformers.ImageGPTModel","sections":[],"depth":2},{"title":"ImageGPTForCausalImageModeling","local":"transformers.ImageGPTForCausalImageModeling","sections":[],"depth":2},{"title":"ImageGPTForImageClassification","local":"transformers.ImageGPTForImageClassification","sections":[],"depth":2}],"depth":1}';function an(M){return So(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class un extends Lo{constructor(a){super(),Eo(this,a,an,nn,Ho,{})}}export{un as component};
