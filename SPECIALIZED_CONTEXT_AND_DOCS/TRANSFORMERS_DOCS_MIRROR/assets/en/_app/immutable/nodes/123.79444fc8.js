import{s as Bn,z as Sn,o as Ln,n as et}from"../chunks/scheduler.18a86fab.js";import{S as Gn,i as On,g as r,s as n,r as p,A as Hn,h as i,f as o,c as s,j as C,x as y,u as h,k as w,y as t,a as l,v as g,d as f,t as u,w as _}from"../chunks/index.98837b22.js";import{T as rn}from"../chunks/Tip.77304350.js";import{D as x}from"../chunks/Docstring.a1ef7999.js";import{C as po}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as mo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as S,E as Vn}from"../chunks/getInferenceSnippets.06c2775f.js";function Xn(I){let d,T="Examples:",m,b,v;return b=new po({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENvbmRpdGlvbmFsRGV0ckNvbmZpZyUyQyUyMENvbmRpdGlvbmFsRGV0ck1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMENvbmRpdGlvbmFsJTIwREVUUiUyMG1pY3Jvc29mdCUyRmNvbmRpdGlvbmFsLWRldHItcmVzbmV0LTUwJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMENvbmRpdGlvbmFsRGV0ckNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBtaWNyb3NvZnQlMkZjb25kaXRpb25hbC1kZXRyLXJlc25ldC01MCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwQ29uZGl0aW9uYWxEZXRyTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ConditionalDetrConfig, ConditionalDetrModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Conditional DETR microsoft/conditional-detr-resnet-50 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = ConditionalDetrConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the microsoft/conditional-detr-resnet-50 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ConditionalDetrModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){d=r("p"),d.textContent=T,m=n(),p(b.$$.fragment)},l(c){d=i(c,"P",{"data-svelte-h":!0}),y(d)!=="svelte-kvfsh7"&&(d.textContent=T),m=s(c),h(b.$$.fragment,c)},m(c,D){l(c,d,D),l(c,m,D),g(b,c,D),v=!0},p:et,i(c){v||(f(b.$$.fragment,c),v=!0)},o(c){u(b.$$.fragment,c),v=!1},d(c){c&&(o(d),o(m)),_(b,c)}}}function An(I){let d,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){d=r("p"),d.innerHTML=T},l(m){d=i(m,"P",{"data-svelte-h":!0}),y(d)!=="svelte-fincs2"&&(d.innerHTML=T)},m(m,b){l(m,d,b)},p:et,d(m){m&&o(d)}}}function Qn(I){let d,T="Examples:",m,b,v;return b=new po({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbCUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHJlcXVlc3RzJTBBJTBBdXJsJTIwJTNEJTIwJTIyaHR0cCUzQSUyRiUyRmltYWdlcy5jb2NvZGF0YXNldC5vcmclMkZ2YWwyMDE3JTJGMDAwMDAwMDM5NzY5LmpwZyUyMiUwQWltYWdlJTIwJTNEJTIwSW1hZ2Uub3BlbihyZXF1ZXN0cy5nZXQodXJsJTJDJTIwc3RyZWFtJTNEVHJ1ZSkucmF3KSUwQSUwQWltYWdlX3Byb2Nlc3NvciUyMCUzRCUyMEF1dG9JbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGY29uZGl0aW9uYWwtZGV0ci1yZXNuZXQtNTAlMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRmNvbmRpdGlvbmFsLWRldHItcmVzbmV0LTUwJTIyKSUwQSUwQSUyMyUyMHByZXBhcmUlMjBpbWFnZSUyMGZvciUyMHRoZSUyMG1vZGVsJTBBaW5wdXRzJTIwJTNEJTIwaW1hZ2VfcHJvY2Vzc29yKGltYWdlcyUzRGltYWdlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBmb3J3YXJkJTIwcGFzcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEElMjMlMjB0aGUlMjBsYXN0JTIwaGlkZGVuJTIwc3RhdGVzJTIwYXJlJTIwdGhlJTIwZmluYWwlMjBxdWVyeSUyMGVtYmVkZGluZ3MlMjBvZiUyMHRoZSUyMFRyYW5zZm9ybWVyJTIwZGVjb2RlciUwQSUyMyUyMHRoZXNlJTIwYXJlJTIwb2YlMjBzaGFwZSUyMChiYXRjaF9zaXplJTJDJTIwbnVtX3F1ZXJpZXMlMkMlMjBoaWRkZW5fc2l6ZSklMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBbGlzdChsYXN0X2hpZGRlbl9zdGF0ZXMuc2hhcGUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, AutoModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/conditional-detr-resnet-50&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModel.from_pretrained(<span class="hljs-string">&quot;microsoft/conditional-detr-resnet-50&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare image for the model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the last hidden states are the final query embeddings of the Transformer decoder</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># these are of shape (batch_size, num_queries, hidden_size)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">300</span>, <span class="hljs-number">256</span>]`,wrap:!1}}),{c(){d=r("p"),d.textContent=T,m=n(),p(b.$$.fragment)},l(c){d=i(c,"P",{"data-svelte-h":!0}),y(d)!=="svelte-kvfsh7"&&(d.textContent=T),m=s(c),h(b.$$.fragment,c)},m(c,D){l(c,d,D),l(c,m,D),g(b,c,D),v=!0},p:et,i(c){v||(f(b.$$.fragment,c),v=!0)},o(c){u(b.$$.fragment,c),v=!1},d(c){c&&(o(d),o(m)),_(b,c)}}}function Yn(I){let d,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){d=r("p"),d.innerHTML=T},l(m){d=i(m,"P",{"data-svelte-h":!0}),y(d)!=="svelte-fincs2"&&(d.innerHTML=T)},m(m,b){l(m,d,b)},p:et,d(m){m&&o(d)}}}function Kn(I){let d,T="Examples:",m,b,v;return b=new po({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbEZvck9iamVjdERldGVjdGlvbiUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHJlcXVlc3RzJTBBJTBBdXJsJTIwJTNEJTIwJTIyaHR0cCUzQSUyRiUyRmltYWdlcy5jb2NvZGF0YXNldC5vcmclMkZ2YWwyMDE3JTJGMDAwMDAwMDM5NzY5LmpwZyUyMiUwQWltYWdlJTIwJTNEJTIwSW1hZ2Uub3BlbihyZXF1ZXN0cy5nZXQodXJsJTJDJTIwc3RyZWFtJTNEVHJ1ZSkucmF3KSUwQSUwQWltYWdlX3Byb2Nlc3NvciUyMCUzRCUyMEF1dG9JbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGY29uZGl0aW9uYWwtZGV0ci1yZXNuZXQtNTAlMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JPYmplY3REZXRlY3Rpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRmNvbmRpdGlvbmFsLWRldHItcmVzbmV0LTUwJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMGltYWdlX3Byb2Nlc3NvcihpbWFnZXMlM0RpbWFnZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQSUyMyUyMGNvbnZlcnQlMjBvdXRwdXRzJTIwKGJvdW5kaW5nJTIwYm94ZXMlMjBhbmQlMjBjbGFzcyUyMGxvZ2l0cyklMjB0byUyMFBhc2NhbCUyMFZPQyUyMGZvcm1hdCUyMCh4bWluJTJDJTIweW1pbiUyQyUyMHhtYXglMkMlMjB5bWF4KSUwQXRhcmdldF9zaXplcyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUJpbWFnZS5zaXplJTVCJTNBJTNBLTElNUQlNUQpJTBBcmVzdWx0cyUyMCUzRCUyMGltYWdlX3Byb2Nlc3Nvci5wb3N0X3Byb2Nlc3Nfb2JqZWN0X2RldGVjdGlvbihvdXRwdXRzJTJDJTIwdGhyZXNob2xkJTNEMC41JTJDJTIwdGFyZ2V0X3NpemVzJTNEdGFyZ2V0X3NpemVzKSU1QiUwQSUyMCUyMCUyMCUyMDAlMEElNUQlMEFmb3IlMjBzY29yZSUyQyUyMGxhYmVsJTJDJTIwYm94JTIwaW4lMjB6aXAocmVzdWx0cyU1QiUyMnNjb3JlcyUyMiU1RCUyQyUyMHJlc3VsdHMlNUIlMjJsYWJlbHMlMjIlNUQlMkMlMjByZXN1bHRzJTVCJTIyYm94ZXMlMjIlNUQpJTNBJTBBJTIwJTIwJTIwJTIwYm94JTIwJTNEJTIwJTVCcm91bmQoaSUyQyUyMDIpJTIwZm9yJTIwaSUyMGluJTIwYm94LnRvbGlzdCgpJTVEJTBBJTIwJTIwJTIwJTIwcHJpbnQoJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZiUyMkRldGVjdGVkJTIwJTdCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCbGFiZWwuaXRlbSgpJTVEJTdEJTIwd2l0aCUyMGNvbmZpZGVuY2UlMjAlMjIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBmJTIyJTdCcm91bmQoc2NvcmUuaXRlbSgpJTJDJTIwMyklN0QlMjBhdCUyMGxvY2F0aW9uJTIwJTdCYm94JTdEJTIyJTBBJTIwJTIwJTIwJTIwKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, AutoModelForObjectDetection
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/conditional-detr-resnet-50&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForObjectDetection.from_pretrained(<span class="hljs-string">&quot;microsoft/conditional-detr-resnet-50&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_sizes = torch.tensor([image.size[::-<span class="hljs-number">1</span>]])
<span class="hljs-meta">&gt;&gt;&gt; </span>results = image_processor.post_process_object_detection(outputs, threshold=<span class="hljs-number">0.5</span>, target_sizes=target_sizes)[
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>
<span class="hljs-meta">... </span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> score, label, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(results[<span class="hljs-string">&quot;scores&quot;</span>], results[<span class="hljs-string">&quot;labels&quot;</span>], results[<span class="hljs-string">&quot;boxes&quot;</span>]):
<span class="hljs-meta">... </span>    box = [<span class="hljs-built_in">round</span>(i, <span class="hljs-number">2</span>) <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> box.tolist()]
<span class="hljs-meta">... </span>    <span class="hljs-built_in">print</span>(
<span class="hljs-meta">... </span>        <span class="hljs-string">f&quot;Detected <span class="hljs-subst">{model.config.id2label[label.item()]}</span> with confidence &quot;</span>
<span class="hljs-meta">... </span>        <span class="hljs-string">f&quot;<span class="hljs-subst">{<span class="hljs-built_in">round</span>(score.item(), <span class="hljs-number">3</span>)}</span> at location <span class="hljs-subst">{box}</span>&quot;</span>
<span class="hljs-meta">... </span>    )
Detected remote <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.833</span> at location [<span class="hljs-number">38.31</span>, <span class="hljs-number">72.1</span>, <span class="hljs-number">177.63</span>, <span class="hljs-number">118.45</span>]
Detected cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.831</span> at location [<span class="hljs-number">9.2</span>, <span class="hljs-number">51.38</span>, <span class="hljs-number">321.13</span>, <span class="hljs-number">469.0</span>]
Detected cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.804</span> at location [<span class="hljs-number">340.3</span>, <span class="hljs-number">16.85</span>, <span class="hljs-number">642.93</span>, <span class="hljs-number">370.95</span>]
Detected remote <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.683</span> at location [<span class="hljs-number">334.48</span>, <span class="hljs-number">73.49</span>, <span class="hljs-number">366.37</span>, <span class="hljs-number">190.01</span>]
Detected couch <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.535</span> at location [<span class="hljs-number">0.52</span>, <span class="hljs-number">1.19</span>, <span class="hljs-number">640.35</span>, <span class="hljs-number">475.1</span>]`,wrap:!1}}),{c(){d=r("p"),d.textContent=T,m=n(),p(b.$$.fragment)},l(c){d=i(c,"P",{"data-svelte-h":!0}),y(d)!=="svelte-kvfsh7"&&(d.textContent=T),m=s(c),h(b.$$.fragment,c)},m(c,D){l(c,d,D),l(c,m,D),g(b,c,D),v=!0},p:et,i(c){v||(f(b.$$.fragment,c),v=!0)},o(c){u(b.$$.fragment,c),v=!1},d(c){c&&(o(d),o(m)),_(b,c)}}}function es(I){let d,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){d=r("p"),d.innerHTML=T},l(m){d=i(m,"P",{"data-svelte-h":!0}),y(d)!=="svelte-fincs2"&&(d.innerHTML=T)},m(m,b){l(m,d,b)},p:et,d(m){m&&o(d)}}}function ts(I){let d,T="Examples:",m,b,v;return b=new po({props:{code:"aW1wb3J0JTIwaW8lMEFpbXBvcnQlMjByZXF1ZXN0cyUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHRvcmNoJTBBaW1wb3J0JTIwbnVtcHklMEElMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwKCUwQSUyMCUyMCUyMCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUwQSUyMCUyMCUyMCUyMENvbmRpdGlvbmFsRGV0ckNvbmZpZyUyQyUwQSUyMCUyMCUyMCUyMENvbmRpdGlvbmFsRGV0ckZvclNlZ21lbnRhdGlvbiUyQyUwQSklMEFmcm9tJTIwdHJhbnNmb3JtZXJzLmltYWdlX3RyYW5zZm9ybXMlMjBpbXBvcnQlMjByZ2JfdG9faWQlMEElMEF1cmwlMjAlM0QlMjAlMjJodHRwJTNBJTJGJTJGaW1hZ2VzLmNvY29kYXRhc2V0Lm9yZyUyRnZhbDIwMTclMkYwMDAwMDAwMzk3NjkuanBnJTIyJTBBaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBJTBBaW1hZ2VfcHJvY2Vzc29yJTIwJTNEJTIwQXV0b0ltYWdlUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZjb25kaXRpb25hbC1kZXRyLXJlc25ldC01MCUyMiklMEElMEElMjMlMjByYW5kb21seSUyMGluaXRpYWxpemUlMjBhbGwlMjB3ZWlnaHRzJTIwb2YlMjB0aGUlMjBtb2RlbCUwQWNvbmZpZyUyMCUzRCUyMENvbmRpdGlvbmFsRGV0ckNvbmZpZygpJTBBbW9kZWwlMjAlM0QlMjBDb25kaXRpb25hbERldHJGb3JTZWdtZW50YXRpb24oY29uZmlnKSUwQSUwQSUyMyUyMHByZXBhcmUlMjBpbWFnZSUyMGZvciUyMHRoZSUyMG1vZGVsJTBBaW5wdXRzJTIwJTNEJTIwaW1hZ2VfcHJvY2Vzc29yKGltYWdlcyUzRGltYWdlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBmb3J3YXJkJTIwcGFzcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEElMjMlMjBVc2UlMjB0aGUlMjAlNjBwb3N0X3Byb2Nlc3NfcGFub3B0aWNfc2VnbWVudGF0aW9uJTYwJTIwbWV0aG9kJTIwb2YlMjB0aGUlMjAlNjBpbWFnZV9wcm9jZXNzb3IlNjAlMjB0byUyMHJldHJpZXZlJTIwcG9zdC1wcm9jZXNzZWQlMjBwYW5vcHRpYyUyMHNlZ21lbnRhdGlvbiUyMG1hcHMlMEElMjMlMjBTZWdtZW50YXRpb24lMjByZXN1bHRzJTIwYXJlJTIwcmV0dXJuZWQlMjBhcyUyMGElMjBsaXN0JTIwb2YlMjBkaWN0aW9uYXJpZXMlMEFyZXN1bHQlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IucG9zdF9wcm9jZXNzX3Bhbm9wdGljX3NlZ21lbnRhdGlvbihvdXRwdXRzJTJDJTIwdGFyZ2V0X3NpemVzJTNEJTVCKDMwMCUyQyUyMDUwMCklNUQpJTBBJTIzJTIwQSUyMHRlbnNvciUyMG9mJTIwc2hhcGUlMjAoaGVpZ2h0JTJDJTIwd2lkdGgpJTIwd2hlcmUlMjBlYWNoJTIwdmFsdWUlMjBkZW5vdGVzJTIwYSUyMHNlZ21lbnQlMjBpZCUyQyUyMGZpbGxlZCUyMHdpdGglMjAtMSUyMGlmJTIwbm8lMjBzZWdtZW50JTIwaXMlMjBmb3VuZCUwQXBhbm9wdGljX3NlZyUyMCUzRCUyMHJlc3VsdCU1QjAlNUQlNUIlMjJzZWdtZW50YXRpb24lMjIlNUQlMEElMjMlMjBHZXQlMjBwcmVkaWN0aW9uJTIwc2NvcmUlMjBhbmQlMjBzZWdtZW50X2lkJTIwdG8lMjBjbGFzc19pZCUyMG1hcHBpbmclMjBvZiUyMGVhY2glMjBzZWdtZW50JTBBcGFub3B0aWNfc2VnbWVudHNfaW5mbyUyMCUzRCUyMHJlc3VsdCU1QjAlNUQlNUIlMjJzZWdtZW50c19pbmZvJTIyJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> io
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> numpy

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> (
<span class="hljs-meta">... </span>    AutoImageProcessor,
<span class="hljs-meta">... </span>    ConditionalDetrConfig,
<span class="hljs-meta">... </span>    ConditionalDetrForSegmentation,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers.image_transforms <span class="hljs-keyword">import</span> rgb_to_id

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/conditional-detr-resnet-50&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># randomly initialize all weights of the model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>config = ConditionalDetrConfig()
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ConditionalDetrForSegmentation(config)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare image for the model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Use the \`post_process_panoptic_segmentation\` method of the \`image_processor\` to retrieve post-processed panoptic segmentation maps</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Segmentation results are returned as a list of dictionaries</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(<span class="hljs-number">300</span>, <span class="hljs-number">500</span>)])
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># A tensor of shape (height, width) where each value denotes a segment id, filled with -1 if no segment is found</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>panoptic_seg = result[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;segmentation&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Get prediction score and segment_id to class_id mapping of each segment</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>panoptic_segments_info = result[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;segments_info&quot;</span>]`,wrap:!1}}),{c(){d=r("p"),d.textContent=T,m=n(),p(b.$$.fragment)},l(c){d=i(c,"P",{"data-svelte-h":!0}),y(d)!=="svelte-kvfsh7"&&(d.textContent=T),m=s(c),h(b.$$.fragment,c)},m(c,D){l(c,d,D),l(c,m,D),g(b,c,D),v=!0},p:et,i(c){v||(f(b.$$.fragment,c),v=!0)},o(c){u(b.$$.fragment,c),v=!1},d(c){c&&(o(d),o(m)),_(b,c)}}}function os(I){let d,T,m,b,v,c="<em>This model was released on 2021-08-13 and added to Hugging Face Transformers on 2022-09-22.</em>",D,he,Ft,G,dn='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',zt,ge,kt,fe,cn='The Conditional DETR model was proposed in <a href="https://huggingface.co/papers/2108.06152" rel="nofollow">Conditional DETR for Fast Training Convergence</a> by Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang. Conditional DETR presents a conditional cross-attention mechanism for fast DETR training. Conditional DETR converges 6.7× to 10× faster than DETR.',Jt,ue,ln="The abstract from the paper is the following:",Ut,_e,mn='<em>The recently-developed DETR approach applies the transformer encoder and decoder architecture to object detection and achieves promising performance. In this paper, we handle the critical issue, slow training convergence, and present a conditional cross-attention mechanism for fast DETR training. Our approach is motivated by that the cross-attention in DETR relies highly on the content embeddings for localizing the four extremities and predicting the box, which increases the need for high-quality content embeddings and thus the training difficulty. Our approach, named conditional DETR, learns a conditional spatial query from the decoder embedding for decoder multi-head cross-attention. The benefit is that through the conditional spatial query, each cross-attention head is able to attend to a band containing a distinct region, e.g., one object extremity or a region inside the object box. This narrows down the spatial range for localizing the distinct regions for object classification and box regression, thus relaxing the dependence on the content embeddings and easing the training. Empirical results show that conditional DETR converges 6.7× faster for the backbones R50 and R101 and 10× faster for stronger backbones DC5-R50 and DC5-R101. Code is available at <a href="https://github.com/Atten4Vis/ConditionalDETR" rel="nofollow">https://github.com/Atten4Vis/ConditionalDETR</a>.</em>',Nt,O,pn,qt,be,hn='Conditional DETR shows much faster convergence compared to the original DETR. Taken from the <a href="https://huggingface.co/papers/2108.06152">original paper</a>.',Pt,ye,gn='This model was contributed by <a href="https://huggingface.co/DepuMeng" rel="nofollow">DepuMeng</a>. The original code can be found <a href="https://github.com/Atten4Vis/ConditionalDETR" rel="nofollow">here</a>.',Et,ve,Rt,we,fn='<li>Scripts for finetuning <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection">ConditionalDetrForObjectDetection</a> with <a href="/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer">Trainer</a> or <a href="https://huggingface.co/docs/accelerate/index" rel="nofollow">Accelerate</a> can be found <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection" rel="nofollow">here</a>.</li> <li>See also: <a href="../tasks/object_detection">Object detection task guide</a>.</li>',Zt,Te,Wt,J,Ce,ho,tt,un=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrModel">ConditionalDetrModel</a>. It is used to instantiate
a Conditional DETR model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Conditional DETR
<a href="https://huggingface.co/microsoft/conditional-detr-resnet-50" rel="nofollow">microsoft/conditional-detr-resnet-50</a> architecture.`,go,ot,_n=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,fo,H,Bt,xe,St,E,De,uo,nt,bn="Constructs a Conditional Detr image processor.",_o,V,Me,bo,st,yn="Preprocess an image or a batch of images so that it can be used by the model.",Lt,je,Gt,M,$e,yo,at,vn="Constructs a fast Conditional Detr image processor.",vo,rt,Ie,wo,X,Fe,To,it,wn=`Converts the raw output of <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection">ConditionalDetrForObjectDetection</a> into final bounding boxes in (top_left_x,
top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.`,Co,A,ze,xo,dt,Tn='Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a> into instance segmentation predictions. Only supports PyTorch.',Do,Q,ke,Mo,ct,Cn='Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a> into semantic segmentation maps. Only supports PyTorch.',jo,Y,Je,$o,lt,xn=`Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a> into image panoptic segmentation predictions. Only supports
PyTorch.`,Ot,Ue,Ht,j,Ne,Io,K,qe,Fo,mt,Dn="Preprocess an image or a batch of images.",zo,ee,Pe,ko,pt,Mn=`Converts the raw output of <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection">ConditionalDetrForObjectDetection</a> into final bounding boxes in (top_left_x,
top_left_y, bottom_right_x, bottom_right_y) format. Only supports PyTorch.`,Jo,te,Ee,Uo,ht,jn='Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a> into instance segmentation predictions. Only supports PyTorch.',No,oe,Re,qo,gt,$n='Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a> into semantic segmentation maps. Only supports PyTorch.',Po,ne,Ze,Eo,ft,In=`Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a> into image panoptic segmentation predictions. Only supports
PyTorch.`,Vt,We,Xt,F,Be,Ro,ut,Fn=`The bare Conditional DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
hidden-states without any specific head on top.`,Zo,_t,zn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Wo,bt,kn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bo,U,Se,So,yt,Jn='The <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrModel">ConditionalDetrModel</a> forward method, overrides the <code>__call__</code> special method.',Lo,se,Go,ae,At,Le,Qt,z,Ge,Oo,vt,Un=`Conditional DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
top, for tasks such as COCO detection.`,Ho,wt,Nn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Vo,Tt,qn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xo,N,Oe,Ao,Ct,Pn='The <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForObjectDetection">ConditionalDetrForObjectDetection</a> forward method, overrides the <code>__call__</code> special method.',Qo,re,Yo,ie,Yt,He,Kt,k,Ve,Ko,xt,En=`Conditional DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top,
for tasks such as COCO panoptic.`,en,Dt,Rn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,tn,Mt,Zn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,on,q,Xe,nn,jt,Wn='The <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a> forward method, overrides the <code>__call__</code> special method.',sn,de,an,ce,eo,Ae,to,$t,oo;return he=new S({props:{title:"Conditional DETR",local:"conditional-detr",headingTag:"h1"}}),ge=new S({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ve=new S({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Te=new S({props:{title:"ConditionalDetrConfig",local:"transformers.ConditionalDetrConfig",headingTag:"h2"}}),Ce=new x({props:{name:"class transformers.ConditionalDetrConfig",anchor:"transformers.ConditionalDetrConfig",parameters:[{name:"use_timm_backbone",val:" = True"},{name:"backbone_config",val:" = None"},{name:"num_channels",val:" = 3"},{name:"num_queries",val:" = 300"},{name:"encoder_layers",val:" = 6"},{name:"encoder_ffn_dim",val:" = 2048"},{name:"encoder_attention_heads",val:" = 8"},{name:"decoder_layers",val:" = 6"},{name:"decoder_ffn_dim",val:" = 2048"},{name:"decoder_attention_heads",val:" = 8"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'relu'"},{name:"d_model",val:" = 256"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"init_xavier_std",val:" = 1.0"},{name:"auxiliary_loss",val:" = False"},{name:"position_embedding_type",val:" = 'sine'"},{name:"backbone",val:" = 'resnet50'"},{name:"use_pretrained_backbone",val:" = True"},{name:"backbone_kwargs",val:" = None"},{name:"dilation",val:" = False"},{name:"class_cost",val:" = 2"},{name:"bbox_cost",val:" = 5"},{name:"giou_cost",val:" = 2"},{name:"mask_loss_coefficient",val:" = 1"},{name:"dice_loss_coefficient",val:" = 1"},{name:"cls_loss_coefficient",val:" = 2"},{name:"bbox_loss_coefficient",val:" = 5"},{name:"giou_loss_coefficient",val:" = 2"},{name:"focal_alpha",val:" = 0.25"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ConditionalDetrConfig.use_timm_backbone",description:`<strong>use_timm_backbone</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to use the <code>timm</code> library for the backbone. If set to <code>False</code>, will use the <a href="/docs/transformers/v4.56.2/en/main_classes/backbones#transformers.AutoBackbone">AutoBackbone</a>
API.`,name:"use_timm_backbone"},{anchor:"transformers.ConditionalDetrConfig.backbone_config",description:`<strong>backbone_config</strong> (<code>PretrainedConfig</code> or <code>dict</code>, <em>optional</em>) &#x2014;
The configuration of the backbone model. Only used in case <code>use_timm_backbone</code> is set to <code>False</code> in which
case it will default to <code>ResNetConfig()</code>.`,name:"backbone_config"},{anchor:"transformers.ConditionalDetrConfig.num_channels",description:`<strong>num_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The number of input channels.`,name:"num_channels"},{anchor:"transformers.ConditionalDetrConfig.num_queries",description:`<strong>num_queries</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
Number of object queries, i.e. detection slots. This is the maximal number of objects
<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrModel">ConditionalDetrModel</a> can detect in a single image. For COCO, we recommend 100 queries.`,name:"num_queries"},{anchor:"transformers.ConditionalDetrConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
This parameter is a general dimension parameter, defining dimensions for components such as the encoder layer and projection parameters in the decoder layer, among others.`,name:"d_model"},{anchor:"transformers.ConditionalDetrConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.ConditionalDetrConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.ConditionalDetrConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.ConditionalDetrConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.ConditionalDetrConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.ConditionalDetrConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.ConditionalDetrConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.ConditionalDetrConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.ConditionalDetrConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.ConditionalDetrConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.ConditionalDetrConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.ConditionalDetrConfig.init_xavier_std",description:`<strong>init_xavier_std</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
The scaling factor used for the Xavier initialization gain in the HM Attention map module.`,name:"init_xavier_std"},{anchor:"transformers.ConditionalDetrConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.ConditionalDetrConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.ConditionalDetrConfig.auxiliary_loss",description:`<strong>auxiliary_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether auxiliary decoding losses (loss at each decoder layer) are to be used.`,name:"auxiliary_loss"},{anchor:"transformers.ConditionalDetrConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;sine&quot;</code>) &#x2014;
Type of position embeddings to be used on top of the image features. One of <code>&quot;sine&quot;</code> or <code>&quot;learned&quot;</code>.`,name:"position_embedding_type"},{anchor:"transformers.ConditionalDetrConfig.backbone",description:`<strong>backbone</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;resnet50&quot;</code>) &#x2014;
Name of backbone to use when <code>backbone_config</code> is <code>None</code>. If <code>use_pretrained_backbone</code> is <code>True</code>, this
will load the corresponding pretrained weights from the timm or transformers library. If <code>use_pretrained_backbone</code>
is <code>False</code>, this loads the backbone&#x2019;s config and uses that to initialize the backbone with random weights.`,name:"backbone"},{anchor:"transformers.ConditionalDetrConfig.use_pretrained_backbone",description:`<strong>use_pretrained_backbone</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use pretrained weights for the backbone.`,name:"use_pretrained_backbone"},{anchor:"transformers.ConditionalDetrConfig.backbone_kwargs",description:`<strong>backbone_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
e.g. <code>{&apos;out_indices&apos;: (0, 1, 2, 3)}</code>. Cannot be specified if <code>backbone_config</code> is set.`,name:"backbone_kwargs"},{anchor:"transformers.ConditionalDetrConfig.dilation",description:`<strong>dilation</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
<code>use_timm_backbone</code> = <code>True</code>.`,name:"dilation"},{anchor:"transformers.ConditionalDetrConfig.class_cost",description:`<strong>class_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
Relative weight of the classification error in the Hungarian matching cost.`,name:"class_cost"},{anchor:"transformers.ConditionalDetrConfig.bbox_cost",description:`<strong>bbox_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 5) &#x2014;
Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.`,name:"bbox_cost"},{anchor:"transformers.ConditionalDetrConfig.giou_cost",description:`<strong>giou_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 2) &#x2014;
Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.`,name:"giou_cost"},{anchor:"transformers.ConditionalDetrConfig.mask_loss_coefficient",description:`<strong>mask_loss_coefficient</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
Relative weight of the Focal loss in the panoptic segmentation loss.`,name:"mask_loss_coefficient"},{anchor:"transformers.ConditionalDetrConfig.dice_loss_coefficient",description:`<strong>dice_loss_coefficient</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.`,name:"dice_loss_coefficient"},{anchor:"transformers.ConditionalDetrConfig.bbox_loss_coefficient",description:`<strong>bbox_loss_coefficient</strong> (<code>float</code>, <em>optional</em>, defaults to 5) &#x2014;
Relative weight of the L1 bounding box loss in the object detection loss.`,name:"bbox_loss_coefficient"},{anchor:"transformers.ConditionalDetrConfig.giou_loss_coefficient",description:`<strong>giou_loss_coefficient</strong> (<code>float</code>, <em>optional</em>, defaults to 2) &#x2014;
Relative weight of the generalized IoU loss in the object detection loss.`,name:"giou_loss_coefficient"},{anchor:"transformers.ConditionalDetrConfig.eos_coefficient",description:`<strong>eos_coefficient</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Relative classification weight of the &#x2018;no-object&#x2019; class in the object detection loss.`,name:"eos_coefficient"},{anchor:"transformers.ConditionalDetrConfig.focal_alpha",description:`<strong>focal_alpha</strong> (<code>float</code>, <em>optional</em>, defaults to 0.25) &#x2014;
Alpha parameter in the focal loss.`,name:"focal_alpha"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/configuration_conditional_detr.py#L32"}}),H=new mo({props:{anchor:"transformers.ConditionalDetrConfig.example",$$slots:{default:[Xn]},$$scope:{ctx:I}}}),xe=new S({props:{title:"ConditionalDetrImageProcessor",local:"transformers.ConditionalDetrImageProcessor",headingTag:"h2"}}),De=new x({props:{name:"class transformers.ConditionalDetrImageProcessor",anchor:"transformers.ConditionalDetrImageProcessor",parameters:[{name:"format",val:": typing.Union[str, transformers.image_utils.AnnotationFormat] = <AnnotationFormat.COCO_DETECTION: 'coco_detection'>"},{name:"do_resize",val:": bool = True"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"resample",val:": Resampling = <Resampling.BILINEAR: 2>"},{name:"do_rescale",val:": bool = True"},{name:"rescale_factor",val:": typing.Union[int, float] = 0.00392156862745098"},{name:"do_normalize",val:": bool = True"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"do_convert_annotations",val:": typing.Optional[bool] = None"},{name:"do_pad",val:": bool = True"},{name:"pad_size",val:": typing.Optional[dict[str, int]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ConditionalDetrImageProcessor.format",description:`<strong>format</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;coco_detection&quot;</code>) &#x2014;
Data format of the annotations. One of &#x201C;coco_detection&#x201D; or &#x201C;coco_panoptic&#x201D;.`,name:"format"},{anchor:"transformers.ConditionalDetrImageProcessor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Controls whether to resize the image&#x2019;s (height, width) dimensions to the specified <code>size</code>. Can be
overridden by the <code>do_resize</code> parameter in the <code>preprocess</code> method.`,name:"do_resize"},{anchor:"transformers.ConditionalDetrImageProcessor.size",description:`<strong>size</strong> (<code>dict[str, int]</code> <em>optional</em>, defaults to <code>{&quot;shortest_edge&quot; -- 800, &quot;longest_edge&quot;: 1333}</code>):
Size of the image&#x2019;s <code>(height, width)</code> dimensions after resizing. Can be overridden by the <code>size</code> parameter
in the <code>preprocess</code> method. Available options are:<ul>
<li><code>{&quot;height&quot;: int, &quot;width&quot;: int}</code>: The image will be resized to the exact size <code>(height, width)</code>.
Do NOT keep the aspect ratio.</li>
<li><code>{&quot;shortest_edge&quot;: int, &quot;longest_edge&quot;: int}</code>: The image will be resized to a maximum size respecting
the aspect ratio and keeping the shortest edge less or equal to <code>shortest_edge</code> and the longest edge
less or equal to <code>longest_edge</code>.</li>
<li><code>{&quot;max_height&quot;: int, &quot;max_width&quot;: int}</code>: The image will be resized to the maximum size respecting the
aspect ratio and keeping the height less or equal to <code>max_height</code> and the width less or equal to
<code>max_width</code>.</li>
</ul>`,name:"size"},{anchor:"transformers.ConditionalDetrImageProcessor.resample",description:`<strong>resample</strong> (<code>PILImageResampling</code>, <em>optional</em>, defaults to <code>PILImageResampling.BILINEAR</code>) &#x2014;
Resampling filter to use if resizing the image.`,name:"resample"},{anchor:"transformers.ConditionalDetrImageProcessor.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Controls whether to rescale the image by the specified scale <code>rescale_factor</code>. Can be overridden by the
<code>do_rescale</code> parameter in the <code>preprocess</code> method.`,name:"do_rescale"},{anchor:"transformers.ConditionalDetrImageProcessor.rescale_factor",description:`<strong>rescale_factor</strong> (<code>int</code> or <code>float</code>, <em>optional</em>, defaults to <code>1/255</code>) &#x2014;
Scale factor to use if rescaling the image. Can be overridden by the <code>rescale_factor</code> parameter in the
<code>preprocess</code> method.`,name:"rescale_factor"},{anchor:"transformers.ConditionalDetrImageProcessor.do_normalize",description:`<strong>do_normalize</strong> &#x2014;
Controls whether to normalize the image. Can be overridden by the <code>do_normalize</code> parameter in the
<code>preprocess</code> method.`,name:"do_normalize"},{anchor:"transformers.ConditionalDetrImageProcessor.image_mean",description:`<strong>image_mean</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>IMAGENET_DEFAULT_MEAN</code>) &#x2014;
Mean values to use when normalizing the image. Can be a single value or a list of values, one for each
channel. Can be overridden by the <code>image_mean</code> parameter in the <code>preprocess</code> method.`,name:"image_mean"},{anchor:"transformers.ConditionalDetrImageProcessor.image_std",description:`<strong>image_std</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to <code>IMAGENET_DEFAULT_STD</code>) &#x2014;
Standard deviation values to use when normalizing the image. Can be a single value or a list of values, one
for each channel. Can be overridden by the <code>image_std</code> parameter in the <code>preprocess</code> method.`,name:"image_std"},{anchor:"transformers.ConditionalDetrImageProcessor.do_convert_annotations",description:`<strong>do_convert_annotations</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Controls whether to convert the annotations to the format expected by the DETR model. Converts the
bounding boxes to the format <code>(center_x, center_y, width, height)</code> and in the range <code>[0, 1]</code>.
Can be overridden by the <code>do_convert_annotations</code> parameter in the <code>preprocess</code> method.`,name:"do_convert_annotations"},{anchor:"transformers.ConditionalDetrImageProcessor.do_pad",description:`<strong>do_pad</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Controls whether to pad the image. Can be overridden by the <code>do_pad</code> parameter in the <code>preprocess</code>
method. If <code>True</code>, padding will be applied to the bottom and right of the image with zeros.
If <code>pad_size</code> is provided, the image will be padded to the specified dimensions.
Otherwise, the image will be padded to the maximum height and width of the batch.`,name:"do_pad"},{anchor:"transformers.ConditionalDetrImageProcessor.pad_size",description:`<strong>pad_size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
The size <code>{&quot;height&quot;: int, &quot;width&quot; int}</code> to pad the images to. Must be larger than any image size
provided for preprocessing. If <code>pad_size</code> is not provided, images will be padded to the largest
height and width in the batch.`,name:"pad_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L807"}}),Me=new x({props:{name:"preprocess",anchor:"transformers.ConditionalDetrImageProcessor.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"annotations",val:": typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None"},{name:"return_segmentation_masks",val:": typing.Optional[bool] = None"},{name:"masks_path",val:": typing.Union[str, pathlib.Path, NoneType] = None"},{name:"do_resize",val:": typing.Optional[bool] = None"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"resample",val:" = None"},{name:"do_rescale",val:": typing.Optional[bool] = None"},{name:"rescale_factor",val:": typing.Union[int, float, NoneType] = None"},{name:"do_normalize",val:": typing.Optional[bool] = None"},{name:"do_convert_annotations",val:": typing.Optional[bool] = None"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"do_pad",val:": typing.Optional[bool] = None"},{name:"format",val:": typing.Union[str, transformers.image_utils.AnnotationFormat, NoneType] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels_first'>"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"pad_size",val:": typing.Optional[dict[str, int]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.images",description:`<strong>images</strong> (<code>ImageInput</code>) &#x2014;
Image or batch of images to preprocess. Expects a single or batch of images with pixel values ranging
from 0 to 255. If passing in images with pixel values between 0 and 1, set <code>do_rescale=False</code>.`,name:"images"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.annotations",description:`<strong>annotations</strong> (<code>AnnotationType</code> or <code>list[AnnotationType]</code>, <em>optional</em>) &#x2014;
List of annotations associated with the image or batch of images. If annotation is for object
detection, the annotations should be a dictionary with the following keys:<ul>
<li>&#x201C;image_id&#x201D; (<code>int</code>): The image id.</li>
<li>&#x201C;annotations&#x201D; (<code>list[Dict]</code>): List of annotations for an image. Each annotation should be a
dictionary. An image can have no annotations, in which case the list should be empty.
If annotation is for segmentation, the annotations should be a dictionary with the following keys:</li>
<li>&#x201C;image_id&#x201D; (<code>int</code>): The image id.</li>
<li>&#x201C;segments_info&#x201D; (<code>list[Dict]</code>): List of segments for an image. Each segment should be a dictionary.
An image can have no segments, in which case the list should be empty.</li>
<li>&#x201C;file_name&#x201D; (<code>str</code>): The file name of the image.</li>
</ul>`,name:"annotations"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.return_segmentation_masks",description:`<strong>return_segmentation_masks</strong> (<code>bool</code>, <em>optional</em>, defaults to self.return_segmentation_masks) &#x2014;
Whether to return segmentation masks.`,name:"return_segmentation_masks"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.masks_path",description:`<strong>masks_path</strong> (<code>str</code> or <code>pathlib.Path</code>, <em>optional</em>) &#x2014;
Path to the directory containing the segmentation masks.`,name:"masks_path"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to self.do_resize) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>, defaults to self.size) &#x2014;
Size of the image&#x2019;s <code>(height, width)</code> dimensions after resizing. Available options are:<ul>
<li><code>{&quot;height&quot;: int, &quot;width&quot;: int}</code>: The image will be resized to the exact size <code>(height, width)</code>.
Do NOT keep the aspect ratio.</li>
<li><code>{&quot;shortest_edge&quot;: int, &quot;longest_edge&quot;: int}</code>: The image will be resized to a maximum size respecting
the aspect ratio and keeping the shortest edge less or equal to <code>shortest_edge</code> and the longest edge
less or equal to <code>longest_edge</code>.</li>
<li><code>{&quot;max_height&quot;: int, &quot;max_width&quot;: int}</code>: The image will be resized to the maximum size respecting the
aspect ratio and keeping the height less or equal to <code>max_height</code> and the width less or equal to
<code>max_width</code>.</li>
</ul>`,name:"size"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.resample",description:`<strong>resample</strong> (<code>PILImageResampling</code>, <em>optional</em>, defaults to self.resample) &#x2014;
Resampling filter to use when resizing the image.`,name:"resample"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to self.do_rescale) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>float</code>, <em>optional</em>, defaults to self.rescale_factor) &#x2014;
Rescale factor to use when rescaling the image.`,name:"rescale_factor"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to self.do_normalize) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.do_convert_annotations",description:`<strong>do_convert_annotations</strong> (<code>bool</code>, <em>optional</em>, defaults to self.do_convert_annotations) &#x2014;
Whether to convert the annotations to the format expected by the model. Converts the bounding
boxes from the format <code>(top_left_x, top_left_y, width, height)</code> to <code>(center_x, center_y, width, height)</code>
and in relative coordinates.`,name:"do_convert_annotations"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to self.image_mean) &#x2014;
Mean to use when normalizing the image.`,name:"image_mean"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.image_std",description:`<strong>image_std</strong> (<code>float</code> or <code>list[float]</code>, <em>optional</em>, defaults to self.image_std) &#x2014;
Standard deviation to use when normalizing the image.`,name:"image_std"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.do_pad",description:`<strong>do_pad</strong> (<code>bool</code>, <em>optional</em>, defaults to self.do_pad) &#x2014;
Whether to pad the image. If <code>True</code>, padding will be applied to the bottom and right of
the image with zeros. If <code>pad_size</code> is provided, the image will be padded to the specified
dimensions. Otherwise, the image will be padded to the maximum height and width of the batch.`,name:"do_pad"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.format",description:`<strong>format</strong> (<code>str</code> or <code>AnnotationFormat</code>, <em>optional</em>, defaults to self.format) &#x2014;
Format of the annotations.`,name:"format"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <code>TensorType</code>, <em>optional</em>, defaults to self.return_tensors) &#x2014;
Type of tensors to return. If <code>None</code>, will return the list of images.`,name:"return_tensors"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.data_format",description:`<strong>data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>, defaults to <code>ChannelDimension.FIRST</code>) &#x2014;
The channel dimension format for the output image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li>Unset: Use the channel dimension format of the input image.</li>
</ul>`,name:"data_format"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>, <em>optional</em>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"},{anchor:"transformers.ConditionalDetrImageProcessor.preprocess.pad_size",description:`<strong>pad_size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
The size <code>{&quot;height&quot;: int, &quot;width&quot; int}</code> to pad the images to. Must be larger than any image size
provided for preprocessing. If <code>pad_size</code> is not provided, images will be padded to the largest
height and width in the batch.`,name:"pad_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1266"}}),je=new S({props:{title:"ConditionalDetrImageProcessorFast",local:"transformers.ConditionalDetrImageProcessorFast",headingTag:"h2"}}),$e=new x({props:{name:"class transformers.ConditionalDetrImageProcessorFast",anchor:"transformers.ConditionalDetrImageProcessorFast",parameters:[{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.conditional_detr.image_processing_conditional_detr_fast.ConditionalDetrFastImageProcessorKwargs]"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr_fast.py#L299"}}),Ie=new x({props:{name:"preprocess",anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"annotations",val:": typing.Union[dict[str, typing.Union[int, str, list[dict]]], list[dict[str, typing.Union[int, str, list[dict]]]], NoneType] = None"},{name:"masks_path",val:": typing.Union[str, pathlib.Path, NoneType] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.conditional_detr.image_processing_conditional_detr_fast.ConditionalDetrFastImageProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.images",description:`<strong>images</strong> (<code>Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[&apos;PIL.Image.Image&apos;], list[numpy.ndarray], list[&apos;torch.Tensor&apos;]]</code>) &#x2014;
Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
passing in images with pixel values between 0 and 1, set <code>do_rescale=False</code>.`,name:"images"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.annotations",description:`<strong>annotations</strong> (<code>AnnotationType</code> or <code>list[AnnotationType]</code>, <em>optional</em>) &#x2014;
List of annotations associated with the image or batch of images. If annotation is for object
detection, the annotations should be a dictionary with the following keys:<ul>
<li>&#x201C;image_id&#x201D; (<code>int</code>): The image id.</li>
<li>&#x201C;annotations&#x201D; (<code>list[Dict]</code>): List of annotations for an image. Each annotation should be a
dictionary. An image can have no annotations, in which case the list should be empty.
If annotation is for segmentation, the annotations should be a dictionary with the following keys:</li>
<li>&#x201C;image_id&#x201D; (<code>int</code>): The image id.</li>
<li>&#x201C;segments_info&#x201D; (<code>list[Dict]</code>): List of segments for an image. Each segment should be a dictionary.
An image can have no segments, in which case the list should be empty.</li>
<li>&#x201C;file_name&#x201D; (<code>str</code>): The file name of the image.</li>
</ul>`,name:"annotations"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.masks_path",description:`<strong>masks_path</strong> (<code>str</code> or <code>pathlib.Path</code>, <em>optional</em>) &#x2014;
Path to the directory containing the segmentation masks.`,name:"masks_path"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
Describes the maximum input dimensions to the model.`,name:"size"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.default_to_square",description:`<strong>default_to_square</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to default to a square image when resizing, if size is an int.`,name:"default_to_square"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.resample",description:`<strong>resample</strong> (<code>Union[PILImageResampling, F.InterpolationMode, NoneType]</code>) &#x2014;
Resampling filter to use if resizing the image. This can be one of the enum <code>PILImageResampling</code>. Only
has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.do_center_crop",description:`<strong>do_center_crop</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to center crop the image.`,name:"do_center_crop"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.crop_size",description:`<strong>crop_size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
Size of the output image after applying <code>center_crop</code>.`,name:"crop_size"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>Union[int, float, NoneType]</code>) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>Union[float, list[float], NoneType]</code>) &#x2014;
Image mean to use for normalization. Only has an effect if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_mean"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.image_std",description:`<strong>image_std</strong> (<code>Union[float, list[float], NoneType]</code>) &#x2014;
Image standard deviation to use for normalization. Only has an effect if <code>do_normalize</code> is set to
<code>True</code>.`,name:"image_std"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.return_tensors",description:"<strong>return_tensors</strong> (<code>Union[str, ~utils.generic.TensorType, NoneType]</code>) &#x2014;\nReturns stacked tensors if set to `pt, otherwise returns a list of tensors.",name:"return_tensors"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.data_format",description:`<strong>data_format</strong> (<code>~image_utils.ChannelDimension</code>, <em>optional</em>) &#x2014;
Only <code>ChannelDimension.FIRST</code> is supported. Added for compatibility with slow processors.`,name:"data_format"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>Union[str, ~image_utils.ChannelDimension, NoneType]</code>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.device",description:`<strong>device</strong> (<code>torch.device</code>, <em>optional</em>) &#x2014;
The device to process the images on. If unset, the device is inferred from the input images.`,name:"device"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.disable_grouping",description:`<strong>disable_grouping</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to disable grouping of images by size to process them individually and not in batches.
If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
empirical observations, as detailed here: <a href="https://github.com/huggingface/transformers/pull/38157" rel="nofollow">https://github.com/huggingface/transformers/pull/38157</a>`,name:"disable_grouping"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.format",description:`<strong>format</strong> (<code>str</code>, <em>optional</em>, defaults to <code>AnnotationFormat.COCO_DETECTION</code>) &#x2014;
Data format of the annotations. One of &#x201C;coco_detection&#x201D; or &#x201C;coco_panoptic&#x201D;.`,name:"format"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.do_convert_annotations",description:`<strong>do_convert_annotations</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Controls whether to convert the annotations to the format expected by the CONDITIONAL_DETR model. Converts the
bounding boxes to the format <code>(center_x, center_y, width, height)</code> and in the range <code>[0, 1]</code>.
Can be overridden by the <code>do_convert_annotations</code> parameter in the <code>preprocess</code> method.`,name:"do_convert_annotations"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.do_pad",description:`<strong>do_pad</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Controls whether to pad the image. Can be overridden by the <code>do_pad</code> parameter in the <code>preprocess</code>
method. If <code>True</code>, padding will be applied to the bottom and right of the image with zeros.
If <code>pad_size</code> is provided, the image will be padded to the specified dimensions.
Otherwise, the image will be padded to the maximum height and width of the batch.`,name:"do_pad"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.pad_size",description:`<strong>pad_size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
The size <code>{&quot;height&quot;: int, &quot;width&quot; int}</code> to pad the images to. Must be larger than any image size
provided for preprocessing. If <code>pad_size</code> is not provided, images will be padded to the largest
height and width in the batch.`,name:"pad_size"},{anchor:"transformers.ConditionalDetrImageProcessorFast.preprocess.return_segmentation_masks",description:`<strong>return_segmentation_masks</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to return segmentation masks.`,name:"return_segmentation_masks"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr_fast.py#L577",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<ul>
<li><strong>data</strong> (<code>dict</code>) — Dictionary of lists/arrays/tensors returned by the <strong>call</strong> method (‘pixel_values’, etc.).</li>
<li><strong>tensor_type</strong> (<code>Union[None, str, TensorType]</code>, <em>optional</em>) — You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
initialization.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>&lt;class 'transformers.image_processing_base.BatchFeature'&gt;</code></p>
`}}),Fe=new x({props:{name:"post_process_object_detection",anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_object_detection",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"target_sizes",val:": typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None"},{name:"top_k",val:": int = 100"}],parametersDescription:[{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_object_detection.outputs",description:`<strong>outputs</strong> (<code>ConditionalDetrObjectDetectionOutput</code>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_object_detection.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Score threshold to keep object detection predictions.`,name:"threshold"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_object_detection.target_sizes",description:`<strong>target_sizes</strong> (<code>torch.Tensor</code> or <code>list[tuple[int, int]]</code>, <em>optional</em>) &#x2014;
Tensor of shape <code>(batch_size, 2)</code> or list of tuples (<code>tuple[int, int]</code>) containing the target size
(height, width) of each image in the batch. If left to None, predictions will not be resized.`,name:"target_sizes"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_object_detection.top_k",description:`<strong>top_k</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
Keep only top k bounding boxes before filtering by thresholding.`,name:"top_k"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr_fast.py#L777",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),ze=new x({props:{name:"post_process_instance_segmentation",anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_instance_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"},{name:"return_coco_annotation",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_instance_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_instance_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_instance_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_instance_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_instance_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction. If unset, predictions will not be resized.`,name:"target_sizes"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_instance_segmentation.return_coco_annotation",description:`<strong>return_coco_annotation</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Defaults to <code>False</code>. If set to <code>True</code>, segmentation maps are returned in COCO run-length encoding (RLE)
format.`,name:"return_coco_annotation"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr_fast.py#L883",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — A tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code> or
<code>list[List]</code> run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
<code>True</code>. Set to <code>None</code> if no mask if found above <code>threshold</code>.</li>
<li><strong>segments_info</strong> — A dictionary that contains additional information on each segment.<ul>
<li><strong>id</strong> — An integer representing the <code>segment_id</code>.</li>
<li><strong>label_id</strong> — An integer representing the label / semantic class id corresponding to <code>segment_id</code>.</li>
<li><strong>score</strong> — Prediction score of segment with <code>segment_id</code>.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),ke=new x({props:{name:"post_process_semantic_segmentation",anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_semantic_segmentation",parameters:[{name:"outputs",val:""},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_semantic_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_semantic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[tuple[int, int]]</code>, <em>optional</em>) &#x2014;
A list of tuples (<code>tuple[int, int]</code>) containing the target size (height, width) of each image in the
batch. If unset, predictions will not be resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr_fast.py#L836",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of length <code>batch_size</code>, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target_sizes entry (if <code>target_sizes</code> is specified). Each entry of each
<code>torch.Tensor</code> correspond to a semantic class id.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[torch.Tensor]</code></p>
`}}),Je=new x({props:{name:"post_process_panoptic_segmentation",anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_panoptic_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"label_ids_to_fuse",val:": typing.Optional[set[int]] = None"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_panoptic_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a>) &#x2014;
The outputs from <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a>.`,name:"outputs"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_panoptic_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_panoptic_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_panoptic_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_panoptic_segmentation.label_ids_to_fuse",description:`<strong>label_ids_to_fuse</strong> (<code>Set[int]</code>, <em>optional</em>) &#x2014;
The labels in this state will have all their instances be fused together. For instance we could say
there can only be one sky in an image, but several persons, so the label ID for sky would be in that
set, but not the one for person.`,name:"label_ids_to_fuse"},{anchor:"transformers.ConditionalDetrImageProcessorFast.post_process_panoptic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction in batch. If unset, predictions will not be resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr_fast.py#L966",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — a tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code> or
<code>None</code> if no mask if found above <code>threshold</code>. If <code>target_sizes</code> is specified, segmentation is resized to
the corresponding <code>target_sizes</code> entry.</li>
<li><strong>segments_info</strong> — A dictionary that contains additional information on each segment.<ul>
<li><strong>id</strong> — an integer representing the <code>segment_id</code>.</li>
<li><strong>label_id</strong> — An integer representing the label / semantic class id corresponding to <code>segment_id</code>.</li>
<li><strong>was_fused</strong> — a boolean, <code>True</code> if <code>label_id</code> was in <code>label_ids_to_fuse</code>, <code>False</code> otherwise.
Multiple instances of the same class / label were fused and assigned a single <code>segment_id</code>.</li>
<li><strong>score</strong> — Prediction score of segment with <code>segment_id</code>.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),Ue=new S({props:{title:"ConditionalDetrFeatureExtractor",local:"transformers.ConditionalDetrFeatureExtractor",headingTag:"h2"}}),Ne=new x({props:{name:"class transformers.ConditionalDetrFeatureExtractor",anchor:"transformers.ConditionalDetrFeatureExtractor",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/feature_extraction_conditional_detr.py#L38"}}),qe=new x({props:{name:"__call__",anchor:"transformers.ConditionalDetrFeatureExtractor.__call__",parameters:[{name:"images",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49"}}),Pe=new x({props:{name:"post_process_object_detection",anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_object_detection",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"target_sizes",val:": typing.Union[transformers.utils.generic.TensorType, list[tuple]] = None"},{name:"top_k",val:": int = 100"}],parametersDescription:[{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_object_detection.outputs",description:`<strong>outputs</strong> (<code>DetrObjectDetectionOutput</code>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_object_detection.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Score threshold to keep object detection predictions.`,name:"threshold"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_object_detection.target_sizes",description:`<strong>target_sizes</strong> (<code>torch.Tensor</code> or <code>list[tuple[int, int]]</code>, <em>optional</em>) &#x2014;
Tensor of shape <code>(batch_size, 2)</code> or list of tuples (<code>tuple[int, int]</code>) containing the target size
(height, width) of each image in the batch. If left to None, predictions will not be resized.`,name:"target_sizes"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_object_detection.top_k",description:`<strong>top_k</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
Keep only top k bounding boxes before filtering by thresholding.`,name:"top_k"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1577",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
in the batch as predicted by the model.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),Ee=new x({props:{name:"post_process_instance_segmentation",anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_instance_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"},{name:"return_coco_annotation",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_instance_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_instance_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_instance_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_instance_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_instance_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction. If unset, predictions will not be resized.`,name:"target_sizes"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_instance_segmentation.return_coco_annotation",description:`<strong>return_coco_annotation</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Defaults to <code>False</code>. If set to <code>True</code>, segmentation maps are returned in COCO run-length encoding (RLE)
format.`,name:"return_coco_annotation"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1685",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — A tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code> or
<code>list[List]</code> run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
<code>True</code>. Set to <code>None</code> if no mask if found above <code>threshold</code>.</li>
<li><strong>segments_info</strong> — A dictionary that contains additional information on each segment.<ul>
<li><strong>id</strong> — An integer representing the <code>segment_id</code>.</li>
<li><strong>label_id</strong> — An integer representing the label / semantic class id corresponding to <code>segment_id</code>.</li>
<li><strong>score</strong> — Prediction score of segment with <code>segment_id</code>.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),Re=new x({props:{name:"post_process_semantic_segmentation",anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_semantic_segmentation",parameters:[{name:"outputs",val:""},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_semantic_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_semantic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[tuple[int, int]]</code>, <em>optional</em>) &#x2014;
A list of tuples (<code>tuple[int, int]</code>) containing the target size (height, width) of each image in the
batch. If unset, predictions will not be resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1637",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of length <code>batch_size</code>, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target_sizes entry (if <code>target_sizes</code> is specified). Each entry of each
<code>torch.Tensor</code> correspond to a semantic class id.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[torch.Tensor]</code></p>
`}}),Ze=new x({props:{name:"post_process_panoptic_segmentation",anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_panoptic_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"label_ids_to_fuse",val:": typing.Optional[set[int]] = None"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_panoptic_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a>) &#x2014;
The outputs from <a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrForSegmentation">ConditionalDetrForSegmentation</a>.`,name:"outputs"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_panoptic_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_panoptic_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_panoptic_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_panoptic_segmentation.label_ids_to_fuse",description:`<strong>label_ids_to_fuse</strong> (<code>Set[int]</code>, <em>optional</em>) &#x2014;
The labels in this state will have all their instances be fused together. For instance we could say
there can only be one sky in an image, but several persons, so the label ID for sky would be in that
set, but not the one for person.`,name:"label_ids_to_fuse"},{anchor:"transformers.ConditionalDetrFeatureExtractor.post_process_panoptic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction in batch. If unset, predictions will not be resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1769",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — a tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code> or
<code>None</code> if no mask if found above <code>threshold</code>. If <code>target_sizes</code> is specified, segmentation is resized to
the corresponding <code>target_sizes</code> entry.</li>
<li><strong>segments_info</strong> — A dictionary that contains additional information on each segment.<ul>
<li><strong>id</strong> — an integer representing the <code>segment_id</code>.</li>
<li><strong>label_id</strong> — An integer representing the label / semantic class id corresponding to <code>segment_id</code>.</li>
<li><strong>was_fused</strong> — a boolean, <code>True</code> if <code>label_id</code> was in <code>label_ids_to_fuse</code>, <code>False</code> otherwise.
Multiple instances of the same class / label were fused and assigned a single <code>segment_id</code>.</li>
<li><strong>score</strong> — Prediction score of segment with <code>segment_id</code>.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),We=new S({props:{title:"ConditionalDetrModel",local:"transformers.ConditionalDetrModel",headingTag:"h2"}}),Be=new x({props:{name:"class transformers.ConditionalDetrModel",anchor:"transformers.ConditionalDetrModel",parameters:[{name:"config",val:": ConditionalDetrConfig"}],parametersDescription:[{anchor:"transformers.ConditionalDetrModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig">ConditionalDetrConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1305"}}),Se=new x({props:{name:"forward",anchor:"transformers.ConditionalDetrModel.forward",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ConditionalDetrModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrImageProcessor">ConditionalDetrImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ConditionalDetrImageProcessor.<strong>call</strong>()</a> for details (<code>processor_class</code> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrImageProcessor">ConditionalDetrImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ConditionalDetrModel.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ConditionalDetrModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries)</code>, <em>optional</em>) &#x2014;
Not used by default. Can be used to mask object queries.`,name:"decoder_attention_mask"},{anchor:"transformers.ConditionalDetrModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.ConditionalDetrModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
can choose to directly pass a flattened representation of an image.`,name:"inputs_embeds"},{anchor:"transformers.ConditionalDetrModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
embedded representation.`,name:"decoder_inputs_embeds"},{anchor:"transformers.ConditionalDetrModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ConditionalDetrModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ConditionalDetrModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1336",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig"
>ConditionalDetrConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>~cache_utils.EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
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
<p><strong>decoder_hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>intermediate_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(config.decoder_layers, batch_size, sequence_length, hidden_size)</code>, <em>optional</em>, returned when <code>config.auxiliary_loss=True</code>) — Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
layernorm.</p>
</li>
<li>
<p><strong>reference_points</strong> (<code>torch.FloatTensor</code> of shape <code>(config.decoder_layers, batch_size, num_queries, 2 (anchor points))</code>) — Reference points (reference points of each layer of the decoder).</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),se=new rn({props:{$$slots:{default:[An]},$$scope:{ctx:I}}}),ae=new mo({props:{anchor:"transformers.ConditionalDetrModel.forward.example",$$slots:{default:[Qn]},$$scope:{ctx:I}}}),Le=new S({props:{title:"ConditionalDetrForObjectDetection",local:"transformers.ConditionalDetrForObjectDetection",headingTag:"h2"}}),Ge=new x({props:{name:"class transformers.ConditionalDetrForObjectDetection",anchor:"transformers.ConditionalDetrForObjectDetection",parameters:[{name:"config",val:": ConditionalDetrConfig"}],parametersDescription:[{anchor:"transformers.ConditionalDetrForObjectDetection.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig">ConditionalDetrConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1498"}}),Oe=new x({props:{name:"forward",anchor:"transformers.ConditionalDetrForObjectDetection.forward",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[list[dict]] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ConditionalDetrForObjectDetection.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrImageProcessor">ConditionalDetrImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ConditionalDetrImageProcessor.<strong>call</strong>()</a> for details (<code>processor_class</code> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrImageProcessor">ConditionalDetrImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries)</code>, <em>optional</em>) &#x2014;
Not used by default. Can be used to mask object queries.`,name:"decoder_attention_mask"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
can choose to directly pass a flattened representation of an image.`,name:"inputs_embeds"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
embedded representation.`,name:"decoder_inputs_embeds"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.labels",description:`<strong>labels</strong> (<code>list[Dict]</code> of len <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
following 2 keys: &#x2018;class_labels&#x2019; and &#x2018;boxes&#x2019; (the class labels and bounding boxes of an image in the batch
respectively). The class labels themselves should be a <code>torch.LongTensor</code> of len <code>(number of bounding boxes in the image,)</code> and the boxes a <code>torch.FloatTensor</code> of shape <code>(number of bounding boxes in the image, 4)</code>.`,name:"labels"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ConditionalDetrForObjectDetection.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1524",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrObjectDetectionOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig"
>ConditionalDetrConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
scale-invariant IoU loss.</p>
</li>
<li>
<p><strong>loss_dict</strong> (<code>Dict</code>, <em>optional</em>) — A dictionary containing the individual losses. Useful for logging.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, num_classes + 1)</code>) — Classification logits (including no-object) for all queries.</p>
</li>
<li>
<p><strong>pred_boxes</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, 4)</code>) — Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
possible padding). You can use <a
  href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrFeatureExtractor.post_process_object_detection"
>post_process_object_detection()</a> to retrieve the
unnormalized bounding boxes.</p>
</li>
<li>
<p><strong>auxiliary_outputs</strong> (<code>list[Dict]</code>, <em>optional</em>) — Optional, only returned when auxiliary losses are activated (i.e. <code>config.auxiliary_loss</code> is set to <code>True</code>)
and labels are provided. It is a list of dictionaries containing the two above keys (<code>logits</code> and
<code>pred_boxes</code>) for each decoder layer.</p>
</li>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>, defaults to <code>None</code>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrObjectDetectionOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),re=new rn({props:{$$slots:{default:[Yn]},$$scope:{ctx:I}}}),ie=new mo({props:{anchor:"transformers.ConditionalDetrForObjectDetection.forward.example",$$slots:{default:[Kn]},$$scope:{ctx:I}}}),He=new S({props:{title:"ConditionalDetrForSegmentation",local:"transformers.ConditionalDetrForSegmentation",headingTag:"h2"}}),Ve=new x({props:{name:"class transformers.ConditionalDetrForSegmentation",anchor:"transformers.ConditionalDetrForSegmentation",parameters:[{name:"config",val:": ConditionalDetrConfig"}],parametersDescription:[{anchor:"transformers.ConditionalDetrForSegmentation.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig">ConditionalDetrConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1666"}}),Xe=new x({props:{name:"forward",anchor:"transformers.ConditionalDetrForSegmentation.forward",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[list[dict]] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ConditionalDetrForSegmentation.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrImageProcessor">ConditionalDetrImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">ConditionalDetrImageProcessor.<strong>call</strong>()</a> for details (<code>processor_class</code> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrImageProcessor">ConditionalDetrImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries)</code>, <em>optional</em>) &#x2014;
Not used by default. Can be used to mask object queries.`,name:"decoder_attention_mask"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
can choose to directly pass a flattened representation of an image.`,name:"inputs_embeds"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
embedded representation.`,name:"decoder_inputs_embeds"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.labels",description:`<strong>labels</strong> (<code>list[Dict]</code> of len <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the bipartite matching loss, DICE/F-1 loss and Focal loss. List of dicts, each
dictionary containing at least the following 3 keys: &#x2018;class_labels&#x2019;, &#x2018;boxes&#x2019; and &#x2018;masks&#x2019; (the class labels,
bounding boxes and segmentation masks of an image in the batch respectively). The class labels themselves
should be a <code>torch.LongTensor</code> of len <code>(number of bounding boxes in the image,)</code>, the boxes a
<code>torch.FloatTensor</code> of shape <code>(number of bounding boxes in the image, 4)</code> and the masks a
<code>torch.FloatTensor</code> of shape <code>(number of bounding boxes in the image, height, width)</code>.`,name:"labels"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ConditionalDetrForSegmentation.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/conditional_detr/modeling_conditional_detr.py#L1688",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrSegmentationOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrConfig"
>ConditionalDetrConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> are provided)) — Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
scale-invariant IoU loss.</p>
</li>
<li>
<p><strong>loss_dict</strong> (<code>Dict</code>, <em>optional</em>) — A dictionary containing the individual losses. Useful for logging.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, num_classes + 1)</code>) — Classification logits (including no-object) for all queries.</p>
</li>
<li>
<p><strong>pred_boxes</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, 4)</code>) — Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
possible padding). You can use <a
  href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrFeatureExtractor.post_process_object_detection"
>post_process_object_detection()</a> to retrieve the
unnormalized bounding boxes.</p>
</li>
<li>
<p><strong>pred_masks</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, height/4, width/4)</code>) — Segmentation masks logits for all queries. See also
<a
  href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrFeatureExtractor.post_process_semantic_segmentation"
>post_process_semantic_segmentation()</a> or
<a
  href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrFeatureExtractor.post_process_instance_segmentation"
>post_process_instance_segmentation()</a>
<a
  href="/docs/transformers/v4.56.2/en/model_doc/conditional_detr#transformers.ConditionalDetrFeatureExtractor.post_process_panoptic_segmentation"
>post_process_panoptic_segmentation()</a> to evaluate semantic, instance and panoptic
segmentation masks respectively.</p>
</li>
<li>
<p><strong>auxiliary_outputs</strong> (<code>list[Dict]</code>, <em>optional</em>) — Optional, only returned when auxiliary losses are activated (i.e. <code>config.auxiliary_loss</code> is set to <code>True</code>)
and labels are provided. It is a list of dictionaries containing the two above keys (<code>logits</code> and
<code>pred_boxes</code>) for each decoder layer.</p>
</li>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>, defaults to <code>None</code>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.conditional_detr.modeling_conditional_detr.ConditionalDetrSegmentationOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),de=new rn({props:{$$slots:{default:[es]},$$scope:{ctx:I}}}),ce=new mo({props:{anchor:"transformers.ConditionalDetrForSegmentation.forward.example",$$slots:{default:[ts]},$$scope:{ctx:I}}}),Ae=new Vn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/conditional_detr.md"}}),{c(){d=r("meta"),T=n(),m=r("p"),b=n(),v=r("p"),v.innerHTML=c,D=n(),p(he.$$.fragment),Ft=n(),G=r("div"),G.innerHTML=dn,zt=n(),p(ge.$$.fragment),kt=n(),fe=r("p"),fe.innerHTML=cn,Jt=n(),ue=r("p"),ue.textContent=ln,Ut=n(),_e=r("p"),_e.innerHTML=mn,Nt=n(),O=r("img"),qt=n(),be=r("small"),be.innerHTML=hn,Pt=n(),ye=r("p"),ye.innerHTML=gn,Et=n(),p(ve.$$.fragment),Rt=n(),we=r("ul"),we.innerHTML=fn,Zt=n(),p(Te.$$.fragment),Wt=n(),J=r("div"),p(Ce.$$.fragment),ho=n(),tt=r("p"),tt.innerHTML=un,go=n(),ot=r("p"),ot.innerHTML=_n,fo=n(),p(H.$$.fragment),Bt=n(),p(xe.$$.fragment),St=n(),E=r("div"),p(De.$$.fragment),uo=n(),nt=r("p"),nt.textContent=bn,_o=n(),V=r("div"),p(Me.$$.fragment),bo=n(),st=r("p"),st.textContent=yn,Lt=n(),p(je.$$.fragment),Gt=n(),M=r("div"),p($e.$$.fragment),yo=n(),at=r("p"),at.textContent=vn,vo=n(),rt=r("div"),p(Ie.$$.fragment),wo=n(),X=r("div"),p(Fe.$$.fragment),To=n(),it=r("p"),it.innerHTML=wn,Co=n(),A=r("div"),p(ze.$$.fragment),xo=n(),dt=r("p"),dt.innerHTML=Tn,Do=n(),Q=r("div"),p(ke.$$.fragment),Mo=n(),ct=r("p"),ct.innerHTML=Cn,jo=n(),Y=r("div"),p(Je.$$.fragment),$o=n(),lt=r("p"),lt.innerHTML=xn,Ot=n(),p(Ue.$$.fragment),Ht=n(),j=r("div"),p(Ne.$$.fragment),Io=n(),K=r("div"),p(qe.$$.fragment),Fo=n(),mt=r("p"),mt.textContent=Dn,zo=n(),ee=r("div"),p(Pe.$$.fragment),ko=n(),pt=r("p"),pt.innerHTML=Mn,Jo=n(),te=r("div"),p(Ee.$$.fragment),Uo=n(),ht=r("p"),ht.innerHTML=jn,No=n(),oe=r("div"),p(Re.$$.fragment),qo=n(),gt=r("p"),gt.innerHTML=$n,Po=n(),ne=r("div"),p(Ze.$$.fragment),Eo=n(),ft=r("p"),ft.innerHTML=In,Vt=n(),p(We.$$.fragment),Xt=n(),F=r("div"),p(Be.$$.fragment),Ro=n(),ut=r("p"),ut.textContent=Fn,Zo=n(),_t=r("p"),_t.innerHTML=zn,Wo=n(),bt=r("p"),bt.innerHTML=kn,Bo=n(),U=r("div"),p(Se.$$.fragment),So=n(),yt=r("p"),yt.innerHTML=Jn,Lo=n(),p(se.$$.fragment),Go=n(),p(ae.$$.fragment),At=n(),p(Le.$$.fragment),Qt=n(),z=r("div"),p(Ge.$$.fragment),Oo=n(),vt=r("p"),vt.textContent=Un,Ho=n(),wt=r("p"),wt.innerHTML=Nn,Vo=n(),Tt=r("p"),Tt.innerHTML=qn,Xo=n(),N=r("div"),p(Oe.$$.fragment),Ao=n(),Ct=r("p"),Ct.innerHTML=Pn,Qo=n(),p(re.$$.fragment),Yo=n(),p(ie.$$.fragment),Yt=n(),p(He.$$.fragment),Kt=n(),k=r("div"),p(Ve.$$.fragment),Ko=n(),xt=r("p"),xt.textContent=En,en=n(),Dt=r("p"),Dt.innerHTML=Rn,tn=n(),Mt=r("p"),Mt.innerHTML=Zn,on=n(),q=r("div"),p(Xe.$$.fragment),nn=n(),jt=r("p"),jt.innerHTML=Wn,sn=n(),p(de.$$.fragment),an=n(),p(ce.$$.fragment),eo=n(),p(Ae.$$.fragment),to=n(),$t=r("p"),this.h()},l(e){const a=Hn("svelte-u9bgzb",document.head);d=i(a,"META",{name:!0,content:!0}),a.forEach(o),T=s(e),m=i(e,"P",{}),C(m).forEach(o),b=s(e),v=i(e,"P",{"data-svelte-h":!0}),y(v)!=="svelte-11sylwk"&&(v.innerHTML=c),D=s(e),h(he.$$.fragment,e),Ft=s(e),G=i(e,"DIV",{class:!0,"data-svelte-h":!0}),y(G)!=="svelte-13t8s2t"&&(G.innerHTML=dn),zt=s(e),h(ge.$$.fragment,e),kt=s(e),fe=i(e,"P",{"data-svelte-h":!0}),y(fe)!=="svelte-1nxmjvx"&&(fe.innerHTML=cn),Jt=s(e),ue=i(e,"P",{"data-svelte-h":!0}),y(ue)!=="svelte-vfdo9a"&&(ue.textContent=ln),Ut=s(e),_e=i(e,"P",{"data-svelte-h":!0}),y(_e)!=="svelte-jvp4bn"&&(_e.innerHTML=mn),Nt=s(e),O=i(e,"IMG",{src:!0,alt:!0,width:!0}),qt=s(e),be=i(e,"SMALL",{"data-svelte-h":!0}),y(be)!=="svelte-p2j4ms"&&(be.innerHTML=hn),Pt=s(e),ye=i(e,"P",{"data-svelte-h":!0}),y(ye)!=="svelte-7exlsi"&&(ye.innerHTML=gn),Et=s(e),h(ve.$$.fragment,e),Rt=s(e),we=i(e,"UL",{"data-svelte-h":!0}),y(we)!=="svelte-17wjkfj"&&(we.innerHTML=fn),Zt=s(e),h(Te.$$.fragment,e),Wt=s(e),J=i(e,"DIV",{class:!0});var R=C(J);h(Ce.$$.fragment,R),ho=s(R),tt=i(R,"P",{"data-svelte-h":!0}),y(tt)!=="svelte-iyszkv"&&(tt.innerHTML=un),go=s(R),ot=i(R,"P",{"data-svelte-h":!0}),y(ot)!=="svelte-1ek1ss9"&&(ot.innerHTML=_n),fo=s(R),h(H.$$.fragment,R),R.forEach(o),Bt=s(e),h(xe.$$.fragment,e),St=s(e),E=i(e,"DIV",{class:!0});var L=C(E);h(De.$$.fragment,L),uo=s(L),nt=i(L,"P",{"data-svelte-h":!0}),y(nt)!=="svelte-17j4jp9"&&(nt.textContent=bn),_o=s(L),V=i(L,"DIV",{class:!0});var Qe=C(V);h(Me.$$.fragment,Qe),bo=s(Qe),st=i(Qe,"P",{"data-svelte-h":!0}),y(st)!=="svelte-jgz2ra"&&(st.textContent=yn),Qe.forEach(o),L.forEach(o),Lt=s(e),h(je.$$.fragment,e),Gt=s(e),M=i(e,"DIV",{class:!0});var $=C(M);h($e.$$.fragment,$),yo=s($),at=i($,"P",{"data-svelte-h":!0}),y(at)!=="svelte-dzrqs3"&&(at.textContent=vn),vo=s($),rt=i($,"DIV",{class:!0});var It=C(rt);h(Ie.$$.fragment,It),It.forEach(o),wo=s($),X=i($,"DIV",{class:!0});var Ye=C(X);h(Fe.$$.fragment,Ye),To=s(Ye),it=i(Ye,"P",{"data-svelte-h":!0}),y(it)!=="svelte-gxeadx"&&(it.innerHTML=wn),Ye.forEach(o),Co=s($),A=i($,"DIV",{class:!0});var Ke=C(A);h(ze.$$.fragment,Ke),xo=s(Ke),dt=i(Ke,"P",{"data-svelte-h":!0}),y(dt)!=="svelte-1xy38dk"&&(dt.innerHTML=Tn),Ke.forEach(o),Do=s($),Q=i($,"DIV",{class:!0});var no=C(Q);h(ke.$$.fragment,no),Mo=s(no),ct=i(no,"P",{"data-svelte-h":!0}),y(ct)!=="svelte-cpbat4"&&(ct.innerHTML=Cn),no.forEach(o),jo=s($),Y=i($,"DIV",{class:!0});var so=C(Y);h(Je.$$.fragment,so),$o=s(so),lt=i(so,"P",{"data-svelte-h":!0}),y(lt)!=="svelte-fjhf5q"&&(lt.innerHTML=xn),so.forEach(o),$.forEach(o),Ot=s(e),h(Ue.$$.fragment,e),Ht=s(e),j=i(e,"DIV",{class:!0});var P=C(j);h(Ne.$$.fragment,P),Io=s(P),K=i(P,"DIV",{class:!0});var ao=C(K);h(qe.$$.fragment,ao),Fo=s(ao),mt=i(ao,"P",{"data-svelte-h":!0}),y(mt)!=="svelte-khengj"&&(mt.textContent=Dn),ao.forEach(o),zo=s(P),ee=i(P,"DIV",{class:!0});var ro=C(ee);h(Pe.$$.fragment,ro),ko=s(ro),pt=i(ro,"P",{"data-svelte-h":!0}),y(pt)!=="svelte-gxeadx"&&(pt.innerHTML=Mn),ro.forEach(o),Jo=s(P),te=i(P,"DIV",{class:!0});var io=C(te);h(Ee.$$.fragment,io),Uo=s(io),ht=i(io,"P",{"data-svelte-h":!0}),y(ht)!=="svelte-1xy38dk"&&(ht.innerHTML=jn),io.forEach(o),No=s(P),oe=i(P,"DIV",{class:!0});var co=C(oe);h(Re.$$.fragment,co),qo=s(co),gt=i(co,"P",{"data-svelte-h":!0}),y(gt)!=="svelte-cpbat4"&&(gt.innerHTML=$n),co.forEach(o),Po=s(P),ne=i(P,"DIV",{class:!0});var lo=C(ne);h(Ze.$$.fragment,lo),Eo=s(lo),ft=i(lo,"P",{"data-svelte-h":!0}),y(ft)!=="svelte-fjhf5q"&&(ft.innerHTML=In),lo.forEach(o),P.forEach(o),Vt=s(e),h(We.$$.fragment,e),Xt=s(e),F=i(e,"DIV",{class:!0});var Z=C(F);h(Be.$$.fragment,Z),Ro=s(Z),ut=i(Z,"P",{"data-svelte-h":!0}),y(ut)!=="svelte-1lwpr2z"&&(ut.textContent=Fn),Zo=s(Z),_t=i(Z,"P",{"data-svelte-h":!0}),y(_t)!=="svelte-q52n56"&&(_t.innerHTML=zn),Wo=s(Z),bt=i(Z,"P",{"data-svelte-h":!0}),y(bt)!=="svelte-hswkmf"&&(bt.innerHTML=kn),Bo=s(Z),U=i(Z,"DIV",{class:!0});var le=C(U);h(Se.$$.fragment,le),So=s(le),yt=i(le,"P",{"data-svelte-h":!0}),y(yt)!=="svelte-16wxxqf"&&(yt.innerHTML=Jn),Lo=s(le),h(se.$$.fragment,le),Go=s(le),h(ae.$$.fragment,le),le.forEach(o),Z.forEach(o),At=s(e),h(Le.$$.fragment,e),Qt=s(e),z=i(e,"DIV",{class:!0});var W=C(z);h(Ge.$$.fragment,W),Oo=s(W),vt=i(W,"P",{"data-svelte-h":!0}),y(vt)!=="svelte-tpxuf8"&&(vt.textContent=Un),Ho=s(W),wt=i(W,"P",{"data-svelte-h":!0}),y(wt)!=="svelte-q52n56"&&(wt.innerHTML=Nn),Vo=s(W),Tt=i(W,"P",{"data-svelte-h":!0}),y(Tt)!=="svelte-hswkmf"&&(Tt.innerHTML=qn),Xo=s(W),N=i(W,"DIV",{class:!0});var me=C(N);h(Oe.$$.fragment,me),Ao=s(me),Ct=i(me,"P",{"data-svelte-h":!0}),y(Ct)!=="svelte-1us7oh3"&&(Ct.innerHTML=Pn),Qo=s(me),h(re.$$.fragment,me),Yo=s(me),h(ie.$$.fragment,me),me.forEach(o),W.forEach(o),Yt=s(e),h(He.$$.fragment,e),Kt=s(e),k=i(e,"DIV",{class:!0});var B=C(k);h(Ve.$$.fragment,B),Ko=s(B),xt=i(B,"P",{"data-svelte-h":!0}),y(xt)!=="svelte-2ql3yv"&&(xt.textContent=En),en=s(B),Dt=i(B,"P",{"data-svelte-h":!0}),y(Dt)!=="svelte-q52n56"&&(Dt.innerHTML=Rn),tn=s(B),Mt=i(B,"P",{"data-svelte-h":!0}),y(Mt)!=="svelte-hswkmf"&&(Mt.innerHTML=Zn),on=s(B),q=i(B,"DIV",{class:!0});var pe=C(q);h(Xe.$$.fragment,pe),nn=s(pe),jt=i(pe,"P",{"data-svelte-h":!0}),y(jt)!=="svelte-hz70zf"&&(jt.innerHTML=Wn),sn=s(pe),h(de.$$.fragment,pe),an=s(pe),h(ce.$$.fragment,pe),pe.forEach(o),B.forEach(o),eo=s(e),h(Ae.$$.fragment,e),to=s(e),$t=i(e,"P",{}),C($t).forEach(o),this.h()},h(){w(d,"name","hf:doc:metadata"),w(d,"content",ns),w(G,"class","flex flex-wrap space-x-1"),Sn(O.src,pn="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/conditional_detr_curve.jpg")||w(O,"src",pn),w(O,"alt","drawing"),w(O,"width","600"),w(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(rt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,a){t(document.head,d),l(e,T,a),l(e,m,a),l(e,b,a),l(e,v,a),l(e,D,a),g(he,e,a),l(e,Ft,a),l(e,G,a),l(e,zt,a),g(ge,e,a),l(e,kt,a),l(e,fe,a),l(e,Jt,a),l(e,ue,a),l(e,Ut,a),l(e,_e,a),l(e,Nt,a),l(e,O,a),l(e,qt,a),l(e,be,a),l(e,Pt,a),l(e,ye,a),l(e,Et,a),g(ve,e,a),l(e,Rt,a),l(e,we,a),l(e,Zt,a),g(Te,e,a),l(e,Wt,a),l(e,J,a),g(Ce,J,null),t(J,ho),t(J,tt),t(J,go),t(J,ot),t(J,fo),g(H,J,null),l(e,Bt,a),g(xe,e,a),l(e,St,a),l(e,E,a),g(De,E,null),t(E,uo),t(E,nt),t(E,_o),t(E,V),g(Me,V,null),t(V,bo),t(V,st),l(e,Lt,a),g(je,e,a),l(e,Gt,a),l(e,M,a),g($e,M,null),t(M,yo),t(M,at),t(M,vo),t(M,rt),g(Ie,rt,null),t(M,wo),t(M,X),g(Fe,X,null),t(X,To),t(X,it),t(M,Co),t(M,A),g(ze,A,null),t(A,xo),t(A,dt),t(M,Do),t(M,Q),g(ke,Q,null),t(Q,Mo),t(Q,ct),t(M,jo),t(M,Y),g(Je,Y,null),t(Y,$o),t(Y,lt),l(e,Ot,a),g(Ue,e,a),l(e,Ht,a),l(e,j,a),g(Ne,j,null),t(j,Io),t(j,K),g(qe,K,null),t(K,Fo),t(K,mt),t(j,zo),t(j,ee),g(Pe,ee,null),t(ee,ko),t(ee,pt),t(j,Jo),t(j,te),g(Ee,te,null),t(te,Uo),t(te,ht),t(j,No),t(j,oe),g(Re,oe,null),t(oe,qo),t(oe,gt),t(j,Po),t(j,ne),g(Ze,ne,null),t(ne,Eo),t(ne,ft),l(e,Vt,a),g(We,e,a),l(e,Xt,a),l(e,F,a),g(Be,F,null),t(F,Ro),t(F,ut),t(F,Zo),t(F,_t),t(F,Wo),t(F,bt),t(F,Bo),t(F,U),g(Se,U,null),t(U,So),t(U,yt),t(U,Lo),g(se,U,null),t(U,Go),g(ae,U,null),l(e,At,a),g(Le,e,a),l(e,Qt,a),l(e,z,a),g(Ge,z,null),t(z,Oo),t(z,vt),t(z,Ho),t(z,wt),t(z,Vo),t(z,Tt),t(z,Xo),t(z,N),g(Oe,N,null),t(N,Ao),t(N,Ct),t(N,Qo),g(re,N,null),t(N,Yo),g(ie,N,null),l(e,Yt,a),g(He,e,a),l(e,Kt,a),l(e,k,a),g(Ve,k,null),t(k,Ko),t(k,xt),t(k,en),t(k,Dt),t(k,tn),t(k,Mt),t(k,on),t(k,q),g(Xe,q,null),t(q,nn),t(q,jt),t(q,sn),g(de,q,null),t(q,an),g(ce,q,null),l(e,eo,a),g(Ae,e,a),l(e,to,a),l(e,$t,a),oo=!0},p(e,[a]){const R={};a&2&&(R.$$scope={dirty:a,ctx:e}),H.$set(R);const L={};a&2&&(L.$$scope={dirty:a,ctx:e}),se.$set(L);const Qe={};a&2&&(Qe.$$scope={dirty:a,ctx:e}),ae.$set(Qe);const $={};a&2&&($.$$scope={dirty:a,ctx:e}),re.$set($);const It={};a&2&&(It.$$scope={dirty:a,ctx:e}),ie.$set(It);const Ye={};a&2&&(Ye.$$scope={dirty:a,ctx:e}),de.$set(Ye);const Ke={};a&2&&(Ke.$$scope={dirty:a,ctx:e}),ce.$set(Ke)},i(e){oo||(f(he.$$.fragment,e),f(ge.$$.fragment,e),f(ve.$$.fragment,e),f(Te.$$.fragment,e),f(Ce.$$.fragment,e),f(H.$$.fragment,e),f(xe.$$.fragment,e),f(De.$$.fragment,e),f(Me.$$.fragment,e),f(je.$$.fragment,e),f($e.$$.fragment,e),f(Ie.$$.fragment,e),f(Fe.$$.fragment,e),f(ze.$$.fragment,e),f(ke.$$.fragment,e),f(Je.$$.fragment,e),f(Ue.$$.fragment,e),f(Ne.$$.fragment,e),f(qe.$$.fragment,e),f(Pe.$$.fragment,e),f(Ee.$$.fragment,e),f(Re.$$.fragment,e),f(Ze.$$.fragment,e),f(We.$$.fragment,e),f(Be.$$.fragment,e),f(Se.$$.fragment,e),f(se.$$.fragment,e),f(ae.$$.fragment,e),f(Le.$$.fragment,e),f(Ge.$$.fragment,e),f(Oe.$$.fragment,e),f(re.$$.fragment,e),f(ie.$$.fragment,e),f(He.$$.fragment,e),f(Ve.$$.fragment,e),f(Xe.$$.fragment,e),f(de.$$.fragment,e),f(ce.$$.fragment,e),f(Ae.$$.fragment,e),oo=!0)},o(e){u(he.$$.fragment,e),u(ge.$$.fragment,e),u(ve.$$.fragment,e),u(Te.$$.fragment,e),u(Ce.$$.fragment,e),u(H.$$.fragment,e),u(xe.$$.fragment,e),u(De.$$.fragment,e),u(Me.$$.fragment,e),u(je.$$.fragment,e),u($e.$$.fragment,e),u(Ie.$$.fragment,e),u(Fe.$$.fragment,e),u(ze.$$.fragment,e),u(ke.$$.fragment,e),u(Je.$$.fragment,e),u(Ue.$$.fragment,e),u(Ne.$$.fragment,e),u(qe.$$.fragment,e),u(Pe.$$.fragment,e),u(Ee.$$.fragment,e),u(Re.$$.fragment,e),u(Ze.$$.fragment,e),u(We.$$.fragment,e),u(Be.$$.fragment,e),u(Se.$$.fragment,e),u(se.$$.fragment,e),u(ae.$$.fragment,e),u(Le.$$.fragment,e),u(Ge.$$.fragment,e),u(Oe.$$.fragment,e),u(re.$$.fragment,e),u(ie.$$.fragment,e),u(He.$$.fragment,e),u(Ve.$$.fragment,e),u(Xe.$$.fragment,e),u(de.$$.fragment,e),u(ce.$$.fragment,e),u(Ae.$$.fragment,e),oo=!1},d(e){e&&(o(T),o(m),o(b),o(v),o(D),o(Ft),o(G),o(zt),o(kt),o(fe),o(Jt),o(ue),o(Ut),o(_e),o(Nt),o(O),o(qt),o(be),o(Pt),o(ye),o(Et),o(Rt),o(we),o(Zt),o(Wt),o(J),o(Bt),o(St),o(E),o(Lt),o(Gt),o(M),o(Ot),o(Ht),o(j),o(Vt),o(Xt),o(F),o(At),o(Qt),o(z),o(Yt),o(Kt),o(k),o(eo),o(to),o($t)),o(d),_(he,e),_(ge,e),_(ve,e),_(Te,e),_(Ce),_(H),_(xe,e),_(De),_(Me),_(je,e),_($e),_(Ie),_(Fe),_(ze),_(ke),_(Je),_(Ue,e),_(Ne),_(qe),_(Pe),_(Ee),_(Re),_(Ze),_(We,e),_(Be),_(Se),_(se),_(ae),_(Le,e),_(Ge),_(Oe),_(re),_(ie),_(He,e),_(Ve),_(Xe),_(de),_(ce),_(Ae,e)}}}const ns='{"title":"Conditional DETR","local":"conditional-detr","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"ConditionalDetrConfig","local":"transformers.ConditionalDetrConfig","sections":[],"depth":2},{"title":"ConditionalDetrImageProcessor","local":"transformers.ConditionalDetrImageProcessor","sections":[],"depth":2},{"title":"ConditionalDetrImageProcessorFast","local":"transformers.ConditionalDetrImageProcessorFast","sections":[],"depth":2},{"title":"ConditionalDetrFeatureExtractor","local":"transformers.ConditionalDetrFeatureExtractor","sections":[],"depth":2},{"title":"ConditionalDetrModel","local":"transformers.ConditionalDetrModel","sections":[],"depth":2},{"title":"ConditionalDetrForObjectDetection","local":"transformers.ConditionalDetrForObjectDetection","sections":[],"depth":2},{"title":"ConditionalDetrForSegmentation","local":"transformers.ConditionalDetrForSegmentation","sections":[],"depth":2}],"depth":1}';function ss(I){return Ln(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ps extends Gn{constructor(d){super(),On(this,d,ss,os,Bn,{})}}export{ps as component};
