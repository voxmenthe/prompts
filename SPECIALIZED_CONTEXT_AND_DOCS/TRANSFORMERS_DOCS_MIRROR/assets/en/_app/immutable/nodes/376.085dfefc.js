import{s as St,o as Qt,n as ke}from"../chunks/scheduler.18a86fab.js";import{S as Pt,i as Lt,g as l,s as a,r as u,A as At,h as c,f as o,c as r,j as N,x as h,u as f,k as U,y as d,a as n,v as g,d as _,t as b,w as T}from"../chunks/index.98837b22.js";import{T as Xt}from"../chunks/Tip.77304350.js";import{D as De}from"../chunks/Docstring.a1ef7999.js";import{C as tt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as wt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{P as Yt}from"../chunks/PipelineTag.7749150e.js";import{H as fe,E as Ot}from"../chunks/getInferenceSnippets.06c2775f.js";function Kt(J){let s,v="Examples:",p,m,y;return m=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJURGV0clYyQ29uZmlnJTJDJTIwUlREZXRyVjJNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBSVC1ERVRSJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBSVERldHJWMkNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBSVERldHJWMk1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RTDetrV2Config, RTDetrV2Model

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a RT-DETR configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = RTDetrV2Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RTDetrV2Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){s=l("p"),s.textContent=v,p=a(),u(m.$$.fragment)},l(i){s=c(i,"P",{"data-svelte-h":!0}),h(s)!=="svelte-kvfsh7"&&(s.textContent=v),p=r(i),f(m.$$.fragment,i)},m(i,M){n(i,s,M),n(i,p,M),g(m,i,M),y=!0},p:ke,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){b(m.$$.fragment,i),y=!1},d(i){i&&(o(s),o(p)),T(m,i)}}}function eo(J){let s,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=l("p"),s.innerHTML=v},l(p){s=c(p,"P",{"data-svelte-h":!0}),h(s)!=="svelte-fincs2"&&(s.innerHTML=v)},m(p,m){n(p,s,m)},p:ke,d(p){p&&o(s)}}}function to(J){let s,v="Examples:",p,m,y;return m=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMFJURGV0clYyTW9kZWwlMEFmcm9tJTIwUElMJTIwaW1wb3J0JTIwSW1hZ2UlMEFpbXBvcnQlMjByZXF1ZXN0cyUwQSUwQXVybCUyMCUzRCUyMCUyMmh0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEElMEFpbWFnZV9wcm9jZXNzb3IlMjAlM0QlMjBBdXRvSW1hZ2VQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMlBla2luZ1UlMkZSVERldHJWMl9yNTB2ZCUyMiklMEFtb2RlbCUyMCUzRCUyMFJURGV0clYyTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMlBla2luZ1UlMkZSVERldHJWMl9yNTB2ZCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IoaW1hZ2VzJTNEaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBbGlzdChsYXN0X2hpZGRlbl9zdGF0ZXMuc2hhcGUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, RTDetrV2Model
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;PekingU/RTDetrV2_r50vd&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RTDetrV2Model.from_pretrained(<span class="hljs-string">&quot;PekingU/RTDetrV2_r50vd&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">300</span>, <span class="hljs-number">256</span>]`,wrap:!1}}),{c(){s=l("p"),s.textContent=v,p=a(),u(m.$$.fragment)},l(i){s=c(i,"P",{"data-svelte-h":!0}),h(s)!=="svelte-kvfsh7"&&(s.textContent=v),p=r(i),f(m.$$.fragment,i)},m(i,M){n(i,s,M),n(i,p,M),g(m,i,M),y=!0},p:ke,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){b(m.$$.fragment,i),y=!1},d(i){i&&(o(s),o(p)),T(m,i)}}}function oo(J){let s,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=l("p"),s.innerHTML=v},l(p){s=c(p,"P",{"data-svelte-h":!0}),h(s)!=="svelte-fincs2"&&(s.innerHTML=v)},m(p,m){n(p,s,m)},p:ke,d(p){p&&o(s)}}}function no(J){let s,v="Examples:",p,m,y;return m=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJURGV0clYySW1hZ2VQcm9jZXNzb3IlMkMlMjBSVERldHJWMkZvck9iamVjdERldGVjdGlvbiUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHJlcXVlc3RzJTBBaW1wb3J0JTIwdG9yY2glMEElMEF1cmwlMjAlM0QlMjAlMjJodHRwJTNBJTJGJTJGaW1hZ2VzLmNvY29kYXRhc2V0Lm9yZyUyRnZhbDIwMTclMkYwMDAwMDAwMzk3NjkuanBnJTIyJTBBaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBJTBBaW1hZ2VfcHJvY2Vzc29yJTIwJTNEJTIwUlREZXRyVjJJbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyUGVraW5nVSUyRlJURGV0clYyX3I1MHZkJTIyKSUwQW1vZGVsJTIwJTNEJTIwUlREZXRyVjJGb3JPYmplY3REZXRlY3Rpb24uZnJvbV9wcmV0cmFpbmVkKCUyMlBla2luZ1UlMkZSVERldHJWMl9yNTB2ZCUyMiklMEElMEElMjMlMjBwcmVwYXJlJTIwaW1hZ2UlMjBmb3IlMjB0aGUlMjBtb2RlbCUwQWlucHV0cyUyMCUzRCUyMGltYWdlX3Byb2Nlc3NvcihpbWFnZXMlM0RpbWFnZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwZm9yd2FyZCUyMHBhc3MlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEFsaXN0KGxvZ2l0cy5zaGFwZSklMEElMEFib3hlcyUyMCUzRCUyMG91dHB1dHMucHJlZF9ib3hlcyUwQWxpc3QoYm94ZXMuc2hhcGUpJTBBJTBBJTIzJTIwY29udmVydCUyMG91dHB1dHMlMjAoYm91bmRpbmclMjBib3hlcyUyMGFuZCUyMGNsYXNzJTIwbG9naXRzKSUyMHRvJTIwUGFzY2FsJTIwVk9DJTIwZm9ybWF0JTIwKHhtaW4lMkMlMjB5bWluJTJDJTIweG1heCUyQyUyMHltYXgpJTBBdGFyZ2V0X3NpemVzJTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QmltYWdlLnNpemUlNUIlM0ElM0EtMSU1RCU1RCklMEFyZXN1bHRzJTIwJTNEJTIwaW1hZ2VfcHJvY2Vzc29yLnBvc3RfcHJvY2Vzc19vYmplY3RfZGV0ZWN0aW9uKG91dHB1dHMlMkMlMjB0aHJlc2hvbGQlM0QwLjklMkMlMjB0YXJnZXRfc2l6ZXMlM0R0YXJnZXRfc2l6ZXMpJTVCJTBBJTIwJTIwJTIwJTIwMCUwQSU1RCUwQSUwQWZvciUyMHNjb3JlJTJDJTIwbGFiZWwlMkMlMjBib3glMjBpbiUyMHppcChyZXN1bHRzJTVCJTIyc2NvcmVzJTIyJTVEJTJDJTIwcmVzdWx0cyU1QiUyMmxhYmVscyUyMiU1RCUyQyUyMHJlc3VsdHMlNUIlMjJib3hlcyUyMiU1RCklM0ElMEElMjAlMjAlMjAlMjBib3glMjAlM0QlMjAlNUJyb3VuZChpJTJDJTIwMiklMjBmb3IlMjBpJTIwaW4lMjBib3gudG9saXN0KCklNUQlMEElMjAlMjAlMjAlMjBwcmludCglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBmJTIyRGV0ZWN0ZWQlMjAlN0Jtb2RlbC5jb25maWcuaWQybGFiZWwlNUJsYWJlbC5pdGVtKCklNUQlN0QlMjB3aXRoJTIwY29uZmlkZW5jZSUyMCUyMiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGYlMjIlN0Jyb3VuZChzY29yZS5pdGVtKCklMkMlMjAzKSU3RCUyMGF0JTIwbG9jYXRpb24lMjAlN0Jib3glN0QlMjIlMEElMjAlMjAlMjAlMjAp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RTDetrV2ImageProcessor, RTDetrV2ForObjectDetection
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = RTDetrV2ImageProcessor.from_pretrained(<span class="hljs-string">&quot;PekingU/RTDetrV2_r50vd&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RTDetrV2ForObjectDetection.from_pretrained(<span class="hljs-string">&quot;PekingU/RTDetrV2_r50vd&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># prepare image for the model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">300</span>, <span class="hljs-number">80</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>boxes = outputs.pred_boxes
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(boxes.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">300</span>, <span class="hljs-number">4</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_sizes = torch.tensor([image.size[::-<span class="hljs-number">1</span>]])
<span class="hljs-meta">&gt;&gt;&gt; </span>results = image_processor.post_process_object_detection(outputs, threshold=<span class="hljs-number">0.9</span>, target_sizes=target_sizes)[
<span class="hljs-meta">... </span>    <span class="hljs-number">0</span>
<span class="hljs-meta">... </span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> score, label, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(results[<span class="hljs-string">&quot;scores&quot;</span>], results[<span class="hljs-string">&quot;labels&quot;</span>], results[<span class="hljs-string">&quot;boxes&quot;</span>]):
<span class="hljs-meta">... </span>    box = [<span class="hljs-built_in">round</span>(i, <span class="hljs-number">2</span>) <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> box.tolist()]
<span class="hljs-meta">... </span>    <span class="hljs-built_in">print</span>(
<span class="hljs-meta">... </span>        <span class="hljs-string">f&quot;Detected <span class="hljs-subst">{model.config.id2label[label.item()]}</span> with confidence &quot;</span>
<span class="hljs-meta">... </span>        <span class="hljs-string">f&quot;<span class="hljs-subst">{<span class="hljs-built_in">round</span>(score.item(), <span class="hljs-number">3</span>)}</span> at location <span class="hljs-subst">{box}</span>&quot;</span>
<span class="hljs-meta">... </span>    )
Detected sofa <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.97</span> at location [<span class="hljs-number">0.14</span>, <span class="hljs-number">0.38</span>, <span class="hljs-number">640.13</span>, <span class="hljs-number">476.21</span>]
Detected cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.96</span> at location [<span class="hljs-number">343.38</span>, <span class="hljs-number">24.28</span>, <span class="hljs-number">640.14</span>, <span class="hljs-number">371.5</span>]
Detected cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.958</span> at location [<span class="hljs-number">13.23</span>, <span class="hljs-number">54.18</span>, <span class="hljs-number">318.98</span>, <span class="hljs-number">472.22</span>]
Detected remote <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.951</span> at location [<span class="hljs-number">40.11</span>, <span class="hljs-number">73.44</span>, <span class="hljs-number">175.96</span>, <span class="hljs-number">118.48</span>]
Detected remote <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.924</span> at location [<span class="hljs-number">333.73</span>, <span class="hljs-number">76.58</span>, <span class="hljs-number">369.97</span>, <span class="hljs-number">186.99</span>]`,wrap:!1}}),{c(){s=l("p"),s.textContent=v,p=a(),u(m.$$.fragment)},l(i){s=c(i,"P",{"data-svelte-h":!0}),h(s)!=="svelte-kvfsh7"&&(s.textContent=v),p=r(i),f(m.$$.fragment,i)},m(i,M){n(i,s,M),n(i,p,M),g(m,i,M),y=!0},p:ke,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){b(m.$$.fragment,i),y=!1},d(i){i&&(o(s),o(p)),T(m,i)}}}function so(J){let s,v,p,m,y,i="<em>This model was released on 2024-07-24 and added to Hugging Face Transformers on 2025-02-06.</em>",M,H,Ce,z,Mt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',$e,G,Ue,X,jt='The RT-DETRv2 model was proposed in <a href="https://huggingface.co/papers/2407.17140" rel="nofollow">RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer</a> by Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu.',ze,S,xt="RT-DETRv2 refines RT-DETR by introducing selective multi-scale feature extraction, a discrete sampling operator for broader deployment compatibility, and improved training strategies like dynamic data augmentation and scale-adaptive hyperparameters. These changes enhance flexibility and practicality while maintaining real-time performance.",Fe,Q,Rt="The abstract from the paper is the following:",Ze,P,Vt="<em>In this report, we present RT-DETRv2, an improved Real-Time DEtection TRansformer (RT-DETR). RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance. To improve the flexibility, we suggest setting a distinct number of sampling points for features at different scales in the deformable attention to achieve selective multi-scale feature extraction by the decoder. To enhance practicality, we propose an optional discrete sampling operator to replace the grid_sample operator that is specific to RT-DETR compared to YOLOs. This removes the deployment constraints typically associated with DETRs. For the training strategy, we propose dynamic data augmentation and scale-adaptive hyperparameters customization to improve performance without loss of speed.</em>",Ie,L,Dt=`This model was contributed by <a href="https://huggingface.co/jadechoghari" rel="nofollow">jadechoghari</a>.
The original code can be found <a href="https://github.com/lyuwenyu/RT-DETR" rel="nofollow">here</a>.`,qe,A,Be,Y,Jt="This second version of RT-DETR improves how the decoder finds objects in an image.",We,O,kt="<li><strong>better sampling</strong> – adjusts offsets so the model looks at the right areas</li> <li><strong>flexible attention</strong> – can use smooth (bilinear) or fixed (discrete) sampling</li> <li><strong>optimized processing</strong> – improves how attention weights mix information</li>",Ee,K,Ne,ee,He,te,Ct="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with RT-DETRv2.",Ge,oe,Xe,ne,$t='<li>Scripts for finetuning <a href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2ForObjectDetection">RTDetrV2ForObjectDetection</a> with <a href="/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer">Trainer</a> or <a href="https://huggingface.co/docs/accelerate/index" rel="nofollow">Accelerate</a> can be found <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection" rel="nofollow">here</a>.</li> <li>See also: <a href="../tasks/object_detection">Object detection task guide</a>.</li> <li>Notebooks for <a href="https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/RT_DETR_v2_inference.ipynb" rel="nofollow">inference</a> and <a href="https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb" rel="nofollow">fine-tuning</a> RT-DETRv2 on a custom dataset (🌎).</li>',Se,se,Qe,w,ae,ot,ge,Ut=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Model">RTDetrV2Model</a>. It is used to instantiate a
RT-DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the RT-DETR architecture.`,nt,_e,zt='e.g. <a href="https://huggingface.co/PekingU/rtdetr_r18vd" rel="nofollow">PekingU/rtdetr_r18vd</a>',st,be,Ft=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,at,F,rt,Z,re,it,Te,Zt=`Instantiate a <a href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config">RTDetrV2Config</a> (or a derived class) from a pre-trained backbone model configuration and DETR model
configuration.`,Pe,ie,Le,j,le,lt,ye,It="RT-DETR Model (consisting of a backbone and encoder-decoder) outputting raw hidden states without any head on top.",ct,ve,qt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,dt,we,Bt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,pt,k,ce,mt,Me,Wt='The <a href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Model">RTDetrV2Model</a> forward method, overrides the <code>__call__</code> special method.',ht,I,ut,q,Ae,de,Ye,x,pe,ft,je,Et=`RT-DETR Model (consisting of a backbone and encoder-decoder) outputting bounding boxes and logits to be further
decoded into scores and classes.`,gt,xe,Nt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_t,Re,Ht=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,bt,C,me,Tt,Ve,Gt='The <a href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2ForObjectDetection">RTDetrV2ForObjectDetection</a> forward method, overrides the <code>__call__</code> special method.',yt,B,vt,W,Oe,he,Ke,Je,et;return H=new fe({props:{title:"RT-DETRv2",local:"rt-detrv2",headingTag:"h1"}}),G=new fe({props:{title:"Overview",local:"overview",headingTag:"h2"}}),A=new fe({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),K=new tt({props:{code:"aW1wb3J0JTIwdG9yY2glMEFpbXBvcnQlMjByZXF1ZXN0cyUwQSUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBSVERldHJWMkZvck9iamVjdERldGVjdGlvbiUyQyUyMFJURGV0ckltYWdlUHJvY2Vzc29yJTBBJTBBdXJsJTIwJTNEJTIwJ2h0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGcnJTBBaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwlMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTBBJTBBaW1hZ2VfcHJvY2Vzc29yJTIwJTNEJTIwUlREZXRySW1hZ2VQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMlBla2luZ1UlMkZydGRldHJfdjJfcjE4dmQlMjIpJTBBbW9kZWwlMjAlM0QlMjBSVERldHJWMkZvck9iamVjdERldGVjdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyUGVraW5nVSUyRnJ0ZGV0cl92Ml9yMTh2ZCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IoaW1hZ2VzJTNEaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBcmVzdWx0cyUyMCUzRCUyMGltYWdlX3Byb2Nlc3Nvci5wb3N0X3Byb2Nlc3Nfb2JqZWN0X2RldGVjdGlvbihvdXRwdXRzJTJDJTIwdGFyZ2V0X3NpemVzJTNEdG9yY2gudGVuc29yKCU1QihpbWFnZS5oZWlnaHQlMkMlMjBpbWFnZS53aWR0aCklNUQpJTJDJTIwdGhyZXNob2xkJTNEMC41KSUwQSUwQWZvciUyMHJlc3VsdCUyMGluJTIwcmVzdWx0cyUzQSUwQSUyMCUyMCUyMCUyMGZvciUyMHNjb3JlJTJDJTIwbGFiZWxfaWQlMkMlMjBib3glMjBpbiUyMHppcChyZXN1bHQlNUIlMjJzY29yZXMlMjIlNUQlMkMlMjByZXN1bHQlNUIlMjJsYWJlbHMlMjIlNUQlMkMlMjByZXN1bHQlNUIlMjJib3hlcyUyMiU1RCklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBzY29yZSUyQyUyMGxhYmVsJTIwJTNEJTIwc2NvcmUuaXRlbSgpJTJDJTIwbGFiZWxfaWQuaXRlbSgpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwYm94JTIwJTNEJTIwJTVCcm91bmQoaSUyQyUyMDIpJTIwZm9yJTIwaSUyMGluJTIwYm94LnRvbGlzdCgpJTVEJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwcHJpbnQoZiUyMiU3Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QmxhYmVsJTVEJTdEJTNBJTIwJTdCc2NvcmUlM0EuMmYlN0QlMjAlN0Jib3glN0QlMjIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RTDetrV2ForObjectDetection, RTDetrImageProcessor

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&#x27;http://images.cocodataset.org/val2017/000000039769.jpg&#x27;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = RTDetrImageProcessor.from_pretrained(<span class="hljs-string">&quot;PekingU/rtdetr_v2_r18vd&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RTDetrV2ForObjectDetection.from_pretrained(<span class="hljs-string">&quot;PekingU/rtdetr_v2_r18vd&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=<span class="hljs-number">0.5</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> result <span class="hljs-keyword">in</span> results:
<span class="hljs-meta">... </span>    <span class="hljs-keyword">for</span> score, label_id, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(result[<span class="hljs-string">&quot;scores&quot;</span>], result[<span class="hljs-string">&quot;labels&quot;</span>], result[<span class="hljs-string">&quot;boxes&quot;</span>]):
<span class="hljs-meta">... </span>        score, label = score.item(), label_id.item()
<span class="hljs-meta">... </span>        box = [<span class="hljs-built_in">round</span>(i, <span class="hljs-number">2</span>) <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> box.tolist()]
<span class="hljs-meta">... </span>        <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;<span class="hljs-subst">{model.config.id2label[label]}</span>: <span class="hljs-subst">{score:<span class="hljs-number">.2</span>f}</span> <span class="hljs-subst">{box}</span>&quot;</span>)
cat: <span class="hljs-number">0.97</span> [<span class="hljs-number">341.14</span>, <span class="hljs-number">25.11</span>, <span class="hljs-number">639.98</span>, <span class="hljs-number">372.89</span>]
cat: <span class="hljs-number">0.96</span> [<span class="hljs-number">12.78</span>, <span class="hljs-number">56.35</span>, <span class="hljs-number">317.67</span>, <span class="hljs-number">471.34</span>]
remote: <span class="hljs-number">0.95</span> [<span class="hljs-number">39.96</span>, <span class="hljs-number">73.12</span>, <span class="hljs-number">175.65</span>, <span class="hljs-number">117.44</span>]
sofa: <span class="hljs-number">0.86</span> [-<span class="hljs-number">0.11</span>, <span class="hljs-number">2.97</span>, <span class="hljs-number">639.89</span>, <span class="hljs-number">473.62</span>]
sofa: <span class="hljs-number">0.82</span> [-<span class="hljs-number">0.12</span>, <span class="hljs-number">1.78</span>, <span class="hljs-number">639.87</span>, <span class="hljs-number">473.52</span>]
remote: <span class="hljs-number">0.79</span> [<span class="hljs-number">333.65</span>, <span class="hljs-number">76.38</span>, <span class="hljs-number">370.69</span>, <span class="hljs-number">187.48</span>]`,wrap:!1}}),ee=new fe({props:{title:"Resources",local:"resources",headingTag:"h2"}}),oe=new Yt({props:{pipeline:"object-detection"}}),se=new fe({props:{title:"RTDetrV2Config",local:"transformers.RTDetrV2Config",headingTag:"h2"}}),ae=new De({props:{name:"class transformers.RTDetrV2Config",anchor:"transformers.RTDetrV2Config",parameters:[{name:"initializer_range",val:" = 0.01"},{name:"initializer_bias_prior_prob",val:" = None"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"batch_norm_eps",val:" = 1e-05"},{name:"backbone_config",val:" = None"},{name:"backbone",val:" = None"},{name:"use_pretrained_backbone",val:" = False"},{name:"use_timm_backbone",val:" = False"},{name:"freeze_backbone_batch_norms",val:" = True"},{name:"backbone_kwargs",val:" = None"},{name:"encoder_hidden_dim",val:" = 256"},{name:"encoder_in_channels",val:" = [512, 1024, 2048]"},{name:"feat_strides",val:" = [8, 16, 32]"},{name:"encoder_layers",val:" = 1"},{name:"encoder_ffn_dim",val:" = 1024"},{name:"encoder_attention_heads",val:" = 8"},{name:"dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"encode_proj_layers",val:" = [2]"},{name:"positional_encoding_temperature",val:" = 10000"},{name:"encoder_activation_function",val:" = 'gelu'"},{name:"activation_function",val:" = 'silu'"},{name:"eval_size",val:" = None"},{name:"normalize_before",val:" = False"},{name:"hidden_expansion",val:" = 1.0"},{name:"d_model",val:" = 256"},{name:"num_queries",val:" = 300"},{name:"decoder_in_channels",val:" = [256, 256, 256]"},{name:"decoder_ffn_dim",val:" = 1024"},{name:"num_feature_levels",val:" = 3"},{name:"decoder_n_points",val:" = 4"},{name:"decoder_layers",val:" = 6"},{name:"decoder_attention_heads",val:" = 8"},{name:"decoder_activation_function",val:" = 'relu'"},{name:"attention_dropout",val:" = 0.0"},{name:"num_denoising",val:" = 100"},{name:"label_noise_ratio",val:" = 0.5"},{name:"box_noise_scale",val:" = 1.0"},{name:"learn_initial_query",val:" = False"},{name:"anchor_image_size",val:" = None"},{name:"with_box_refine",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"matcher_alpha",val:" = 0.25"},{name:"matcher_gamma",val:" = 2.0"},{name:"matcher_class_cost",val:" = 2.0"},{name:"matcher_bbox_cost",val:" = 5.0"},{name:"matcher_giou_cost",val:" = 2.0"},{name:"use_focal_loss",val:" = True"},{name:"auxiliary_loss",val:" = True"},{name:"focal_loss_alpha",val:" = 0.75"},{name:"focal_loss_gamma",val:" = 2.0"},{name:"weight_loss_vfl",val:" = 1.0"},{name:"weight_loss_bbox",val:" = 5.0"},{name:"weight_loss_giou",val:" = 2.0"},{name:"eos_coefficient",val:" = 0.0001"},{name:"decoder_n_levels",val:" = 3"},{name:"decoder_offset_scale",val:" = 0.5"},{name:"decoder_method",val:" = 'default'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RTDetrV2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.01) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.RTDetrV2Config.initializer_bias_prior_prob",description:`<strong>initializer_bias_prior_prob</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The prior probability used by the bias initializer to initialize biases for <code>enc_score_head</code> and <code>class_embed</code>.
If <code>None</code>, <code>prior_prob</code> computed as <code>prior_prob = 1 / (num_labels + 1)</code> while initializing model weights.`,name:"initializer_bias_prior_prob"},{anchor:"transformers.RTDetrV2Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.RTDetrV2Config.batch_norm_eps",description:`<strong>batch_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the batch normalization layers.`,name:"batch_norm_eps"},{anchor:"transformers.RTDetrV2Config.backbone_config",description:`<strong>backbone_config</strong> (<code>Dict</code>, <em>optional</em>, defaults to <code>RTDetrV2ResNetConfig()</code>) &#x2014;
The configuration of the backbone model.`,name:"backbone_config"},{anchor:"transformers.RTDetrV2Config.backbone",description:`<strong>backbone</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Name of backbone to use when <code>backbone_config</code> is <code>None</code>. If <code>use_pretrained_backbone</code> is <code>True</code>, this
will load the corresponding pretrained weights from the timm or transformers library. If <code>use_pretrained_backbone</code>
is <code>False</code>, this loads the backbone&#x2019;s config and uses that to initialize the backbone with random weights.`,name:"backbone"},{anchor:"transformers.RTDetrV2Config.use_pretrained_backbone",description:`<strong>use_pretrained_backbone</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use pretrained weights for the backbone.`,name:"use_pretrained_backbone"},{anchor:"transformers.RTDetrV2Config.use_timm_backbone",description:`<strong>use_timm_backbone</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to load <code>backbone</code> from the timm library. If <code>False</code>, the backbone is loaded from the transformers
library.`,name:"use_timm_backbone"},{anchor:"transformers.RTDetrV2Config.freeze_backbone_batch_norms",description:`<strong>freeze_backbone_batch_norms</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to freeze the batch normalization layers in the backbone.`,name:"freeze_backbone_batch_norms"},{anchor:"transformers.RTDetrV2Config.backbone_kwargs",description:`<strong>backbone_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
e.g. <code>{&apos;out_indices&apos;: (0, 1, 2, 3)}</code>. Cannot be specified if <code>backbone_config</code> is set.`,name:"backbone_kwargs"},{anchor:"transformers.RTDetrV2Config.encoder_hidden_dim",description:`<strong>encoder_hidden_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimension of the layers in hybrid encoder.`,name:"encoder_hidden_dim"},{anchor:"transformers.RTDetrV2Config.encoder_in_channels",description:`<strong>encoder_in_channels</strong> (<code>list</code>, <em>optional</em>, defaults to <code>[512, 1024, 2048]</code>) &#x2014;
Multi level features input for encoder.`,name:"encoder_in_channels"},{anchor:"transformers.RTDetrV2Config.feat_strides",description:`<strong>feat_strides</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[8, 16, 32]</code>) &#x2014;
Strides used in each feature map.`,name:"feat_strides"},{anchor:"transformers.RTDetrV2Config.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Total of layers to be used by the encoder.`,name:"encoder_layers"},{anchor:"transformers.RTDetrV2Config.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.RTDetrV2Config.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.RTDetrV2Config.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The ratio for all dropout layers.`,name:"dropout"},{anchor:"transformers.RTDetrV2Config.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.RTDetrV2Config.encode_proj_layers",description:`<strong>encode_proj_layers</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[2]</code>) &#x2014;
Indexes of the projected layers to be used in the encoder.`,name:"encode_proj_layers"},{anchor:"transformers.RTDetrV2Config.positional_encoding_temperature",description:`<strong>positional_encoding_temperature</strong> (<code>int</code>, <em>optional</em>, defaults to 10000) &#x2014;
The temperature parameter used to create the positional encodings.`,name:"positional_encoding_temperature"},{anchor:"transformers.RTDetrV2Config.encoder_activation_function",description:`<strong>encoder_activation_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"encoder_activation_function"},{anchor:"transformers.RTDetrV2Config.activation_function",description:`<strong>activation_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the general layer. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.RTDetrV2Config.eval_size",description:`<strong>eval_size</strong> (<code>tuple[int, int]</code>, <em>optional</em>) &#x2014;
Height and width used to compute the effective height and width of the position embeddings after taking
into account the stride.`,name:"eval_size"},{anchor:"transformers.RTDetrV2Config.normalize_before",description:`<strong>normalize_before</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
feed-forward modules.`,name:"normalize_before"},{anchor:"transformers.RTDetrV2Config.hidden_expansion",description:`<strong>hidden_expansion</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.`,name:"hidden_expansion"},{anchor:"transformers.RTDetrV2Config.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimension of the layers exclude hybrid encoder.`,name:"d_model"},{anchor:"transformers.RTDetrV2Config.num_queries",description:`<strong>num_queries</strong> (<code>int</code>, <em>optional</em>, defaults to 300) &#x2014;
Number of object queries.`,name:"num_queries"},{anchor:"transformers.RTDetrV2Config.decoder_in_channels",description:`<strong>decoder_in_channels</strong> (<code>list</code>, <em>optional</em>, defaults to <code>[256, 256, 256]</code>) &#x2014;
Multi level features dimension for decoder`,name:"decoder_in_channels"},{anchor:"transformers.RTDetrV2Config.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.RTDetrV2Config.num_feature_levels",description:`<strong>num_feature_levels</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The number of input feature levels.`,name:"num_feature_levels"},{anchor:"transformers.RTDetrV2Config.decoder_n_points",description:`<strong>decoder_n_points</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The number of sampled keys in each feature level for each attention head in the decoder.`,name:"decoder_n_points"},{anchor:"transformers.RTDetrV2Config.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.RTDetrV2Config.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.RTDetrV2Config.decoder_activation_function",description:`<strong>decoder_activation_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"decoder_activation_function"},{anchor:"transformers.RTDetrV2Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.RTDetrV2Config.num_denoising",description:`<strong>num_denoising</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
The total number of denoising tasks or queries to be used for contrastive denoising.`,name:"num_denoising"},{anchor:"transformers.RTDetrV2Config.label_noise_ratio",description:`<strong>label_noise_ratio</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The fraction of denoising labels to which random noise should be added.`,name:"label_noise_ratio"},{anchor:"transformers.RTDetrV2Config.box_noise_scale",description:`<strong>box_noise_scale</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Scale or magnitude of noise to be added to the bounding boxes.`,name:"box_noise_scale"},{anchor:"transformers.RTDetrV2Config.learn_initial_query",description:`<strong>learn_initial_query</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Indicates whether the initial query embeddings for the decoder should be learned during training`,name:"learn_initial_query"},{anchor:"transformers.RTDetrV2Config.anchor_image_size",description:`<strong>anchor_image_size</strong> (<code>tuple[int, int]</code>, <em>optional</em>) &#x2014;
Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.`,name:"anchor_image_size"},{anchor:"transformers.RTDetrV2Config.with_box_refine",description:`<strong>with_box_refine</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
based on the predictions from the previous layer.`,name:"with_box_refine"},{anchor:"transformers.RTDetrV2Config.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the architecture has an encoder decoder structure.`,name:"is_encoder_decoder"},{anchor:"transformers.RTDetrV2Config.matcher_alpha",description:`<strong>matcher_alpha</strong> (<code>float</code>, <em>optional</em>, defaults to 0.25) &#x2014;
Parameter alpha used by the Hungarian Matcher.`,name:"matcher_alpha"},{anchor:"transformers.RTDetrV2Config.matcher_gamma",description:`<strong>matcher_gamma</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
Parameter gamma used by the Hungarian Matcher.`,name:"matcher_gamma"},{anchor:"transformers.RTDetrV2Config.matcher_class_cost",description:`<strong>matcher_class_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
The relative weight of the class loss used by the Hungarian Matcher.`,name:"matcher_class_cost"},{anchor:"transformers.RTDetrV2Config.matcher_bbox_cost",description:`<strong>matcher_bbox_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 5.0) &#x2014;
The relative weight of the bounding box loss used by the Hungarian Matcher.`,name:"matcher_bbox_cost"},{anchor:"transformers.RTDetrV2Config.matcher_giou_cost",description:`<strong>matcher_giou_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
The relative weight of the giou loss of used by the Hungarian Matcher.`,name:"matcher_giou_cost"},{anchor:"transformers.RTDetrV2Config.use_focal_loss",description:`<strong>use_focal_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Parameter informing if focal loss should be used.`,name:"use_focal_loss"},{anchor:"transformers.RTDetrV2Config.auxiliary_loss",description:`<strong>auxiliary_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether auxiliary decoding losses (loss at each decoder layer) are to be used.`,name:"auxiliary_loss"},{anchor:"transformers.RTDetrV2Config.focal_loss_alpha",description:`<strong>focal_loss_alpha</strong> (<code>float</code>, <em>optional</em>, defaults to 0.75) &#x2014;
Parameter alpha used to compute the focal loss.`,name:"focal_loss_alpha"},{anchor:"transformers.RTDetrV2Config.focal_loss_gamma",description:`<strong>focal_loss_gamma</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
Parameter gamma used to compute the focal loss.`,name:"focal_loss_gamma"},{anchor:"transformers.RTDetrV2Config.weight_loss_vfl",description:`<strong>weight_loss_vfl</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Relative weight of the varifocal loss in the object detection loss.`,name:"weight_loss_vfl"},{anchor:"transformers.RTDetrV2Config.weight_loss_bbox",description:`<strong>weight_loss_bbox</strong> (<code>float</code>, <em>optional</em>, defaults to 5.0) &#x2014;
Relative weight of the L1 bounding box loss in the object detection loss.`,name:"weight_loss_bbox"},{anchor:"transformers.RTDetrV2Config.weight_loss_giou",description:`<strong>weight_loss_giou</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
Relative weight of the generalized IoU loss in the object detection loss.`,name:"weight_loss_giou"},{anchor:"transformers.RTDetrV2Config.eos_coefficient",description:`<strong>eos_coefficient</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0001) &#x2014;
Relative classification weight of the &#x2018;no-object&#x2019; class in the object detection loss.`,name:"eos_coefficient"},{anchor:"transformers.RTDetrV2Config.decoder_n_levels",description:`<strong>decoder_n_levels</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The number of feature levels used by the decoder.`,name:"decoder_n_levels"},{anchor:"transformers.RTDetrV2Config.decoder_offset_scale",description:`<strong>decoder_offset_scale</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Scaling factor applied to the attention offsets in the decoder.`,name:"decoder_offset_scale"},{anchor:"transformers.RTDetrV2Config.decoder_method",description:`<strong>decoder_method</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;default&quot;</code>) &#x2014;
The method to use for the decoder: <code>&quot;default&quot;</code> or <code>&quot;discrete&quot;</code>.`,name:"decoder_method"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py#L31"}}),F=new wt({props:{anchor:"transformers.RTDetrV2Config.example",$$slots:{default:[Kt]},$$scope:{ctx:J}}}),re=new De({props:{name:"from_backbone_configs",anchor:"transformers.RTDetrV2Config.from_backbone_configs",parameters:[{name:"backbone_config",val:": PretrainedConfig"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RTDetrV2Config.from_backbone_configs.backbone_config",description:`<strong>backbone_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>) &#x2014;
The backbone configuration.`,name:"backbone_config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/configuration_rt_detr_v2.py#L369",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>An instance of a configuration object</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config"
>RTDetrV2Config</a></p>
`}}),ie=new fe({props:{title:"RTDetrV2Model",local:"transformers.RTDetrV2Model",headingTag:"h2"}}),le=new De({props:{name:"class transformers.RTDetrV2Model",anchor:"transformers.RTDetrV2Model",parameters:[{name:"config",val:": RTDetrV2Config"}],parametersDescription:[{anchor:"transformers.RTDetrV2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config">RTDetrV2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py#L1367"}}),ce=new De({props:{name:"forward",anchor:"transformers.RTDetrV2Model.forward",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[list[dict]] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RTDetrV2Model.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<code>image_processor_class</code>. See <code>image_processor_class.__call__</code> for details (<code>processor_class</code> uses
<code>image_processor_class</code> for processing images).`,name:"pixel_values"},{anchor:"transformers.RTDetrV2Model.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.RTDetrV2Model.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.RTDetrV2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
can choose to directly pass a flattened representation of an image.`,name:"inputs_embeds"},{anchor:"transformers.RTDetrV2Model.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
embedded representation.`,name:"decoder_inputs_embeds"},{anchor:"transformers.RTDetrV2Model.forward.labels",description:`<strong>labels</strong> (<code>list[Dict]</code> of len <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
following 2 keys: &#x2018;class_labels&#x2019; and &#x2018;boxes&#x2019; (the class labels and bounding boxes of an image in the batch
respectively). The class labels themselves should be a <code>torch.LongTensor</code> of len <code>(number of bounding boxes in the image,)</code> and the boxes a <code>torch.FloatTensor</code> of shape <code>(number of bounding boxes in the image, 4)</code>.`,name:"labels"},{anchor:"transformers.RTDetrV2Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RTDetrV2Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RTDetrV2Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py#L1480",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config"
>RTDetrV2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>intermediate_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, num_queries, hidden_size)</code>) — Stacked intermediate hidden states (output of each layer of the decoder).</p>
</li>
<li>
<p><strong>intermediate_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, sequence_length, config.num_labels)</code>) — Stacked intermediate logits (logits of each layer of the decoder).</p>
</li>
<li>
<p><strong>intermediate_reference_points</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, num_queries, 4)</code>) — Stacked intermediate reference points (reference points of each layer of the decoder).</p>
</li>
<li>
<p><strong>intermediate_predicted_corners</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, num_queries, 4)</code>) — Stacked intermediate predicted corners (predicted corners of each layer of the decoder).</p>
</li>
<li>
<p><strong>initial_reference_points</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, 4)</code>) — Initial reference points used for the first decoder layer.</p>
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
<li>
<p><strong>init_reference_points</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, 4)</code>) — Initial reference points sent through the Transformer decoder.</p>
</li>
<li>
<p><strong>enc_topk_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) — Predicted bounding boxes scores where the top <code>config.two_stage_num_proposals</code> scoring bounding boxes are
picked as region proposals in the encoder stage. Output of bounding box binary classification (i.e.
foreground and background).</p>
</li>
<li>
<p><strong>enc_topk_bboxes</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>) — Logits of predicted bounding boxes coordinates in the encoder stage.</p>
</li>
<li>
<p><strong>enc_outputs_class</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>, <em>optional</em>, returned when <code>config.with_box_refine=True</code> and <code>config.two_stage=True</code>) — Predicted bounding boxes scores where the top <code>config.two_stage_num_proposals</code> scoring bounding boxes are
picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
foreground and background).</p>
</li>
<li>
<p><strong>enc_outputs_coord_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>, returned when <code>config.with_box_refine=True</code> and <code>config.two_stage=True</code>) — Logits of predicted bounding boxes coordinates in the first stage.</p>
</li>
<li>
<p><strong>denoising_meta_values</strong> (<code>dict</code>, <em>optional</em>, defaults to <code>None</code>) — Extra dictionary for the denoising related values.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),I=new Xt({props:{$$slots:{default:[eo]},$$scope:{ctx:J}}}),q=new wt({props:{anchor:"transformers.RTDetrV2Model.forward.example",$$slots:{default:[to]},$$scope:{ctx:J}}}),de=new fe({props:{title:"RTDetrV2ForObjectDetection",local:"transformers.RTDetrV2ForObjectDetection",headingTag:"h2"}}),pe=new De({props:{name:"class transformers.RTDetrV2ForObjectDetection",anchor:"transformers.RTDetrV2ForObjectDetection",parameters:[{name:"config",val:": RTDetrV2Config"}],parametersDescription:[{anchor:"transformers.RTDetrV2ForObjectDetection.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config">RTDetrV2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py#L1810"}}),me=new De({props:{name:"forward",anchor:"transformers.RTDetrV2ForObjectDetection.forward",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[list[dict]] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RTDetrV2ForObjectDetection.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<code>image_processor_class</code>. See <code>image_processor_class.__call__</code> for details (<code>processor_class</code> uses
<code>image_processor_class</code> for processing images).`,name:"pixel_values"},{anchor:"transformers.RTDetrV2ForObjectDetection.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.RTDetrV2ForObjectDetection.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.RTDetrV2ForObjectDetection.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
can choose to directly pass a flattened representation of an image.`,name:"inputs_embeds"},{anchor:"transformers.RTDetrV2ForObjectDetection.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
embedded representation.`,name:"decoder_inputs_embeds"},{anchor:"transformers.RTDetrV2ForObjectDetection.forward.labels",description:`<strong>labels</strong> (<code>list[Dict]</code> of len <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
following 2 keys: &#x2018;class_labels&#x2019; and &#x2018;boxes&#x2019; (the class labels and bounding boxes of an image in the batch
respectively). The class labels themselves should be a <code>torch.LongTensor</code> of len <code>(number of bounding boxes in the image,)</code> and the boxes a <code>torch.FloatTensor</code> of shape <code>(number of bounding boxes in the image, 4)</code>.`,name:"labels"},{anchor:"transformers.RTDetrV2ForObjectDetection.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RTDetrV2ForObjectDetection.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RTDetrV2ForObjectDetection.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py#L1841",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ObjectDetectionOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/rt_detr_v2#transformers.RTDetrV2Config"
>RTDetrV2Config</a>) and inputs.</p>
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
possible padding). You can use <code>~RTDetrV2ImageProcessor.post_process_object_detection</code> to retrieve the
unnormalized (absolute) bounding boxes.</p>
</li>
<li>
<p><strong>auxiliary_outputs</strong> (<code>list[Dict]</code>, <em>optional</em>) — Optional, only returned when auxiliary losses are activated (i.e. <code>config.auxiliary_loss</code> is set to <code>True</code>)
and labels are provided. It is a list of dictionaries containing the two above keys (<code>logits</code> and
<code>pred_boxes</code>) for each decoder layer.</p>
</li>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>intermediate_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, num_queries, hidden_size)</code>) — Stacked intermediate hidden states (output of each layer of the decoder).</p>
</li>
<li>
<p><strong>intermediate_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, num_queries, config.num_labels)</code>) — Stacked intermediate logits (logits of each layer of the decoder).</p>
</li>
<li>
<p><strong>intermediate_reference_points</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, num_queries, 4)</code>) — Stacked intermediate reference points (reference points of each layer of the decoder).</p>
</li>
<li>
<p><strong>intermediate_predicted_corners</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, num_queries, 4)</code>) — Stacked intermediate predicted corners (predicted corners of each layer of the decoder).</p>
</li>
<li>
<p><strong>initial_reference_points</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.decoder_layers, num_queries, 4)</code>) — Stacked initial reference points (initial reference points of each layer of the decoder).</p>
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
<li>
<p><strong>init_reference_points</strong> (<code>torch.FloatTensor</code> of shape  <code>(batch_size, num_queries, 4)</code>) — Initial reference points sent through the Transformer decoder.</p>
</li>
<li>
<p><strong>enc_topk_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>, <em>optional</em>, returned when <code>config.with_box_refine=True</code> and <code>config.two_stage=True</code>) — Logits of predicted bounding boxes coordinates in the encoder.</p>
</li>
<li>
<p><strong>enc_topk_bboxes</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>, returned when <code>config.with_box_refine=True</code> and <code>config.two_stage=True</code>) — Logits of predicted bounding boxes coordinates in the encoder.</p>
</li>
<li>
<p><strong>enc_outputs_class</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>, <em>optional</em>, returned when <code>config.with_box_refine=True</code> and <code>config.two_stage=True</code>) — Predicted bounding boxes scores where the top <code>config.two_stage_num_proposals</code> scoring bounding boxes are
picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
foreground and background).</p>
</li>
<li>
<p><strong>enc_outputs_coord_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, 4)</code>, <em>optional</em>, returned when <code>config.with_box_refine=True</code> and <code>config.two_stage=True</code>) — Logits of predicted bounding boxes coordinates in the first stage.</p>
</li>
<li>
<p><strong>denoising_meta_values</strong> (<code>dict</code>, <em>optional</em>, defaults to <code>None</code>) — Extra dictionary for the denoising related values</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.rt_detr_v2.modeling_rt_detr_v2.RTDetrV2ObjectDetectionOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),B=new Xt({props:{$$slots:{default:[oo]},$$scope:{ctx:J}}}),W=new wt({props:{anchor:"transformers.RTDetrV2ForObjectDetection.forward.example",$$slots:{default:[no]},$$scope:{ctx:J}}}),he=new Ot({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/rt_detr_v2.md"}}),{c(){s=l("meta"),v=a(),p=l("p"),m=a(),y=l("p"),y.innerHTML=i,M=a(),u(H.$$.fragment),Ce=a(),z=l("div"),z.innerHTML=Mt,$e=a(),u(G.$$.fragment),Ue=a(),X=l("p"),X.innerHTML=jt,ze=a(),S=l("p"),S.textContent=xt,Fe=a(),Q=l("p"),Q.textContent=Rt,Ze=a(),P=l("p"),P.innerHTML=Vt,Ie=a(),L=l("p"),L.innerHTML=Dt,qe=a(),u(A.$$.fragment),Be=a(),Y=l("p"),Y.textContent=Jt,We=a(),O=l("ul"),O.innerHTML=kt,Ee=a(),u(K.$$.fragment),Ne=a(),u(ee.$$.fragment),He=a(),te=l("p"),te.textContent=Ct,Ge=a(),u(oe.$$.fragment),Xe=a(),ne=l("ul"),ne.innerHTML=$t,Se=a(),u(se.$$.fragment),Qe=a(),w=l("div"),u(ae.$$.fragment),ot=a(),ge=l("p"),ge.innerHTML=Ut,nt=a(),_e=l("p"),_e.innerHTML=zt,st=a(),be=l("p"),be.innerHTML=Ft,at=a(),u(F.$$.fragment),rt=a(),Z=l("div"),u(re.$$.fragment),it=a(),Te=l("p"),Te.innerHTML=Zt,Pe=a(),u(ie.$$.fragment),Le=a(),j=l("div"),u(le.$$.fragment),lt=a(),ye=l("p"),ye.textContent=It,ct=a(),ve=l("p"),ve.innerHTML=qt,dt=a(),we=l("p"),we.innerHTML=Bt,pt=a(),k=l("div"),u(ce.$$.fragment),mt=a(),Me=l("p"),Me.innerHTML=Wt,ht=a(),u(I.$$.fragment),ut=a(),u(q.$$.fragment),Ae=a(),u(de.$$.fragment),Ye=a(),x=l("div"),u(pe.$$.fragment),ft=a(),je=l("p"),je.textContent=Et,gt=a(),xe=l("p"),xe.innerHTML=Nt,_t=a(),Re=l("p"),Re.innerHTML=Ht,bt=a(),C=l("div"),u(me.$$.fragment),Tt=a(),Ve=l("p"),Ve.innerHTML=Gt,yt=a(),u(B.$$.fragment),vt=a(),u(W.$$.fragment),Oe=a(),u(he.$$.fragment),Ke=a(),Je=l("p"),this.h()},l(e){const t=At("svelte-u9bgzb",document.head);s=c(t,"META",{name:!0,content:!0}),t.forEach(o),v=r(e),p=c(e,"P",{}),N(p).forEach(o),m=r(e),y=c(e,"P",{"data-svelte-h":!0}),h(y)!=="svelte-19wormi"&&(y.innerHTML=i),M=r(e),f(H.$$.fragment,e),Ce=r(e),z=c(e,"DIV",{class:!0,"data-svelte-h":!0}),h(z)!=="svelte-13t8s2t"&&(z.innerHTML=Mt),$e=r(e),f(G.$$.fragment,e),Ue=r(e),X=c(e,"P",{"data-svelte-h":!0}),h(X)!=="svelte-yioiqv"&&(X.innerHTML=jt),ze=r(e),S=c(e,"P",{"data-svelte-h":!0}),h(S)!=="svelte-1rimzbz"&&(S.textContent=xt),Fe=r(e),Q=c(e,"P",{"data-svelte-h":!0}),h(Q)!=="svelte-vfdo9a"&&(Q.textContent=Rt),Ze=r(e),P=c(e,"P",{"data-svelte-h":!0}),h(P)!=="svelte-shd4nm"&&(P.innerHTML=Vt),Ie=r(e),L=c(e,"P",{"data-svelte-h":!0}),h(L)!=="svelte-1st0rtl"&&(L.innerHTML=Dt),qe=r(e),f(A.$$.fragment,e),Be=r(e),Y=c(e,"P",{"data-svelte-h":!0}),h(Y)!=="svelte-1wqfppu"&&(Y.textContent=Jt),We=r(e),O=c(e,"UL",{"data-svelte-h":!0}),h(O)!=="svelte-61wopf"&&(O.innerHTML=kt),Ee=r(e),f(K.$$.fragment,e),Ne=r(e),f(ee.$$.fragment,e),He=r(e),te=c(e,"P",{"data-svelte-h":!0}),h(te)!=="svelte-12k2k33"&&(te.textContent=Ct),Ge=r(e),f(oe.$$.fragment,e),Xe=r(e),ne=c(e,"UL",{"data-svelte-h":!0}),h(ne)!=="svelte-16i1d6q"&&(ne.innerHTML=$t),Se=r(e),f(se.$$.fragment,e),Qe=r(e),w=c(e,"DIV",{class:!0});var R=N(w);f(ae.$$.fragment,R),ot=r(R),ge=c(R,"P",{"data-svelte-h":!0}),h(ge)!=="svelte-1h53rbb"&&(ge.innerHTML=Ut),nt=r(R),_e=c(R,"P",{"data-svelte-h":!0}),h(_e)!=="svelte-s96u4h"&&(_e.innerHTML=zt),st=r(R),be=c(R,"P",{"data-svelte-h":!0}),h(be)!=="svelte-1ek1ss9"&&(be.innerHTML=Ft),at=r(R),f(F.$$.fragment,R),rt=r(R),Z=c(R,"DIV",{class:!0});var ue=N(Z);f(re.$$.fragment,ue),it=r(ue),Te=c(ue,"P",{"data-svelte-h":!0}),h(Te)!=="svelte-11gg410"&&(Te.innerHTML=Zt),ue.forEach(o),R.forEach(o),Pe=r(e),f(ie.$$.fragment,e),Le=r(e),j=c(e,"DIV",{class:!0});var V=N(j);f(le.$$.fragment,V),lt=r(V),ye=c(V,"P",{"data-svelte-h":!0}),h(ye)!=="svelte-121t6ur"&&(ye.textContent=It),ct=r(V),ve=c(V,"P",{"data-svelte-h":!0}),h(ve)!=="svelte-q52n56"&&(ve.innerHTML=qt),dt=r(V),we=c(V,"P",{"data-svelte-h":!0}),h(we)!=="svelte-hswkmf"&&(we.innerHTML=Bt),pt=r(V),k=c(V,"DIV",{class:!0});var $=N(k);f(ce.$$.fragment,$),mt=r($),Me=c($,"P",{"data-svelte-h":!0}),h(Me)!=="svelte-1kwdhpk"&&(Me.innerHTML=Wt),ht=r($),f(I.$$.fragment,$),ut=r($),f(q.$$.fragment,$),$.forEach(o),V.forEach(o),Ae=r(e),f(de.$$.fragment,e),Ye=r(e),x=c(e,"DIV",{class:!0});var D=N(x);f(pe.$$.fragment,D),ft=r(D),je=c(D,"P",{"data-svelte-h":!0}),h(je)!=="svelte-xszdn5"&&(je.textContent=Et),gt=r(D),xe=c(D,"P",{"data-svelte-h":!0}),h(xe)!=="svelte-q52n56"&&(xe.innerHTML=Nt),_t=r(D),Re=c(D,"P",{"data-svelte-h":!0}),h(Re)!=="svelte-hswkmf"&&(Re.innerHTML=Ht),bt=r(D),C=c(D,"DIV",{class:!0});var E=N(C);f(me.$$.fragment,E),Tt=r(E),Ve=c(E,"P",{"data-svelte-h":!0}),h(Ve)!=="svelte-1n4mys"&&(Ve.innerHTML=Gt),yt=r(E),f(B.$$.fragment,E),vt=r(E),f(W.$$.fragment,E),E.forEach(o),D.forEach(o),Oe=r(e),f(he.$$.fragment,e),Ke=r(e),Je=c(e,"P",{}),N(Je).forEach(o),this.h()},h(){U(s,"name","hf:doc:metadata"),U(s,"content",ao),U(z,"class","flex flex-wrap space-x-1"),U(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){d(document.head,s),n(e,v,t),n(e,p,t),n(e,m,t),n(e,y,t),n(e,M,t),g(H,e,t),n(e,Ce,t),n(e,z,t),n(e,$e,t),g(G,e,t),n(e,Ue,t),n(e,X,t),n(e,ze,t),n(e,S,t),n(e,Fe,t),n(e,Q,t),n(e,Ze,t),n(e,P,t),n(e,Ie,t),n(e,L,t),n(e,qe,t),g(A,e,t),n(e,Be,t),n(e,Y,t),n(e,We,t),n(e,O,t),n(e,Ee,t),g(K,e,t),n(e,Ne,t),g(ee,e,t),n(e,He,t),n(e,te,t),n(e,Ge,t),g(oe,e,t),n(e,Xe,t),n(e,ne,t),n(e,Se,t),g(se,e,t),n(e,Qe,t),n(e,w,t),g(ae,w,null),d(w,ot),d(w,ge),d(w,nt),d(w,_e),d(w,st),d(w,be),d(w,at),g(F,w,null),d(w,rt),d(w,Z),g(re,Z,null),d(Z,it),d(Z,Te),n(e,Pe,t),g(ie,e,t),n(e,Le,t),n(e,j,t),g(le,j,null),d(j,lt),d(j,ye),d(j,ct),d(j,ve),d(j,dt),d(j,we),d(j,pt),d(j,k),g(ce,k,null),d(k,mt),d(k,Me),d(k,ht),g(I,k,null),d(k,ut),g(q,k,null),n(e,Ae,t),g(de,e,t),n(e,Ye,t),n(e,x,t),g(pe,x,null),d(x,ft),d(x,je),d(x,gt),d(x,xe),d(x,_t),d(x,Re),d(x,bt),d(x,C),g(me,C,null),d(C,Tt),d(C,Ve),d(C,yt),g(B,C,null),d(C,vt),g(W,C,null),n(e,Oe,t),g(he,e,t),n(e,Ke,t),n(e,Je,t),et=!0},p(e,[t]){const R={};t&2&&(R.$$scope={dirty:t,ctx:e}),F.$set(R);const ue={};t&2&&(ue.$$scope={dirty:t,ctx:e}),I.$set(ue);const V={};t&2&&(V.$$scope={dirty:t,ctx:e}),q.$set(V);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),B.$set($);const D={};t&2&&(D.$$scope={dirty:t,ctx:e}),W.$set(D)},i(e){et||(_(H.$$.fragment,e),_(G.$$.fragment,e),_(A.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(oe.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(F.$$.fragment,e),_(re.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),_(ce.$$.fragment,e),_(I.$$.fragment,e),_(q.$$.fragment,e),_(de.$$.fragment,e),_(pe.$$.fragment,e),_(me.$$.fragment,e),_(B.$$.fragment,e),_(W.$$.fragment,e),_(he.$$.fragment,e),et=!0)},o(e){b(H.$$.fragment,e),b(G.$$.fragment,e),b(A.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(oe.$$.fragment,e),b(se.$$.fragment,e),b(ae.$$.fragment,e),b(F.$$.fragment,e),b(re.$$.fragment,e),b(ie.$$.fragment,e),b(le.$$.fragment,e),b(ce.$$.fragment,e),b(I.$$.fragment,e),b(q.$$.fragment,e),b(de.$$.fragment,e),b(pe.$$.fragment,e),b(me.$$.fragment,e),b(B.$$.fragment,e),b(W.$$.fragment,e),b(he.$$.fragment,e),et=!1},d(e){e&&(o(v),o(p),o(m),o(y),o(M),o(Ce),o(z),o($e),o(Ue),o(X),o(ze),o(S),o(Fe),o(Q),o(Ze),o(P),o(Ie),o(L),o(qe),o(Be),o(Y),o(We),o(O),o(Ee),o(Ne),o(He),o(te),o(Ge),o(Xe),o(ne),o(Se),o(Qe),o(w),o(Pe),o(Le),o(j),o(Ae),o(Ye),o(x),o(Oe),o(Ke),o(Je)),o(s),T(H,e),T(G,e),T(A,e),T(K,e),T(ee,e),T(oe,e),T(se,e),T(ae),T(F),T(re),T(ie,e),T(le),T(ce),T(I),T(q),T(de,e),T(pe),T(me),T(B),T(W),T(he,e)}}}const ao='{"title":"RT-DETRv2","local":"rt-detrv2","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"RTDetrV2Config","local":"transformers.RTDetrV2Config","sections":[],"depth":2},{"title":"RTDetrV2Model","local":"transformers.RTDetrV2Model","sections":[],"depth":2},{"title":"RTDetrV2ForObjectDetection","local":"transformers.RTDetrV2ForObjectDetection","sections":[],"depth":2}],"depth":1}';function ro(J){return Qt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class go extends Pt{constructor(s){super(),Lt(this,s,ro,so,St,{})}}export{go as component};
