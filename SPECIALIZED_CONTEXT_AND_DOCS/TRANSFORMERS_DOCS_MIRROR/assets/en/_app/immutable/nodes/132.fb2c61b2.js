import{s as _o,o as bo,n as We}from"../chunks/scheduler.18a86fab.js";import{S as yo,i as wo,g as c,s,r as h,A as Mo,h as d,f as n,c as a,j as N,x as y,u,k as E,y as l,a as r,v as f,d as g,t as _,w as b}from"../chunks/index.98837b22.js";import{T as fo}from"../chunks/Tip.77304350.js";import{D as ge}from"../chunks/Docstring.a1ef7999.js";import{C as Ke}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as go}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as _e,E as vo}from"../chunks/getInferenceSnippets.06c2775f.js";function To(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(p){t=d(p,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(p,m){r(p,t,m)},p:We,d(p){p&&n(t)}}}function jo(k){let t,M="Examples:",p,m,w;return m=new Ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyMERGaW5lTW9kZWwlMEFmcm9tJTIwUElMJTIwaW1wb3J0JTIwSW1hZ2UlMEFpbXBvcnQlMjByZXF1ZXN0cyUwQSUwQXVybCUyMCUzRCUyMCUyMmh0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEElMEFpbWFnZV9wcm9jZXNzb3IlMjAlM0QlMjBBdXRvSW1hZ2VQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMlBla2luZ1UlMkZERmluZV9yNTB2ZCUyMiklMEFtb2RlbCUyMCUzRCUyMERGaW5lTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMlBla2luZ1UlMkZERmluZV9yNTB2ZCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IoaW1hZ2VzJTNEaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBbGlzdChsYXN0X2hpZGRlbl9zdGF0ZXMuc2hhcGUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, DFineModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;PekingU/DFine_r50vd&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DFineModel.from_pretrained(<span class="hljs-string">&quot;PekingU/DFine_r50vd&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">300</span>, <span class="hljs-number">256</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,p=s(),h(m.$$.fragment)},l(i){t=d(i,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=M),p=a(i),u(m.$$.fragment,i)},m(i,x){r(i,t,x),r(i,p,x),f(m,i,x),w=!0},p:We,i(i){w||(g(m.$$.fragment,i),w=!0)},o(i){_(m.$$.fragment,i),w=!1},d(i){i&&(n(t),n(p)),b(m,i)}}}function xo(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(p){t=d(p,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(p,m){r(p,t,m)},p:We,d(p){p&&n(t)}}}function Fo(k){let t,M="Example:",p,m,w;return m=new Ke({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzLmltYWdlX3V0aWxzJTIwaW1wb3J0JTIwbG9hZF9pbWFnZSUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBBdXRvSW1hZ2VQcm9jZXNzb3IlMkMlMjBERmluZUZvck9iamVjdERldGVjdGlvbiUwQSUwQXVybCUyMCUzRCUyMCUyMmh0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMGxvYWRfaW1hZ2UodXJsKSUwQSUwQWltYWdlX3Byb2Nlc3NvciUyMCUzRCUyMEF1dG9JbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIydXN0Yy1jb21tdW5pdHklMkZkZmluZS14bGFyZ2UtY29jbyUyMiklMEFtb2RlbCUyMCUzRCUyMERGaW5lRm9yT2JqZWN0RGV0ZWN0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ1c3RjLWNvbW11bml0eSUyRmRmaW5lLXhsYXJnZS1jb2NvJTIyKSUwQSUwQSUyMyUyMHByZXBhcmUlMjBpbWFnZSUyMGZvciUyMHRoZSUyMG1vZGVsJTBBaW5wdXRzJTIwJTNEJTIwaW1hZ2VfcHJvY2Vzc29yKGltYWdlcyUzRGltYWdlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBmb3J3YXJkJTIwcGFzcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQWxpc3QobG9naXRzLnNoYXBlKSUwQSUwQWJveGVzJTIwJTNEJTIwb3V0cHV0cy5wcmVkX2JveGVzJTBBbGlzdChib3hlcy5zaGFwZSklMEElMEElMjMlMjBjb252ZXJ0JTIwb3V0cHV0cyUyMChib3VuZGluZyUyMGJveGVzJTIwYW5kJTIwY2xhc3MlMjBsb2dpdHMpJTIwdG8lMjBQYXNjYWwlMjBWT0MlMjBmb3JtYXQlMjAoeG1pbiUyQyUyMHltaW4lMkMlMjB4bWF4JTJDJTIweW1heCklMEF0YXJnZXRfc2l6ZXMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCaW1hZ2Uuc2l6ZSU1QiUzQSUzQS0xJTVEJTVEKSUwQXJlc3VsdHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IucG9zdF9wcm9jZXNzX29iamVjdF9kZXRlY3Rpb24ob3V0cHV0cyUyQyUyMHRocmVzaG9sZCUzRDAuOSUyQyUyMHRhcmdldF9zaXplcyUzRHRhcmdldF9zaXplcyklMEFyZXN1bHQlMjAlM0QlMjByZXN1bHRzJTVCMCU1RCUyMCUyMCUyMyUyMGZpcnN0JTIwaW1hZ2UlMjBpbiUyMGJhdGNoJTBBJTBBZm9yJTIwc2NvcmUlMkMlMjBsYWJlbCUyQyUyMGJveCUyMGluJTIwemlwKHJlc3VsdCU1QiUyMnNjb3JlcyUyMiU1RCUyQyUyMHJlc3VsdCU1QiUyMmxhYmVscyUyMiU1RCUyQyUyMHJlc3VsdCU1QiUyMmJveGVzJTIyJTVEKSUzQSUwQSUyMCUyMCUyMCUyMGJveCUyMCUzRCUyMCU1QnJvdW5kKGklMkMlMjAyKSUyMGZvciUyMGklMjBpbiUyMGJveC50b2xpc3QoKSU1RCUwQSUyMCUyMCUyMCUyMHByaW50KCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGYlMjJEZXRlY3RlZCUyMCU3Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QmxhYmVsLml0ZW0oKSU1RCU3RCUyMHdpdGglMjBjb25maWRlbmNlJTIwJTIyJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwZiUyMiU3QnJvdW5kKHNjb3JlLml0ZW0oKSUyQyUyMDMpJTdEJTIwYXQlMjBsb2NhdGlvbiUyMCU3QmJveCU3RCUyMiUwQSUyMCUyMCUyMCUyMCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers.image_utils <span class="hljs-keyword">import</span> load_image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, DFineForObjectDetection

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = load_image(url)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;ustc-community/dfine-xlarge-coco&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DFineForObjectDetection.from_pretrained(<span class="hljs-string">&quot;ustc-community/dfine-xlarge-coco&quot;</span>)

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
<span class="hljs-meta">&gt;&gt;&gt; </span>results = image_processor.post_process_object_detection(outputs, threshold=<span class="hljs-number">0.9</span>, target_sizes=target_sizes)
<span class="hljs-meta">&gt;&gt;&gt; </span>result = results[<span class="hljs-number">0</span>]  <span class="hljs-comment"># first image in batch</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> score, label, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(result[<span class="hljs-string">&quot;scores&quot;</span>], result[<span class="hljs-string">&quot;labels&quot;</span>], result[<span class="hljs-string">&quot;boxes&quot;</span>]):
<span class="hljs-meta">... </span>    box = [<span class="hljs-built_in">round</span>(i, <span class="hljs-number">2</span>) <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> box.tolist()]
<span class="hljs-meta">... </span>    <span class="hljs-built_in">print</span>(
<span class="hljs-meta">... </span>        <span class="hljs-string">f&quot;Detected <span class="hljs-subst">{model.config.id2label[label.item()]}</span> with confidence &quot;</span>
<span class="hljs-meta">... </span>        <span class="hljs-string">f&quot;<span class="hljs-subst">{<span class="hljs-built_in">round</span>(score.item(), <span class="hljs-number">3</span>)}</span> at location <span class="hljs-subst">{box}</span>&quot;</span>
<span class="hljs-meta">... </span>    )
Detected cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.958</span> at location [<span class="hljs-number">344.49</span>, <span class="hljs-number">23.4</span>, <span class="hljs-number">639.84</span>, <span class="hljs-number">374.27</span>]
Detected cat <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.956</span> at location [<span class="hljs-number">11.71</span>, <span class="hljs-number">53.52</span>, <span class="hljs-number">316.64</span>, <span class="hljs-number">472.33</span>]
Detected remote <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.947</span> at location [<span class="hljs-number">40.46</span>, <span class="hljs-number">73.7</span>, <span class="hljs-number">175.62</span>, <span class="hljs-number">117.57</span>]
Detected sofa <span class="hljs-keyword">with</span> confidence <span class="hljs-number">0.918</span> at location [<span class="hljs-number">0.59</span>, <span class="hljs-number">1.88</span>, <span class="hljs-number">640.25</span>, <span class="hljs-number">474.74</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,p=s(),h(m.$$.fragment)},l(i){t=d(i,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=M),p=a(i),u(m.$$.fragment,i)},m(i,x){r(i,t,x),r(i,p,x),f(m,i,x),w=!0},p:We,i(i){w||(g(m.$$.fragment,i),w=!0)},o(i){_(m.$$.fragment,i),w=!1},d(i){i&&(n(t),n(p)),b(m,i)}}}function Co(k){let t,M,p,m,w,i="<em>This model was released on 2024-10-17 and added to Hugging Face Transformers on 2025-04-29.</em>",x,V,ye,B,we,H,eo=`The D-FINE model was proposed in <a href="https://huggingface.co/papers/2410.13842" rel="nofollow">D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement</a> by
Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, Feng Wu`,Me,G,oo="The abstract from the paper is the following:",ve,S,no=`<em>We introduce D-FINE, a powerful real-time object detector that achieves outstanding localization precision by redefining the bounding box regression task in DETR models. D-FINE comprises two key components: Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD).
FDR transforms the regression process from predicting fixed coordinates to iteratively refining probability distributions, providing a fine-grained intermediate representation that significantly enhances localization accuracy. GO-LSD is a bidirectional optimization strategy that transfers localization knowledge from refined distributions to shallower layers through self-distillation, while also simplifying the residual prediction tasks for deeper layers. Additionally, D-FINE incorporates lightweight optimizations in computationally intensive modules and operations, achieving a better balance between speed and accuracy. Specifically, D-FINE-L / X achieves 54.0% / 55.8% AP on the COCO dataset at 124 / 78 FPS on an NVIDIA T4 GPU. When pretrained on Objects365, D-FINE-L / X attains 57.1% / 59.3% AP, surpassing all existing real-time detectors. Furthermore, our method significantly enhances the performance of a wide range of DETR models by up to 5.3% AP with negligible extra parameters and training costs. Our code and pretrained models: this https URL.</em>`,Te,Q,to=`This model was contributed by <a href="https://github.com/VladOS95-cyber" rel="nofollow">VladOS95-cyber</a>.
The original code can be found <a href="https://github.com/Peterande/D-FINE" rel="nofollow">here</a>.`,je,X,xe,L,Fe,O,Ce,U,P,Ze,re,so=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineModel">DFineModel</a>. It is used to instantiate a D-FINE
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of D-FINE-X-COCO ”<a href="https://huggingface.co/ustc-community/dfine-xlarge-coco%22" rel="nofollow">ustc-community/dfine-xlarge-coco”</a>.
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ie,$,Y,Re,ie,ao=`Instantiate a <a href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig">DFineConfig</a> (or a derived class) from a pre-trained backbone model configuration and DETR model
configuration.`,Ue,A,De,v,K,Ne,le,ro="RT-DETR Model (consisting of a backbone and encoder-decoder) outputting raw hidden states without any head on top.",Ee,ce,io=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ve,de,lo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Be,F,ee,He,pe,co='The <a href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineModel">DFineModel</a> forward method, overrides the <code>__call__</code> special method.',Ge,q,Se,W,ke,oe,ze,T,ne,Qe,me,po=`RT-DETR Model (consisting of a backbone and encoder-decoder) outputting bounding boxes and logits to be further
decoded into scores and classes.`,Xe,he,mo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Le,ue,ho=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Oe,C,te,Pe,fe,uo='The <a href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineForObjectDetection">DFineForObjectDetection</a> forward method, overrides the <code>__call__</code> special method.',Ye,Z,Ae,I,Je,se,$e,be,qe;return V=new _e({props:{title:"D-FINE",local:"d-fine",headingTag:"h1"}}),B=new _e({props:{title:"Overview",local:"overview",headingTag:"h2"}}),X=new _e({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),L=new Ke({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzLmltYWdlX3V0aWxzJTIwaW1wb3J0JTIwbG9hZF9pbWFnZSUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBERmluZUZvck9iamVjdERldGVjdGlvbiUyQyUyMEF1dG9JbWFnZVByb2Nlc3NvciUwQSUwQXVybCUyMCUzRCUyMCdodHRwJTNBJTJGJTJGaW1hZ2VzLmNvY29kYXRhc2V0Lm9yZyUyRnZhbDIwMTclMkYwMDAwMDAwMzk3NjkuanBnJyUwQWltYWdlJTIwJTNEJTIwbG9hZF9pbWFnZSh1cmwpJTBBJTBBaW1hZ2VfcHJvY2Vzc29yJTIwJTNEJTIwQXV0b0ltYWdlUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZCglMjJ1c3RjLWNvbW11bml0eSUyRmRmaW5lX3hfY29jbyUyMiklMEFtb2RlbCUyMCUzRCUyMERGaW5lRm9yT2JqZWN0RGV0ZWN0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ1c3RjLWNvbW11bml0eSUyRmRmaW5lX3hfY29jbyUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IoaW1hZ2VzJTNEaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBcmVzdWx0cyUyMCUzRCUyMGltYWdlX3Byb2Nlc3Nvci5wb3N0X3Byb2Nlc3Nfb2JqZWN0X2RldGVjdGlvbihvdXRwdXRzJTJDJTIwdGFyZ2V0X3NpemVzJTNEJTVCKGltYWdlLmhlaWdodCUyQyUyMGltYWdlLndpZHRoKSU1RCUyQyUyMHRocmVzaG9sZCUzRDAuNSklMEElMEFmb3IlMjByZXN1bHQlMjBpbiUyMHJlc3VsdHMlM0ElMEElMjAlMjAlMjAlMjBmb3IlMjBzY29yZSUyQyUyMGxhYmVsX2lkJTJDJTIwYm94JTIwaW4lMjB6aXAocmVzdWx0JTVCJTIyc2NvcmVzJTIyJTVEJTJDJTIwcmVzdWx0JTVCJTIybGFiZWxzJTIyJTVEJTJDJTIwcmVzdWx0JTVCJTIyYm94ZXMlMjIlNUQpJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc2NvcmUlMkMlMjBsYWJlbCUyMCUzRCUyMHNjb3JlLml0ZW0oKSUyQyUyMGxhYmVsX2lkLml0ZW0oKSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGJveCUyMCUzRCUyMCU1QnJvdW5kKGklMkMlMjAyKSUyMGZvciUyMGklMjBpbiUyMGJveC50b2xpc3QoKSU1RCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMHByaW50KGYlMjIlN0Jtb2RlbC5jb25maWcuaWQybGFiZWwlNUJsYWJlbCU1RCU3RCUzQSUyMCU3QnNjb3JlJTNBLjJmJTdEJTIwJTdCYm94JTdEJTIyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers.image_utils <span class="hljs-keyword">import</span> load_image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DFineForObjectDetection, AutoImageProcessor

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&#x27;http://images.cocodataset.org/val2017/000000039769.jpg&#x27;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = load_image(url)

<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;ustc-community/dfine_x_coco&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DFineForObjectDetection.from_pretrained(<span class="hljs-string">&quot;ustc-community/dfine_x_coco&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>results = image_processor.post_process_object_detection(outputs, target_sizes=[(image.height, image.width)], threshold=<span class="hljs-number">0.5</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> result <span class="hljs-keyword">in</span> results:
<span class="hljs-meta">... </span>    <span class="hljs-keyword">for</span> score, label_id, box <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(result[<span class="hljs-string">&quot;scores&quot;</span>], result[<span class="hljs-string">&quot;labels&quot;</span>], result[<span class="hljs-string">&quot;boxes&quot;</span>]):
<span class="hljs-meta">... </span>        score, label = score.item(), label_id.item()
<span class="hljs-meta">... </span>        box = [<span class="hljs-built_in">round</span>(i, <span class="hljs-number">2</span>) <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> box.tolist()]
<span class="hljs-meta">... </span>        <span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;<span class="hljs-subst">{model.config.id2label[label]}</span>: <span class="hljs-subst">{score:<span class="hljs-number">.2</span>f}</span> <span class="hljs-subst">{box}</span>&quot;</span>)
cat: <span class="hljs-number">0.96</span> [<span class="hljs-number">344.49</span>, <span class="hljs-number">23.4</span>, <span class="hljs-number">639.84</span>, <span class="hljs-number">374.27</span>]
cat: <span class="hljs-number">0.96</span> [<span class="hljs-number">11.71</span>, <span class="hljs-number">53.52</span>, <span class="hljs-number">316.64</span>, <span class="hljs-number">472.33</span>]
remote: <span class="hljs-number">0.95</span> [<span class="hljs-number">40.46</span>, <span class="hljs-number">73.7</span>, <span class="hljs-number">175.62</span>, <span class="hljs-number">117.57</span>]
sofa: <span class="hljs-number">0.92</span> [<span class="hljs-number">0.59</span>, <span class="hljs-number">1.88</span>, <span class="hljs-number">640.25</span>, <span class="hljs-number">474.74</span>]
remote: <span class="hljs-number">0.89</span> [<span class="hljs-number">333.48</span>, <span class="hljs-number">77.04</span>, <span class="hljs-number">370.77</span>, <span class="hljs-number">187.3</span>]`,wrap:!1}}),O=new _e({props:{title:"DFineConfig",local:"transformers.DFineConfig",headingTag:"h2"}}),P=new ge({props:{name:"class transformers.DFineConfig",anchor:"transformers.DFineConfig",parameters:[{name:"initializer_range",val:" = 0.01"},{name:"initializer_bias_prior_prob",val:" = None"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"batch_norm_eps",val:" = 1e-05"},{name:"backbone_config",val:" = None"},{name:"backbone",val:" = None"},{name:"use_pretrained_backbone",val:" = False"},{name:"use_timm_backbone",val:" = False"},{name:"freeze_backbone_batch_norms",val:" = True"},{name:"backbone_kwargs",val:" = None"},{name:"encoder_hidden_dim",val:" = 256"},{name:"encoder_in_channels",val:" = [512, 1024, 2048]"},{name:"feat_strides",val:" = [8, 16, 32]"},{name:"encoder_layers",val:" = 1"},{name:"encoder_ffn_dim",val:" = 1024"},{name:"encoder_attention_heads",val:" = 8"},{name:"dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"encode_proj_layers",val:" = [2]"},{name:"positional_encoding_temperature",val:" = 10000"},{name:"encoder_activation_function",val:" = 'gelu'"},{name:"activation_function",val:" = 'silu'"},{name:"eval_size",val:" = None"},{name:"normalize_before",val:" = False"},{name:"hidden_expansion",val:" = 1.0"},{name:"d_model",val:" = 256"},{name:"num_queries",val:" = 300"},{name:"decoder_in_channels",val:" = [256, 256, 256]"},{name:"decoder_ffn_dim",val:" = 1024"},{name:"num_feature_levels",val:" = 3"},{name:"decoder_n_points",val:" = 4"},{name:"decoder_layers",val:" = 6"},{name:"decoder_attention_heads",val:" = 8"},{name:"decoder_activation_function",val:" = 'relu'"},{name:"attention_dropout",val:" = 0.0"},{name:"num_denoising",val:" = 100"},{name:"label_noise_ratio",val:" = 0.5"},{name:"box_noise_scale",val:" = 1.0"},{name:"learn_initial_query",val:" = False"},{name:"anchor_image_size",val:" = None"},{name:"with_box_refine",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"matcher_alpha",val:" = 0.25"},{name:"matcher_gamma",val:" = 2.0"},{name:"matcher_class_cost",val:" = 2.0"},{name:"matcher_bbox_cost",val:" = 5.0"},{name:"matcher_giou_cost",val:" = 2.0"},{name:"use_focal_loss",val:" = True"},{name:"auxiliary_loss",val:" = True"},{name:"focal_loss_alpha",val:" = 0.75"},{name:"focal_loss_gamma",val:" = 2.0"},{name:"weight_loss_vfl",val:" = 1.0"},{name:"weight_loss_bbox",val:" = 5.0"},{name:"weight_loss_giou",val:" = 2.0"},{name:"weight_loss_fgl",val:" = 0.15"},{name:"weight_loss_ddf",val:" = 1.5"},{name:"eos_coefficient",val:" = 0.0001"},{name:"eval_idx",val:" = -1"},{name:"layer_scale",val:" = 1"},{name:"max_num_bins",val:" = 32"},{name:"reg_scale",val:" = 4.0"},{name:"depth_mult",val:" = 1.0"},{name:"top_prob_values",val:" = 4"},{name:"lqe_hidden_dim",val:" = 64"},{name:"lqe_layers",val:" = 2"},{name:"decoder_offset_scale",val:" = 0.5"},{name:"decoder_method",val:" = 'default'"},{name:"up",val:" = 0.5"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DFineConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.01) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DFineConfig.initializer_bias_prior_prob",description:`<strong>initializer_bias_prior_prob</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The prior probability used by the bias initializer to initialize biases for <code>enc_score_head</code> and <code>class_embed</code>.
If <code>None</code>, <code>prior_prob</code> computed as <code>prior_prob = 1 / (num_labels + 1)</code> while initializing model weights.`,name:"initializer_bias_prior_prob"},{anchor:"transformers.DFineConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.DFineConfig.batch_norm_eps",description:`<strong>batch_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the batch normalization layers.`,name:"batch_norm_eps"},{anchor:"transformers.DFineConfig.backbone_config",description:`<strong>backbone_config</strong> (<code>Dict</code>, <em>optional</em>, defaults to <code>RTDetrResNetConfig()</code>) &#x2014;
The configuration of the backbone model.`,name:"backbone_config"},{anchor:"transformers.DFineConfig.backbone",description:`<strong>backbone</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Name of backbone to use when <code>backbone_config</code> is <code>None</code>. If <code>use_pretrained_backbone</code> is <code>True</code>, this
will load the corresponding pretrained weights from the timm or transformers library. If <code>use_pretrained_backbone</code>
is <code>False</code>, this loads the backbone&#x2019;s config and uses that to initialize the backbone with random weights.`,name:"backbone"},{anchor:"transformers.DFineConfig.use_pretrained_backbone",description:`<strong>use_pretrained_backbone</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use pretrained weights for the backbone.`,name:"use_pretrained_backbone"},{anchor:"transformers.DFineConfig.use_timm_backbone",description:`<strong>use_timm_backbone</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to load <code>backbone</code> from the timm library. If <code>False</code>, the backbone is loaded from the transformers
library.`,name:"use_timm_backbone"},{anchor:"transformers.DFineConfig.freeze_backbone_batch_norms",description:`<strong>freeze_backbone_batch_norms</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to freeze the batch normalization layers in the backbone.`,name:"freeze_backbone_batch_norms"},{anchor:"transformers.DFineConfig.backbone_kwargs",description:`<strong>backbone_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
e.g. <code>{&apos;out_indices&apos;: (0, 1, 2, 3)}</code>. Cannot be specified if <code>backbone_config</code> is set.`,name:"backbone_kwargs"},{anchor:"transformers.DFineConfig.encoder_hidden_dim",description:`<strong>encoder_hidden_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimension of the layers in hybrid encoder.`,name:"encoder_hidden_dim"},{anchor:"transformers.DFineConfig.encoder_in_channels",description:`<strong>encoder_in_channels</strong> (<code>list</code>, <em>optional</em>, defaults to <code>[512, 1024, 2048]</code>) &#x2014;
Multi level features input for encoder.`,name:"encoder_in_channels"},{anchor:"transformers.DFineConfig.feat_strides",description:`<strong>feat_strides</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[8, 16, 32]</code>) &#x2014;
Strides used in each feature map.`,name:"feat_strides"},{anchor:"transformers.DFineConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Total of layers to be used by the encoder.`,name:"encoder_layers"},{anchor:"transformers.DFineConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.DFineConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.DFineConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The ratio for all dropout layers.`,name:"dropout"},{anchor:"transformers.DFineConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.DFineConfig.encode_proj_layers",description:`<strong>encode_proj_layers</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[2]</code>) &#x2014;
Indexes of the projected layers to be used in the encoder.`,name:"encode_proj_layers"},{anchor:"transformers.DFineConfig.positional_encoding_temperature",description:`<strong>positional_encoding_temperature</strong> (<code>int</code>, <em>optional</em>, defaults to 10000) &#x2014;
The temperature parameter used to create the positional encodings.`,name:"positional_encoding_temperature"},{anchor:"transformers.DFineConfig.encoder_activation_function",description:`<strong>encoder_activation_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"encoder_activation_function"},{anchor:"transformers.DFineConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the general layer. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.DFineConfig.eval_size",description:`<strong>eval_size</strong> (<code>tuple[int, int]</code>, <em>optional</em>) &#x2014;
Height and width used to computes the effective height and width of the position embeddings after taking
into account the stride.`,name:"eval_size"},{anchor:"transformers.DFineConfig.normalize_before",description:`<strong>normalize_before</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Determine whether to apply layer normalization in the transformer encoder layer before self-attention and
feed-forward modules.`,name:"normalize_before"},{anchor:"transformers.DFineConfig.hidden_expansion",description:`<strong>hidden_expansion</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Expansion ratio to enlarge the dimension size of RepVGGBlock and CSPRepLayer.`,name:"hidden_expansion"},{anchor:"transformers.DFineConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimension of the layers exclude hybrid encoder.`,name:"d_model"},{anchor:"transformers.DFineConfig.num_queries",description:`<strong>num_queries</strong> (<code>int</code>, <em>optional</em>, defaults to 300) &#x2014;
Number of object queries.`,name:"num_queries"},{anchor:"transformers.DFineConfig.decoder_in_channels",description:`<strong>decoder_in_channels</strong> (<code>list</code>, <em>optional</em>, defaults to <code>[256, 256, 256]</code>) &#x2014;
Multi level features dimension for decoder`,name:"decoder_in_channels"},{anchor:"transformers.DFineConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.DFineConfig.num_feature_levels",description:`<strong>num_feature_levels</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The number of input feature levels.`,name:"num_feature_levels"},{anchor:"transformers.DFineConfig.decoder_n_points",description:`<strong>decoder_n_points</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The number of sampled keys in each feature level for each attention head in the decoder.`,name:"decoder_n_points"},{anchor:"transformers.DFineConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.DFineConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.DFineConfig.decoder_activation_function",description:`<strong>decoder_activation_function</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"decoder_activation_function"},{anchor:"transformers.DFineConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.DFineConfig.num_denoising",description:`<strong>num_denoising</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
The total number of denoising tasks or queries to be used for contrastive denoising.`,name:"num_denoising"},{anchor:"transformers.DFineConfig.label_noise_ratio",description:`<strong>label_noise_ratio</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The fraction of denoising labels to which random noise should be added.`,name:"label_noise_ratio"},{anchor:"transformers.DFineConfig.box_noise_scale",description:`<strong>box_noise_scale</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Scale or magnitude of noise to be added to the bounding boxes.`,name:"box_noise_scale"},{anchor:"transformers.DFineConfig.learn_initial_query",description:`<strong>learn_initial_query</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Indicates whether the initial query embeddings for the decoder should be learned during training`,name:"learn_initial_query"},{anchor:"transformers.DFineConfig.anchor_image_size",description:`<strong>anchor_image_size</strong> (<code>tuple[int, int]</code>, <em>optional</em>) &#x2014;
Height and width of the input image used during evaluation to generate the bounding box anchors. If None, automatic generate anchor is applied.`,name:"anchor_image_size"},{anchor:"transformers.DFineConfig.with_box_refine",description:`<strong>with_box_refine</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
based on the predictions from the previous layer.`,name:"with_box_refine"},{anchor:"transformers.DFineConfig.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the architecture has an encoder decoder structure.`,name:"is_encoder_decoder"},{anchor:"transformers.DFineConfig.matcher_alpha",description:`<strong>matcher_alpha</strong> (<code>float</code>, <em>optional</em>, defaults to 0.25) &#x2014;
Parameter alpha used by the Hungarian Matcher.`,name:"matcher_alpha"},{anchor:"transformers.DFineConfig.matcher_gamma",description:`<strong>matcher_gamma</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
Parameter gamma used by the Hungarian Matcher.`,name:"matcher_gamma"},{anchor:"transformers.DFineConfig.matcher_class_cost",description:`<strong>matcher_class_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
The relative weight of the class loss used by the Hungarian Matcher.`,name:"matcher_class_cost"},{anchor:"transformers.DFineConfig.matcher_bbox_cost",description:`<strong>matcher_bbox_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 5.0) &#x2014;
The relative weight of the bounding box loss used by the Hungarian Matcher.`,name:"matcher_bbox_cost"},{anchor:"transformers.DFineConfig.matcher_giou_cost",description:`<strong>matcher_giou_cost</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
The relative weight of the giou loss of used by the Hungarian Matcher.`,name:"matcher_giou_cost"},{anchor:"transformers.DFineConfig.use_focal_loss",description:`<strong>use_focal_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Parameter informing if focal focal should be used.`,name:"use_focal_loss"},{anchor:"transformers.DFineConfig.auxiliary_loss",description:`<strong>auxiliary_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether auxiliary decoding losses (loss at each decoder layer) are to be used.`,name:"auxiliary_loss"},{anchor:"transformers.DFineConfig.focal_loss_alpha",description:`<strong>focal_loss_alpha</strong> (<code>float</code>, <em>optional</em>, defaults to 0.75) &#x2014;
Parameter alpha used to compute the focal loss.`,name:"focal_loss_alpha"},{anchor:"transformers.DFineConfig.focal_loss_gamma",description:`<strong>focal_loss_gamma</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
Parameter gamma used to compute the focal loss.`,name:"focal_loss_gamma"},{anchor:"transformers.DFineConfig.weight_loss_vfl",description:`<strong>weight_loss_vfl</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Relative weight of the varifocal loss in the object detection loss.`,name:"weight_loss_vfl"},{anchor:"transformers.DFineConfig.weight_loss_bbox",description:`<strong>weight_loss_bbox</strong> (<code>float</code>, <em>optional</em>, defaults to 5.0) &#x2014;
Relative weight of the L1 bounding box loss in the object detection loss.`,name:"weight_loss_bbox"},{anchor:"transformers.DFineConfig.weight_loss_giou",description:`<strong>weight_loss_giou</strong> (<code>float</code>, <em>optional</em>, defaults to 2.0) &#x2014;
Relative weight of the generalized IoU loss in the object detection loss.`,name:"weight_loss_giou"},{anchor:"transformers.DFineConfig.weight_loss_fgl",description:`<strong>weight_loss_fgl</strong> (<code>float</code>, <em>optional</em>, defaults to 0.15) &#x2014;
Relative weight of the fine-grained localization loss in the object detection loss.`,name:"weight_loss_fgl"},{anchor:"transformers.DFineConfig.weight_loss_ddf",description:`<strong>weight_loss_ddf</strong> (<code>float</code>, <em>optional</em>, defaults to 1.5) &#x2014;
Relative weight of the decoupled distillation focal loss in the object detection loss.`,name:"weight_loss_ddf"},{anchor:"transformers.DFineConfig.eos_coefficient",description:`<strong>eos_coefficient</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0001) &#x2014;
Relative classification weight of the &#x2018;no-object&#x2019; class in the object detection loss.`,name:"eos_coefficient"},{anchor:"transformers.DFineConfig.eval_idx",description:`<strong>eval_idx</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
Index of the decoder layer to use for evaluation. If negative, counts from the end
(e.g., -1 means use the last layer). This allows for early prediction in the decoder
stack while still training later layers.`,name:"eval_idx"},{anchor:"transformers.DFineConfig.layer_scale",description:`<strong>layer_scale</strong> (<code>float</code>, <em>optional</em>, defaults to <code>1.0</code>) &#x2014;
Scaling factor for the hidden dimension in later decoder layers. Used to adjust the
model capacity after the evaluation layer.`,name:"layer_scale"},{anchor:"transformers.DFineConfig.max_num_bins",description:`<strong>max_num_bins</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Maximum number of bins for the distribution-guided bounding box refinement.
Higher values allow for more fine-grained localization but increase computation.`,name:"max_num_bins"},{anchor:"transformers.DFineConfig.reg_scale",description:`<strong>reg_scale</strong> (<code>float</code>, <em>optional</em>, defaults to 4.0) &#x2014;
Scale factor for the regression distribution. Controls the range and granularity
of the bounding box refinement process.`,name:"reg_scale"},{anchor:"transformers.DFineConfig.depth_mult",description:`<strong>depth_mult</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Multiplier for the number of blocks in RepNCSPELAN4 layers. Used to scale the model&#x2019;s
depth while maintaining its architecture.`,name:"depth_mult"},{anchor:"transformers.DFineConfig.top_prob_values",description:`<strong>top_prob_values</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of top probability values to consider from each corner&#x2019;s distribution.`,name:"top_prob_values"},{anchor:"transformers.DFineConfig.lqe_hidden_dim",description:`<strong>lqe_hidden_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Hidden dimension size for the Location Quality Estimator (LQE) network.`,name:"lqe_hidden_dim"},{anchor:"transformers.DFineConfig.lqe_layers",description:`<strong>lqe_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of layers in the Location Quality Estimator MLP.`,name:"lqe_layers"},{anchor:"transformers.DFineConfig.decoder_offset_scale",description:`<strong>decoder_offset_scale</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Offset scale used in deformable attention.`,name:"decoder_offset_scale"},{anchor:"transformers.DFineConfig.decoder_method",description:`<strong>decoder_method</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;default&quot;</code>) &#x2014;
The method to use for the decoder: <code>&quot;default&quot;</code> or <code>&quot;discrete&quot;</code>.`,name:"decoder_method"},{anchor:"transformers.DFineConfig.up",description:`<strong>up</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Controls the upper bounds of the Weighting Function.`,name:"up"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/d_fine/configuration_d_fine.py#L32"}}),Y=new ge({props:{name:"from_backbone_configs",anchor:"transformers.DFineConfig.from_backbone_configs",parameters:[{name:"backbone_config",val:": PretrainedConfig"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DFineConfig.from_backbone_configs.backbone_config",description:`<strong>backbone_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>) &#x2014;
The backbone configuration.`,name:"backbone_config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/d_fine/configuration_d_fine.py#L415",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>An instance of a configuration object</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig"
>DFineConfig</a></p>
`}}),A=new _e({props:{title:"DFineModel",local:"transformers.DFineModel",headingTag:"h2"}}),K=new ge({props:{name:"class transformers.DFineModel",anchor:"transformers.DFineModel",parameters:[{name:"config",val:": DFineConfig"}],parametersDescription:[{anchor:"transformers.DFineModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig">DFineConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/d_fine/modeling_d_fine.py#L1110"}}),ee=new ge({props:{name:"forward",anchor:"transformers.DFineModel.forward",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[list[dict]] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DFineModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<code>image_processor_class</code>. See <code>image_processor_class.__call__</code> for details (<code>processor_class</code> uses
<code>image_processor_class</code> for processing images).`,name:"pixel_values"},{anchor:"transformers.DFineModel.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.DFineModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.DFineModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
can choose to directly pass a flattened representation of an image.`,name:"inputs_embeds"},{anchor:"transformers.DFineModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_queries, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
embedded representation.`,name:"decoder_inputs_embeds"},{anchor:"transformers.DFineModel.forward.labels",description:`<strong>labels</strong> (<code>list[Dict]</code> of len <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
following 2 keys: &#x2018;class_labels&#x2019; and &#x2018;boxes&#x2019; (the class labels and bounding boxes of an image in the batch
respectively). The class labels themselves should be a <code>torch.LongTensor</code> of len <code>(number of bounding boxes in the image,)</code> and the boxes a <code>torch.FloatTensor</code> of shape <code>(number of bounding boxes in the image, 4)</code>.`,name:"labels"},{anchor:"transformers.DFineModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DFineModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DFineModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/d_fine/modeling_d_fine.py#L1230",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.d_fine.modeling_d_fine.DFineModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig"
>DFineConfig</a>) and inputs.</p>
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


<p><code>transformers.models.d_fine.modeling_d_fine.DFineModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),q=new fo({props:{$$slots:{default:[To]},$$scope:{ctx:k}}}),W=new go({props:{anchor:"transformers.DFineModel.forward.example",$$slots:{default:[jo]},$$scope:{ctx:k}}}),oe=new _e({props:{title:"DFineForObjectDetection",local:"transformers.DFineForObjectDetection",headingTag:"h2"}}),ne=new ge({props:{name:"class transformers.DFineForObjectDetection",anchor:"transformers.DFineForObjectDetection",parameters:[{name:"config",val:": DFineConfig"}],parametersDescription:[{anchor:"transformers.DFineForObjectDetection.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig">DFineConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/d_fine/modeling_d_fine.py#L1537"}}),te=new ge({props:{name:"forward",anchor:"transformers.DFineForObjectDetection.forward",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"pixel_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[list[dict]] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DFineForObjectDetection.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<code>image_processor_class</code>. See <code>image_processor_class.__call__</code> for details (<code>processor_class</code> uses
<code>image_processor_class</code> for processing images).`,name:"pixel_values"},{anchor:"transformers.DFineForObjectDetection.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.DFineForObjectDetection.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.DFineForObjectDetection.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DFineForObjectDetection.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.DFineForObjectDetection.forward.labels",description:`<strong>labels</strong> (<code>list[dict]</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DFineForObjectDetection.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DFineForObjectDetection.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DFineForObjectDetection.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/d_fine/modeling_d_fine.py#L1577",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.d_fine.modeling_d_fine.DFineObjectDetectionOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/d_fine#transformers.DFineConfig"
>DFineConfig</a>) and inputs.</p>
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
possible padding). You can use <code>~DFineImageProcessor.post_process_object_detection</code> to retrieve the
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


<p><code>transformers.models.d_fine.modeling_d_fine.DFineObjectDetectionOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Z=new fo({props:{$$slots:{default:[xo]},$$scope:{ctx:k}}}),I=new go({props:{anchor:"transformers.DFineForObjectDetection.forward.example",$$slots:{default:[Fo]},$$scope:{ctx:k}}}),se=new vo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/d_fine.md"}}),{c(){t=c("meta"),M=s(),p=c("p"),m=s(),w=c("p"),w.innerHTML=i,x=s(),h(V.$$.fragment),ye=s(),h(B.$$.fragment),we=s(),H=c("p"),H.innerHTML=eo,Me=s(),G=c("p"),G.textContent=oo,ve=s(),S=c("p"),S.innerHTML=no,Te=s(),Q=c("p"),Q.innerHTML=to,je=s(),h(X.$$.fragment),xe=s(),h(L.$$.fragment),Fe=s(),h(O.$$.fragment),Ce=s(),U=c("div"),h(P.$$.fragment),Ze=s(),re=c("p"),re.innerHTML=so,Ie=s(),$=c("div"),h(Y.$$.fragment),Re=s(),ie=c("p"),ie.innerHTML=ao,Ue=s(),h(A.$$.fragment),De=s(),v=c("div"),h(K.$$.fragment),Ne=s(),le=c("p"),le.textContent=ro,Ee=s(),ce=c("p"),ce.innerHTML=io,Ve=s(),de=c("p"),de.innerHTML=lo,Be=s(),F=c("div"),h(ee.$$.fragment),He=s(),pe=c("p"),pe.innerHTML=co,Ge=s(),h(q.$$.fragment),Se=s(),h(W.$$.fragment),ke=s(),h(oe.$$.fragment),ze=s(),T=c("div"),h(ne.$$.fragment),Qe=s(),me=c("p"),me.textContent=po,Xe=s(),he=c("p"),he.innerHTML=mo,Le=s(),ue=c("p"),ue.innerHTML=ho,Oe=s(),C=c("div"),h(te.$$.fragment),Pe=s(),fe=c("p"),fe.innerHTML=uo,Ye=s(),h(Z.$$.fragment),Ae=s(),h(I.$$.fragment),Je=s(),h(se.$$.fragment),$e=s(),be=c("p"),this.h()},l(e){const o=Mo("svelte-u9bgzb",document.head);t=d(o,"META",{name:!0,content:!0}),o.forEach(n),M=a(e),p=d(e,"P",{}),N(p).forEach(n),m=a(e),w=d(e,"P",{"data-svelte-h":!0}),y(w)!=="svelte-yfqpjn"&&(w.innerHTML=i),x=a(e),u(V.$$.fragment,e),ye=a(e),u(B.$$.fragment,e),we=a(e),H=d(e,"P",{"data-svelte-h":!0}),y(H)!=="svelte-2wmkbi"&&(H.innerHTML=eo),Me=a(e),G=d(e,"P",{"data-svelte-h":!0}),y(G)!=="svelte-vfdo9a"&&(G.textContent=oo),ve=a(e),S=d(e,"P",{"data-svelte-h":!0}),y(S)!=="svelte-jxtc3a"&&(S.innerHTML=no),Te=a(e),Q=d(e,"P",{"data-svelte-h":!0}),y(Q)!=="svelte-10gtpja"&&(Q.innerHTML=to),je=a(e),u(X.$$.fragment,e),xe=a(e),u(L.$$.fragment,e),Fe=a(e),u(O.$$.fragment,e),Ce=a(e),U=d(e,"DIV",{class:!0});var J=N(U);u(P.$$.fragment,J),Ze=a(J),re=d(J,"P",{"data-svelte-h":!0}),y(re)!=="svelte-1ug7cwm"&&(re.innerHTML=so),Ie=a(J),$=d(J,"DIV",{class:!0});var ae=N($);u(Y.$$.fragment,ae),Re=a(ae),ie=d(ae,"P",{"data-svelte-h":!0}),y(ie)!=="svelte-aikxzg"&&(ie.innerHTML=ao),ae.forEach(n),J.forEach(n),Ue=a(e),u(A.$$.fragment,e),De=a(e),v=d(e,"DIV",{class:!0});var j=N(v);u(K.$$.fragment,j),Ne=a(j),le=d(j,"P",{"data-svelte-h":!0}),y(le)!=="svelte-121t6ur"&&(le.textContent=ro),Ee=a(j),ce=d(j,"P",{"data-svelte-h":!0}),y(ce)!=="svelte-q52n56"&&(ce.innerHTML=io),Ve=a(j),de=d(j,"P",{"data-svelte-h":!0}),y(de)!=="svelte-hswkmf"&&(de.innerHTML=lo),Be=a(j),F=d(j,"DIV",{class:!0});var D=N(F);u(ee.$$.fragment,D),He=a(D),pe=d(D,"P",{"data-svelte-h":!0}),y(pe)!=="svelte-v7rmqk"&&(pe.innerHTML=co),Ge=a(D),u(q.$$.fragment,D),Se=a(D),u(W.$$.fragment,D),D.forEach(n),j.forEach(n),ke=a(e),u(oe.$$.fragment,e),ze=a(e),T=d(e,"DIV",{class:!0});var z=N(T);u(ne.$$.fragment,z),Qe=a(z),me=d(z,"P",{"data-svelte-h":!0}),y(me)!=="svelte-xszdn5"&&(me.textContent=po),Xe=a(z),he=d(z,"P",{"data-svelte-h":!0}),y(he)!=="svelte-q52n56"&&(he.innerHTML=mo),Le=a(z),ue=d(z,"P",{"data-svelte-h":!0}),y(ue)!=="svelte-hswkmf"&&(ue.innerHTML=ho),Oe=a(z),C=d(z,"DIV",{class:!0});var R=N(C);u(te.$$.fragment,R),Pe=a(R),fe=d(R,"P",{"data-svelte-h":!0}),y(fe)!=="svelte-1qp62zq"&&(fe.innerHTML=uo),Ye=a(R),u(Z.$$.fragment,R),Ae=a(R),u(I.$$.fragment,R),R.forEach(n),z.forEach(n),Je=a(e),u(se.$$.fragment,e),$e=a(e),be=d(e,"P",{}),N(be).forEach(n),this.h()},h(){E(t,"name","hf:doc:metadata"),E(t,"content",Uo),E($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),E(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),E(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),E(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),E(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),E(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){l(document.head,t),r(e,M,o),r(e,p,o),r(e,m,o),r(e,w,o),r(e,x,o),f(V,e,o),r(e,ye,o),f(B,e,o),r(e,we,o),r(e,H,o),r(e,Me,o),r(e,G,o),r(e,ve,o),r(e,S,o),r(e,Te,o),r(e,Q,o),r(e,je,o),f(X,e,o),r(e,xe,o),f(L,e,o),r(e,Fe,o),f(O,e,o),r(e,Ce,o),r(e,U,o),f(P,U,null),l(U,Ze),l(U,re),l(U,Ie),l(U,$),f(Y,$,null),l($,Re),l($,ie),r(e,Ue,o),f(A,e,o),r(e,De,o),r(e,v,o),f(K,v,null),l(v,Ne),l(v,le),l(v,Ee),l(v,ce),l(v,Ve),l(v,de),l(v,Be),l(v,F),f(ee,F,null),l(F,He),l(F,pe),l(F,Ge),f(q,F,null),l(F,Se),f(W,F,null),r(e,ke,o),f(oe,e,o),r(e,ze,o),r(e,T,o),f(ne,T,null),l(T,Qe),l(T,me),l(T,Xe),l(T,he),l(T,Le),l(T,ue),l(T,Oe),l(T,C),f(te,C,null),l(C,Pe),l(C,fe),l(C,Ye),f(Z,C,null),l(C,Ae),f(I,C,null),r(e,Je,o),f(se,e,o),r(e,$e,o),r(e,be,o),qe=!0},p(e,[o]){const J={};o&2&&(J.$$scope={dirty:o,ctx:e}),q.$set(J);const ae={};o&2&&(ae.$$scope={dirty:o,ctx:e}),W.$set(ae);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),Z.$set(j);const D={};o&2&&(D.$$scope={dirty:o,ctx:e}),I.$set(D)},i(e){qe||(g(V.$$.fragment,e),g(B.$$.fragment,e),g(X.$$.fragment,e),g(L.$$.fragment,e),g(O.$$.fragment,e),g(P.$$.fragment,e),g(Y.$$.fragment,e),g(A.$$.fragment,e),g(K.$$.fragment,e),g(ee.$$.fragment,e),g(q.$$.fragment,e),g(W.$$.fragment,e),g(oe.$$.fragment,e),g(ne.$$.fragment,e),g(te.$$.fragment,e),g(Z.$$.fragment,e),g(I.$$.fragment,e),g(se.$$.fragment,e),qe=!0)},o(e){_(V.$$.fragment,e),_(B.$$.fragment,e),_(X.$$.fragment,e),_(L.$$.fragment,e),_(O.$$.fragment,e),_(P.$$.fragment,e),_(Y.$$.fragment,e),_(A.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(q.$$.fragment,e),_(W.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(te.$$.fragment,e),_(Z.$$.fragment,e),_(I.$$.fragment,e),_(se.$$.fragment,e),qe=!1},d(e){e&&(n(M),n(p),n(m),n(w),n(x),n(ye),n(we),n(H),n(Me),n(G),n(ve),n(S),n(Te),n(Q),n(je),n(xe),n(Fe),n(Ce),n(U),n(Ue),n(De),n(v),n(ke),n(ze),n(T),n(Je),n($e),n(be)),n(t),b(V,e),b(B,e),b(X,e),b(L,e),b(O,e),b(P),b(Y),b(A,e),b(K),b(ee),b(q),b(W),b(oe,e),b(ne),b(te),b(Z),b(I),b(se,e)}}}const Uo='{"title":"D-FINE","local":"d-fine","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"DFineConfig","local":"transformers.DFineConfig","sections":[],"depth":2},{"title":"DFineModel","local":"transformers.DFineModel","sections":[],"depth":2},{"title":"DFineForObjectDetection","local":"transformers.DFineForObjectDetection","sections":[],"depth":2}],"depth":1}';function Do(k){return bo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Io extends yo{constructor(t){super(),wo(this,t,Do,Co,_o,{})}}export{Io as component};
