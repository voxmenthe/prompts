import{s as Mn,z as yn,o as Fn,n as it}from"../chunks/scheduler.18a86fab.js";import{S as vn,i as wn,g as a,s,r as m,A as Tn,h as i,f as o,c as n,j as F,x as _,u as p,k as M,y as t,a as c,v as g,d as h,t as f,w as u}from"../chunks/index.98837b22.js";import{T as Ps}from"../chunks/Tip.77304350.js";import{D as w}from"../chunks/Docstring.a1ef7999.js";import{C as Uo}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Po}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{P as xn}from"../chunks/PipelineTag.7749150e.js";import{H as B,E as In}from"../chunks/getInferenceSnippets.06c2775f.js";function $n(N){let d,v=`This is a recently introduced model so the API hasn’t been tested extensively. There may be some bugs or slight
breaking changes to fix it in the future. If you see something strange, file a <a href="https://github.com/huggingface/transformers/issues/new?assignees=&amp;labels=&amp;template=bug-report.md&amp;title" rel="nofollow">Github Issue</a>.`;return{c(){d=a("p"),d.innerHTML=v},l(b){d=i(b,"P",{"data-svelte-h":!0}),_(d)!=="svelte-j665pk"&&(d.innerHTML=v)},m(b,k){c(b,d,k)},p:it,d(b){b&&o(d)}}}function zn(N){let d,v="Examples:",b,k,y;return k=new Uo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hc2tGb3JtZXJDb25maWclMkMlMjBNYXNrRm9ybWVyTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTWFza0Zvcm1lciUyMGZhY2Vib29rJTJGbWFza2Zvcm1lci1zd2luLWJhc2UtYWRlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBNYXNrRm9ybWVyQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGZhY2Vib29rJTJGbWFza2Zvcm1lci1zd2luLWJhc2UtYWRlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBNYXNrRm9ybWVyTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MaskFormerConfig, MaskFormerModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a MaskFormer facebook/maskformer-swin-base-ade configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MaskFormerConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the facebook/maskformer-swin-base-ade style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MaskFormerModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){d=a("p"),d.textContent=v,b=s(),m(k.$$.fragment)},l(l){d=i(l,"P",{"data-svelte-h":!0}),_(d)!=="svelte-kvfsh7"&&(d.textContent=v),b=n(l),p(k.$$.fragment,l)},m(l,x){c(l,d,x),c(l,b,x),g(k,l,x),y=!0},p:it,i(l){y||(h(k.$$.fragment,l),y=!0)},o(l){f(k.$$.fragment,l),y=!1},d(l){l&&(o(d),o(b)),u(k,l)}}}function jn(N){let d,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){d=a("p"),d.innerHTML=v},l(b){d=i(b,"P",{"data-svelte-h":!0}),_(d)!=="svelte-fincs2"&&(d.innerHTML=v)},m(b,k){c(b,d,k)},p:it,d(b){b&&o(d)}}}function Nn(N){let d,v="Examples:",b,k,y;return k=new Uo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyME1hc2tGb3JtZXJNb2RlbCUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHJlcXVlc3RzJTBBJTBBJTIzJTIwbG9hZCUyME1hc2tGb3JtZXIlMjBmaW5lLXR1bmVkJTIwb24lMjBBREUyMGslMjBzZW1hbnRpYyUyMHNlZ21lbnRhdGlvbiUwQWltYWdlX3Byb2Nlc3NvciUyMCUzRCUyMEF1dG9JbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYXNrZm9ybWVyLXN3aW4tYmFzZS1hZGUlMjIpJTBBbW9kZWwlMjAlM0QlMjBNYXNrRm9ybWVyTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWFza2Zvcm1lci1zd2luLWJhc2UtYWRlJTIyKSUwQSUwQXVybCUyMCUzRCUyMCUyMmh0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEElMEFpbnB1dHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IoaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMGZvcndhcmQlMjBwYXNzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQSUyMyUyMHRoZSUyMGRlY29kZXIlMjBvZiUyME1hc2tGb3JtZXIlMjBvdXRwdXRzJTIwaGlkZGVuJTIwc3RhdGVzJTIwb2YlMjBzaGFwZSUyMChiYXRjaF9zaXplJTJDJTIwbnVtX3F1ZXJpZXMlMkMlMjBoaWRkZW5fc2l6ZSklMEF0cmFuc2Zvcm1lcl9kZWNvZGVyX2xhc3RfaGlkZGVuX3N0YXRlJTIwJTNEJTIwb3V0cHV0cy50cmFuc2Zvcm1lcl9kZWNvZGVyX2xhc3RfaGlkZGVuX3N0YXRlJTBBbGlzdCh0cmFuc2Zvcm1lcl9kZWNvZGVyX2xhc3RfaGlkZGVuX3N0YXRlLnNoYXBlKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, MaskFormerModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># load MaskFormer fine-tuned on ADE20k semantic segmentation</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/maskformer-swin-base-ade&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MaskFormerModel.from_pretrained(<span class="hljs-string">&quot;facebook/maskformer-swin-base-ade&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the decoder of MaskFormer outputs hidden states of shape (batch_size, num_queries, hidden_size)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>transformer_decoder_last_hidden_state = outputs.transformer_decoder_last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(transformer_decoder_last_hidden_state.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">100</span>, <span class="hljs-number">256</span>]`,wrap:!1}}),{c(){d=a("p"),d.textContent=v,b=s(),m(k.$$.fragment)},l(l){d=i(l,"P",{"data-svelte-h":!0}),_(d)!=="svelte-kvfsh7"&&(d.textContent=v),b=n(l),p(k.$$.fragment,l)},m(l,x){c(l,d,x),c(l,b,x),g(k,l,x),y=!0},p:it,i(l){y||(h(k.$$.fragment,l),y=!0)},o(l){f(k.$$.fragment,l),y=!1},d(l){l&&(o(d),o(b)),u(k,l)}}}function Pn(N){let d,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){d=a("p"),d.innerHTML=v},l(b){d=i(b,"P",{"data-svelte-h":!0}),_(d)!=="svelte-fincs2"&&(d.innerHTML=v)},m(b,k){c(b,d,k)},p:it,d(b){b&&o(d)}}}function Un(N){let d,v="Semantic segmentation example:",b,k,y;return k=new Uo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyME1hc2tGb3JtZXJGb3JJbnN0YW5jZVNlZ21lbnRhdGlvbiUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHJlcXVlc3RzJTBBJTBBJTIzJTIwbG9hZCUyME1hc2tGb3JtZXIlMjBmaW5lLXR1bmVkJTIwb24lMjBBREUyMGslMjBzZW1hbnRpYyUyMHNlZ21lbnRhdGlvbiUwQWltYWdlX3Byb2Nlc3NvciUyMCUzRCUyMEF1dG9JbWFnZVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYXNrZm9ybWVyLXN3aW4tYmFzZS1hZGUlMjIpJTBBbW9kZWwlMjAlM0QlMjBNYXNrRm9ybWVyRm9ySW5zdGFuY2VTZWdtZW50YXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWFza2Zvcm1lci1zd2luLWJhc2UtYWRlJTIyKSUwQSUwQXVybCUyMCUzRCUyMCglMEElMjAlMjAlMjAlMjAlMjJodHRwcyUzQSUyRiUyRmh1Z2dpbmdmYWNlLmNvJTJGZGF0YXNldHMlMkZoZi1pbnRlcm5hbC10ZXN0aW5nJTJGZml4dHVyZXNfYWRlMjBrJTJGcmVzb2x2ZSUyRm1haW4lMkZBREVfdmFsXzAwMDAwMDAxLmpwZyUyMiUwQSklMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEFpbnB1dHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IoaW1hZ2VzJTNEaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMjMlMjBtb2RlbCUyMHByZWRpY3RzJTIwY2xhc3NfcXVlcmllc19sb2dpdHMlMjBvZiUyMHNoYXBlJTIwJTYwKGJhdGNoX3NpemUlMkMlMjBudW1fcXVlcmllcyklNjAlMEElMjMlMjBhbmQlMjBtYXNrc19xdWVyaWVzX2xvZ2l0cyUyMG9mJTIwc2hhcGUlMjAlNjAoYmF0Y2hfc2l6ZSUyQyUyMG51bV9xdWVyaWVzJTJDJTIwaGVpZ2h0JTJDJTIwd2lkdGgpJTYwJTBBY2xhc3NfcXVlcmllc19sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmNsYXNzX3F1ZXJpZXNfbG9naXRzJTBBbWFza3NfcXVlcmllc19sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLm1hc2tzX3F1ZXJpZXNfbG9naXRzJTBBJTBBJTIzJTIweW91JTIwY2FuJTIwcGFzcyUyMHRoZW0lMjB0byUyMGltYWdlX3Byb2Nlc3NvciUyMGZvciUyMHBvc3Rwcm9jZXNzaW5nJTBBcHJlZGljdGVkX3NlbWFudGljX21hcCUyMCUzRCUyMGltYWdlX3Byb2Nlc3Nvci5wb3N0X3Byb2Nlc3Nfc2VtYW50aWNfc2VnbWVudGF0aW9uKCUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMkMlMjB0YXJnZXRfc2l6ZXMlM0QlNUIoaW1hZ2UuaGVpZ2h0JTJDJTIwaW1hZ2Uud2lkdGgpJTVEJTBBKSU1QjAlNUQlMEElMEElMjMlMjB3ZSUyMHJlZmVyJTIwdG8lMjB0aGUlMjBkZW1vJTIwbm90ZWJvb2tzJTIwZm9yJTIwdmlzdWFsaXphdGlvbiUyMChzZWUlMjAlMjJSZXNvdXJjZXMlMjIlMjBzZWN0aW9uJTIwaW4lMjB0aGUlMjBNYXNrRm9ybWVyJTIwZG9jcyklMEFsaXN0KHByZWRpY3RlZF9zZW1hbnRpY19tYXAuc2hhcGUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, MaskFormerForInstanceSegmentation
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># load MaskFormer fine-tuned on ADE20k semantic segmentation</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/maskformer-swin-base-ade&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MaskFormerForInstanceSegmentation.from_pretrained(<span class="hljs-string">&quot;facebook/maskformer-swin-base-ade&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = (
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># model predicts class_queries_logits of shape \`(batch_size, num_queries)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># and masks_queries_logits of shape \`(batch_size, num_queries, height, width)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>class_queries_logits = outputs.class_queries_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>masks_queries_logits = outputs.masks_queries_logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># you can pass them to image_processor for postprocessing</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_semantic_map = image_processor.post_process_semantic_segmentation(
<span class="hljs-meta">... </span>    outputs, target_sizes=[(image.height, image.width)]
<span class="hljs-meta">... </span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># we refer to the demo notebooks for visualization (see &quot;Resources&quot; section in the MaskFormer docs)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(predicted_semantic_map.shape)
[<span class="hljs-number">512</span>, <span class="hljs-number">683</span>]`,wrap:!1}}),{c(){d=a("p"),d.textContent=v,b=s(),m(k.$$.fragment)},l(l){d=i(l,"P",{"data-svelte-h":!0}),_(d)!=="svelte-ec2k7m"&&(d.textContent=v),b=n(l),p(k.$$.fragment,l)},m(l,x){c(l,d,x),c(l,b,x),g(k,l,x),y=!0},p:it,i(l){y||(h(k.$$.fragment,l),y=!0)},o(l){f(k.$$.fragment,l),y=!1},d(l){l&&(o(d),o(b)),u(k,l)}}}function Cn(N){let d,v="Panoptic segmentation example:",b,k,y;return k=new Uo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9JbWFnZVByb2Nlc3NvciUyQyUyME1hc2tGb3JtZXJGb3JJbnN0YW5jZVNlZ21lbnRhdGlvbiUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMHJlcXVlc3RzJTBBJTBBJTIzJTIwbG9hZCUyME1hc2tGb3JtZXIlMjBmaW5lLXR1bmVkJTIwb24lMjBDT0NPJTIwcGFub3B0aWMlMjBzZWdtZW50YXRpb24lMEFpbWFnZV9wcm9jZXNzb3IlMjAlM0QlMjBBdXRvSW1hZ2VQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbWFza2Zvcm1lci1zd2luLWJhc2UtY29jbyUyMiklMEFtb2RlbCUyMCUzRCUyME1hc2tGb3JtZXJGb3JJbnN0YW5jZVNlZ21lbnRhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtYXNrZm9ybWVyLXN3aW4tYmFzZS1jb2NvJTIyKSUwQSUwQXVybCUyMCUzRCUyMCUyMmh0dHAlM0ElMkYlMkZpbWFnZXMuY29jb2RhdGFzZXQub3JnJTJGdmFsMjAxNyUyRjAwMDAwMDAzOTc2OS5qcGclMjIlMEFpbWFnZSUyMCUzRCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybCUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMEFpbnB1dHMlMjAlM0QlMjBpbWFnZV9wcm9jZXNzb3IoaW1hZ2VzJTNEaW1hZ2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMjMlMjBtb2RlbCUyMHByZWRpY3RzJTIwY2xhc3NfcXVlcmllc19sb2dpdHMlMjBvZiUyMHNoYXBlJTIwJTYwKGJhdGNoX3NpemUlMkMlMjBudW1fcXVlcmllcyklNjAlMEElMjMlMjBhbmQlMjBtYXNrc19xdWVyaWVzX2xvZ2l0cyUyMG9mJTIwc2hhcGUlMjAlNjAoYmF0Y2hfc2l6ZSUyQyUyMG51bV9xdWVyaWVzJTJDJTIwaGVpZ2h0JTJDJTIwd2lkdGgpJTYwJTBBY2xhc3NfcXVlcmllc19sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmNsYXNzX3F1ZXJpZXNfbG9naXRzJTBBbWFza3NfcXVlcmllc19sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLm1hc2tzX3F1ZXJpZXNfbG9naXRzJTBBJTBBJTIzJTIweW91JTIwY2FuJTIwcGFzcyUyMHRoZW0lMjB0byUyMGltYWdlX3Byb2Nlc3NvciUyMGZvciUyMHBvc3Rwcm9jZXNzaW5nJTBBcmVzdWx0JTIwJTNEJTIwaW1hZ2VfcHJvY2Vzc29yLnBvc3RfcHJvY2Vzc19wYW5vcHRpY19zZWdtZW50YXRpb24ob3V0cHV0cyUyQyUyMHRhcmdldF9zaXplcyUzRCU1QihpbWFnZS5oZWlnaHQlMkMlMjBpbWFnZS53aWR0aCklNUQpJTVCMCU1RCUwQSUwQSUyMyUyMHdlJTIwcmVmZXIlMjB0byUyMHRoZSUyMGRlbW8lMjBub3RlYm9va3MlMjBmb3IlMjB2aXN1YWxpemF0aW9uJTIwKHNlZSUyMCUyMlJlc291cmNlcyUyMiUyMHNlY3Rpb24lMjBpbiUyMHRoZSUyME1hc2tGb3JtZXIlMjBkb2NzKSUwQXByZWRpY3RlZF9wYW5vcHRpY19tYXAlMjAlM0QlMjByZXN1bHQlNUIlMjJzZWdtZW50YXRpb24lMjIlNUQlMEFsaXN0KHByZWRpY3RlZF9wYW5vcHRpY19tYXAuc2hhcGUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoImageProcessor, MaskFormerForInstanceSegmentation
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># load MaskFormer fine-tuned on COCO panoptic segmentation</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image_processor = AutoImageProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/maskformer-swin-base-coco&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MaskFormerForInstanceSegmentation.from_pretrained(<span class="hljs-string">&quot;facebook/maskformer-swin-base-coco&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>url = <span class="hljs-string">&quot;http://images.cocodataset.org/val2017/000000039769.jpg&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image = Image.<span class="hljs-built_in">open</span>(requests.get(url, stream=<span class="hljs-literal">True</span>).raw)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = image_processor(images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># model predicts class_queries_logits of shape \`(batch_size, num_queries)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># and masks_queries_logits of shape \`(batch_size, num_queries, height, width)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>class_queries_logits = outputs.class_queries_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>masks_queries_logits = outputs.masks_queries_logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># you can pass them to image_processor for postprocessing</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(image.height, image.width)])[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># we refer to the demo notebooks for visualization (see &quot;Resources&quot; section in the MaskFormer docs)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_panoptic_map = result[<span class="hljs-string">&quot;segmentation&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(predicted_panoptic_map.shape)
[<span class="hljs-number">480</span>, <span class="hljs-number">640</span>]`,wrap:!1}}),{c(){d=a("p"),d.textContent=v,b=s(),m(k.$$.fragment)},l(l){d=i(l,"P",{"data-svelte-h":!0}),_(d)!=="svelte-1hqqxa2"&&(d.textContent=v),b=n(l),p(k.$$.fragment,l)},m(l,x){c(l,d,x),c(l,b,x),g(k,l,x),y=!0},p:it,i(l){y||(h(k.$$.fragment,l),y=!0)},o(l){f(k.$$.fragment,l),y=!1},d(l){l&&(o(d),o(b)),u(k,l)}}}function Zn(N){let d,v,b,k,y,l="<em>This model was released on 2021-07-13 and added to Hugging Face Transformers on 2022-03-02.</em>",x,ge,Ht,G,Us='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Ot,D,Xt,he,Vt,fe,Cs='The MaskFormer model was proposed in <a href="https://huggingface.co/papers/2107.06278" rel="nofollow">Per-Pixel Classification is Not All You Need for Semantic Segmentation</a> by Bowen Cheng, Alexander G. Schwing, Alexander Kirillov. MaskFormer addresses semantic segmentation with a mask classification paradigm instead of performing classic pixel-level classification.',qt,ue,Zs="The abstract from the paper is the following:",Gt,_e,Ws="<em>Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a single global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models.</em>",Dt,be,Js='The figure below illustrates the architecture of MaskFormer. Taken from the <a href="https://huggingface.co/papers/2107.06278" rel="nofollow">original paper</a>.',At,ke,Rs,Yt,Me,Ls='This model was contributed by <a href="https://huggingface.co/francesco" rel="nofollow">francesco</a>. The original code can be found <a href="https://github.com/facebookresearch/MaskFormer" rel="nofollow">here</a>.',Qt,ye,Kt,Fe,Bs=`<li>MaskFormer’s Transformer decoder is identical to the decoder of <a href="detr">DETR</a>. During training, the authors of DETR did find it helpful to use auxiliary losses in the decoder, especially to help the model output the correct number of objects of each class. If you set the parameter <code>use_auxiliary_loss</code> of <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig">MaskFormerConfig</a> to <code>True</code>, then prediction feedforward neural networks and Hungarian losses are added after each decoder layer (with the FFNs sharing parameters).</li> <li>If you want to train the model in a distributed environment across multiple nodes, then one should update the
<code>get_num_masks</code> function inside in the <code>MaskFormerLoss</code> class of <code>modeling_maskformer.py</code>. When training on multiple nodes, this should be
set to the average number of target masks across all nodes, as can be seen in the original implementation <a href="https://github.com/facebookresearch/MaskFormer/blob/da3e60d85fdeedcb31476b5edd7d328826ce56cc/mask_former/modeling/criterion.py#L169" rel="nofollow">here</a>.</li> <li>One can use <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor">MaskFormerImageProcessor</a> to prepare images for the model and optional targets for the model.</li> <li>To get the final segmentation, depending on the task, you can call <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_semantic_segmentation">post_process_semantic_segmentation()</a> or <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation">post_process_panoptic_segmentation()</a>. Both tasks can be solved using <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a> output, panoptic segmentation accepts an optional <code>label_ids_to_fuse</code> argument to fuse instances of the target object/s (e.g. sky) together.</li>`,eo,ve,to,we,oo,Te,Ss='<li>All notebooks that illustrate inference as well as fine-tuning on custom data with MaskFormer can be found <a href="https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MaskFormer" rel="nofollow">here</a>.</li> <li>Scripts for finetuning <code>MaskFormer</code> with <a href="/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer">Trainer</a> or <a href="https://huggingface.co/docs/accelerate/index" rel="nofollow">Accelerate</a> can be found <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation" rel="nofollow">here</a>.</li>',so,xe,no,O,Ie,Co,dt,Es='Class for outputs of <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerModel">MaskFormerModel</a>. This class returns all the needed hidden states to compute the logits.',ro,L,$e,Zo,ct,Hs='Class for outputs of <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>.',Wo,lt,Os=`This output can be directly passed to <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_semantic_segmentation">post_process_semantic_segmentation()</a> or or
<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_instance_segmentation">post_process_instance_segmentation()</a> or
<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation">post_process_panoptic_segmentation()</a> depending on the task. Please, see
[\`~MaskFormerImageProcessor] for details regarding usage.`,ao,ze,io,$,je,Jo,mt,Xs=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerModel">MaskFormerModel</a>. It is used to instantiate a
MaskFormer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MaskFormer
<a href="https://huggingface.co/facebook/maskformer-swin-base-ade" rel="nofollow">facebook/maskformer-swin-base-ade</a> architecture trained
on <a href="https://huggingface.co/datasets/scene_parse_150" rel="nofollow">ADE20k-150</a>.`,Ro,pt,Vs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Lo,gt,qs='Currently, MaskFormer only supports the <a href="swin">Swin Transformer</a> as backbone.',Bo,A,So,Y,Ne,Eo,ht,Gs=`Instantiate a <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig">MaskFormerConfig</a> (or a derived class) from a pre-trained backbone model configuration and DETR model
configuration.`,co,Pe,lo,T,Ue,Ho,ft,Ds=`Constructs a MaskFormer image processor. The image processor can be used to prepare image(s) and optional targets
for the model.`,Oo,ut,As=`This image processor inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BaseImageProcessor">BaseImageProcessor</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Xo,_t,Ce,Vo,S,Ze,qo,bt,Ys="Pad images up to the largest image in a batch and create a corresponding <code>pixel_mask</code>.",Go,kt,Qs=`MaskFormer addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps
will be converted to lists of binary masks and their respective labels. Let’s see an example, assuming
<code>segmentation_maps = [[2,6,7,9]]</code>, the output will contain <code>mask_labels = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]</code> (four binary masks) and <code>class_labels = [2,6,7,9]</code>, the labels for
each mask.`,Do,Q,We,Ao,Mt,Ks=`Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a> into semantic segmentation maps. Only supports
PyTorch.`,Yo,K,Je,Qo,yt,en=`Converts the output of <code>MaskFormerForInstanceSegmentationOutput</code> into instance segmentation predictions. Only
supports PyTorch. If instances could overlap, set either return_coco_annotation or return_binary_maps
to <code>True</code> to get the correct segmentation result.`,Ko,ee,Re,es,Ft,tn=`Converts the output of <code>MaskFormerForInstanceSegmentationOutput</code> into image panoptic segmentation
predictions. Only supports PyTorch.`,mo,Le,po,z,Be,ts,vt,on="Constructs a fast Maskformer image processor.",os,wt,Se,ss,te,Ee,ns,Tt,sn=`Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a> into semantic segmentation maps. Only supports
PyTorch.`,rs,oe,He,as,xt,nn=`Converts the output of <code>MaskFormerForInstanceSegmentationOutput</code> into instance segmentation predictions. Only
supports PyTorch. If instances could overlap, set either return_coco_annotation or return_binary_maps
to <code>True</code> to get the correct segmentation result.`,is,se,Oe,ds,It,rn=`Converts the output of <code>MaskFormerForInstanceSegmentationOutput</code> into image panoptic segmentation
predictions. Only supports PyTorch.`,go,Xe,ho,j,Ve,cs,$t,qe,ls,E,Ge,ms,zt,an="Pad images up to the largest image in a batch and create a corresponding <code>pixel_mask</code>.",ps,jt,dn=`MaskFormer addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps
will be converted to lists of binary masks and their respective labels. Let’s see an example, assuming
<code>segmentation_maps = [[2,6,7,9]]</code>, the output will contain <code>mask_labels = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]</code> (four binary masks) and <code>class_labels = [2,6,7,9]</code>, the labels for
each mask.`,gs,ne,De,hs,Nt,cn=`Converts the output of <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a> into semantic segmentation maps. Only supports
PyTorch.`,fs,re,Ae,us,Pt,ln=`Converts the output of <code>MaskFormerForInstanceSegmentationOutput</code> into instance segmentation predictions. Only
supports PyTorch. If instances could overlap, set either return_coco_annotation or return_binary_maps
to <code>True</code> to get the correct segmentation result.`,_s,ae,Ye,bs,Ut,mn=`Converts the output of <code>MaskFormerForInstanceSegmentationOutput</code> into image panoptic segmentation
predictions. Only supports PyTorch.`,fo,Qe,uo,U,Ke,ks,Ct,pn="The bare Maskformer Model outputting raw hidden-states without any specific head on top.",Ms,Zt,gn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ys,Wt,hn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fs,Z,et,vs,Jt,fn='The <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerModel">MaskFormerModel</a> forward method, overrides the <code>__call__</code> special method.',ws,ie,Ts,de,_o,tt,bo,X,ot,xs,P,st,Is,Rt,un='The <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a> forward method, overrides the <code>__call__</code> special method.',$s,ce,zs,Lt,_n="Examples:",js,le,Ns,me,ko,nt,Mo,St,yo;return ge=new B({props:{title:"MaskFormer",local:"maskformer",headingTag:"h1"}}),D=new Ps({props:{$$slots:{default:[$n]},$$scope:{ctx:N}}}),he=new B({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ye=new B({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),ve=new B({props:{title:"Resources",local:"resources",headingTag:"h2"}}),we=new xn({props:{pipeline:"image-segmentation"}}),xe=new B({props:{title:"MaskFormer specific outputs",local:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput",headingTag:"h2"}}),Ie=new w({props:{name:"class transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput",anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput",parameters:[{name:"encoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_decoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"transformer_decoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"pixel_decoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"transformer_decoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.encoder_last_hidden_state",description:`<strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Last hidden states (final feature map) of the last stage of the encoder model (backbone).`,name:"encoder_last_hidden_state"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.pixel_decoder_last_hidden_state",description:`<strong>pixel_decoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).`,name:"pixel_decoder_last_hidden_state"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.transformer_decoder_last_hidden_state",description:`<strong>transformer_decoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) &#x2014;
Last hidden states (final feature map) of the last stage of the transformer decoder model.`,name:"transformer_decoder_last_hidden_state"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, num_channels, height, width)</code>. Hidden-states (also called feature maps) of the encoder
model at the output of each stage.`,name:"encoder_hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.pixel_decoder_hidden_states",description:`<strong>pixel_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, num_channels, height, width)</code>. Hidden-states (also called feature maps) of the pixel
decoder model at the output of each stage.`,name:"pixel_decoder_hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.transformer_decoder_hidden_states",description:`<strong>transformer_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, sequence_length, hidden_size)</code>. Hidden-states (also called feature maps) of the
transformer decoder at the output of each stage.`,name:"transformer_decoder_hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.hidden_states",description:`<strong>hidden_states</strong> <code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> containing <code>encoder_hidden_states</code>, <code>pixel_decoder_hidden_states</code> and
<code>decoder_hidden_states</code>`,name:"hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/modeling_maskformer.py#L136"}}),$e=new w({props:{name:"class transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput",anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"class_queries_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"masks_queries_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"auxiliary_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"pixel_decoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"transformer_decoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"pixel_decoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"transformer_decoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.loss",description:`<strong>loss</strong> (<code>torch.Tensor</code>, <em>optional</em>) &#x2014;
The computed loss, returned when labels are present.`,name:"loss"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.class_queries_logits",description:`<strong>class_queries_logits</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) &#x2014;
A tensor of shape <code>(batch_size, num_queries, num_labels + 1)</code> representing the proposed classes for each
query. Note the <code>+ 1</code> is needed because we incorporate the null class.`,name:"class_queries_logits"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.masks_queries_logits",description:`<strong>masks_queries_logits</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) &#x2014;
A tensor of shape <code>(batch_size, num_queries, height, width)</code> representing the proposed masks for each
query.`,name:"masks_queries_logits"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.auxiliary_logits",description:`<strong>auxiliary_logits</strong> (<code>Dict[str, torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_auxiliary_logits=True</code>) &#x2014;
Dictionary containing auxiliary predictions for each decoder layer when auxiliary losses are enabled.`,name:"auxiliary_logits"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.encoder_last_hidden_state",description:`<strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Last hidden states (final feature map) of the last stage of the encoder model (backbone).`,name:"encoder_last_hidden_state"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.pixel_decoder_last_hidden_state",description:`<strong>pixel_decoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) &#x2014;
Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).`,name:"pixel_decoder_last_hidden_state"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.transformer_decoder_last_hidden_state",description:`<strong>transformer_decoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) &#x2014;
Last hidden states (final feature map) of the last stage of the transformer decoder model.`,name:"transformer_decoder_last_hidden_state"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, num_channels, height, width)</code>. Hidden-states (also called feature maps) of the encoder
model at the output of each stage.`,name:"encoder_hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.pixel_decoder_hidden_states",description:`<strong>pixel_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, num_channels, height, width)</code>. Hidden-states (also called feature maps) of the pixel
decoder model at the output of each stage.`,name:"pixel_decoder_hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.transformer_decoder_hidden_states",description:`<strong>transformer_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, sequence_length, hidden_size)</code>. Hidden-states of the transformer decoder at the output
of each stage.`,name:"transformer_decoder_hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.hidden_states",description:`<strong>hidden_states</strong> <code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> containing <code>encoder_hidden_states</code>, <code>pixel_decoder_hidden_states</code> and
<code>decoder_hidden_states</code>.`,name:"hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/modeling_maskformer.py#L182"}}),ze=new B({props:{title:"MaskFormerConfig",local:"transformers.MaskFormerConfig",headingTag:"h2"}}),je=new w({props:{name:"class transformers.MaskFormerConfig",anchor:"transformers.MaskFormerConfig",parameters:[{name:"fpn_feature_size",val:": int = 256"},{name:"mask_feature_size",val:": int = 256"},{name:"no_object_weight",val:": float = 0.1"},{name:"use_auxiliary_loss",val:": bool = False"},{name:"backbone_config",val:": typing.Optional[dict] = None"},{name:"decoder_config",val:": typing.Optional[dict] = None"},{name:"init_std",val:": float = 0.02"},{name:"init_xavier_std",val:": float = 1.0"},{name:"dice_weight",val:": float = 1.0"},{name:"cross_entropy_weight",val:": float = 1.0"},{name:"mask_weight",val:": float = 20.0"},{name:"output_auxiliary_logits",val:": typing.Optional[bool] = None"},{name:"backbone",val:": typing.Optional[str] = None"},{name:"use_pretrained_backbone",val:": bool = False"},{name:"use_timm_backbone",val:": bool = False"},{name:"backbone_kwargs",val:": typing.Optional[dict] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MaskFormerConfig.mask_feature_size",description:`<strong>mask_feature_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The masks&#x2019; features size, this value will also be used to specify the Feature Pyramid Network features&#x2019;
size.`,name:"mask_feature_size"},{anchor:"transformers.MaskFormerConfig.no_object_weight",description:`<strong>no_object_weight</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Weight to apply to the null (no object) class.`,name:"no_object_weight"},{anchor:"transformers.MaskFormerConfig.use_auxiliary_loss(bool,",description:`<strong>use_auxiliary_loss(<code>bool</code>,</strong> <em>optional</em>, defaults to <code>False</code>) &#x2014;
If <code>True</code> <code>MaskFormerForInstanceSegmentationOutput</code> will contain the auxiliary losses computed using the
logits from each decoder&#x2019;s stage.`,name:"use_auxiliary_loss(bool,"},{anchor:"transformers.MaskFormerConfig.backbone_config",description:`<strong>backbone_config</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
The configuration passed to the backbone, if unset, the configuration corresponding to
<code>swin-base-patch4-window12-384</code> will be used.`,name:"backbone_config"},{anchor:"transformers.MaskFormerConfig.backbone",description:`<strong>backbone</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Name of backbone to use when <code>backbone_config</code> is <code>None</code>. If <code>use_pretrained_backbone</code> is <code>True</code>, this
will load the corresponding pretrained weights from the timm or transformers library. If <code>use_pretrained_backbone</code>
is <code>False</code>, this loads the backbone&#x2019;s config and uses that to initialize the backbone with random weights.`,name:"backbone"},{anchor:"transformers.MaskFormerConfig.use_pretrained_backbone",description:`<strong>use_pretrained_backbone</strong> (<code>bool</code>, <em>optional</em>, <code>False</code>) &#x2014;
Whether to use pretrained weights for the backbone.`,name:"use_pretrained_backbone"},{anchor:"transformers.MaskFormerConfig.use_timm_backbone",description:`<strong>use_timm_backbone</strong> (<code>bool</code>, <em>optional</em>, <code>False</code>) &#x2014;
Whether to load <code>backbone</code> from the timm library. If <code>False</code>, the backbone is loaded from the transformers
library.`,name:"use_timm_backbone"},{anchor:"transformers.MaskFormerConfig.backbone_kwargs",description:`<strong>backbone_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
e.g. <code>{&apos;out_indices&apos;: (0, 1, 2, 3)}</code>. Cannot be specified if <code>backbone_config</code> is set.`,name:"backbone_kwargs"},{anchor:"transformers.MaskFormerConfig.decoder_config",description:`<strong>decoder_config</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
The configuration passed to the transformer decoder model, if unset the base config for <code>detr-resnet-50</code>
will be used.`,name:"decoder_config"},{anchor:"transformers.MaskFormerConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.MaskFormerConfig.init_xavier_std",description:`<strong>init_xavier_std</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
The scaling factor used for the Xavier initialization gain in the HM Attention map module.`,name:"init_xavier_std"},{anchor:"transformers.MaskFormerConfig.dice_weight",description:`<strong>dice_weight</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
The weight for the dice loss.`,name:"dice_weight"},{anchor:"transformers.MaskFormerConfig.cross_entropy_weight",description:`<strong>cross_entropy_weight</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
The weight for the cross entropy loss.`,name:"cross_entropy_weight"},{anchor:"transformers.MaskFormerConfig.mask_weight",description:`<strong>mask_weight</strong> (<code>float</code>, <em>optional</em>, defaults to 20.0) &#x2014;
The weight for the mask loss.`,name:"mask_weight"},{anchor:"transformers.MaskFormerConfig.output_auxiliary_logits",description:`<strong>output_auxiliary_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Should the model output its <code>auxiliary_logits</code> or not.`,name:"output_auxiliary_logits"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/configuration_maskformer.py#L30",raiseDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<ul>
<li><code>ValueError</code> —
Raised if the backbone model type selected is not in <code>["swin"]</code> or the decoder model type selected is not
in <code>["detr"]</code></li>
</ul>
`,raiseType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>ValueError</code></p>
`}}),A=new Po({props:{anchor:"transformers.MaskFormerConfig.example",$$slots:{default:[zn]},$$scope:{ctx:N}}}),Ne=new w({props:{name:"from_backbone_and_decoder_configs",anchor:"transformers.MaskFormerConfig.from_backbone_and_decoder_configs",parameters:[{name:"backbone_config",val:": PretrainedConfig"},{name:"decoder_config",val:": PretrainedConfig"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MaskFormerConfig.from_backbone_and_decoder_configs.backbone_config",description:`<strong>backbone_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>) &#x2014;
The backbone configuration.`,name:"backbone_config"},{anchor:"transformers.MaskFormerConfig.from_backbone_and_decoder_configs.decoder_config",description:`<strong>decoder_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>) &#x2014;
The transformer decoder configuration to use.`,name:"decoder_config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/configuration_maskformer.py#L212",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>An instance of a configuration object</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig"
>MaskFormerConfig</a></p>
`}}),Pe=new B({props:{title:"MaskFormerImageProcessor",local:"transformers.MaskFormerImageProcessor",headingTag:"h2"}}),Ue=new w({props:{name:"class transformers.MaskFormerImageProcessor",anchor:"transformers.MaskFormerImageProcessor",parameters:[{name:"do_resize",val:": bool = True"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"size_divisor",val:": int = 32"},{name:"resample",val:": Resampling = <Resampling.BILINEAR: 2>"},{name:"do_rescale",val:": bool = True"},{name:"rescale_factor",val:": float = 0.00392156862745098"},{name:"do_normalize",val:": bool = True"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"ignore_index",val:": typing.Optional[int] = None"},{name:"do_reduce_labels",val:": bool = False"},{name:"num_labels",val:": typing.Optional[int] = None"},{name:"pad_size",val:": typing.Optional[dict[str, int]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessor.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to resize the input to a certain <code>size</code>.`,name:"do_resize"},{anchor:"transformers.MaskFormerImageProcessor.size",description:`<strong>size</strong> (<code>int</code>, <em>optional</em>, defaults to 800) &#x2014;
Resize the input to the given size. Only has an effect if <code>do_resize</code> is set to <code>True</code>. If size is a
sequence like <code>(width, height)</code>, output size will be matched to this. If size is an int, smaller edge of
the image will be matched to this number. i.e, if <code>height &gt; width</code>, then image will be rescaled to <code>(size * height / width, size)</code>.`,name:"size"},{anchor:"transformers.MaskFormerImageProcessor.size_divisor",description:`<strong>size_divisor</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Some backbones need images divisible by a certain number. If not passed, it defaults to the value used in
Swin Transformer.`,name:"size_divisor"},{anchor:"transformers.MaskFormerImageProcessor.resample",description:`<strong>resample</strong> (<code>int</code>, <em>optional</em>, defaults to <code>Resampling.BILINEAR</code>) &#x2014;
An optional resampling filter. This can be one of <code>PIL.Image.Resampling.NEAREST</code>,
<code>PIL.Image.Resampling.BOX</code>, <code>PIL.Image.Resampling.BILINEAR</code>, <code>PIL.Image.Resampling.HAMMING</code>,
<code>PIL.Image.Resampling.BICUBIC</code> or <code>PIL.Image.Resampling.LANCZOS</code>. Only has an effect if <code>do_resize</code> is set
to <code>True</code>.`,name:"resample"},{anchor:"transformers.MaskFormerImageProcessor.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to rescale the input to a certain <code>scale</code>.`,name:"do_rescale"},{anchor:"transformers.MaskFormerImageProcessor.rescale_factor",description:`<strong>rescale_factor</strong> (<code>float</code>, <em>optional</em>, defaults to <code>1/ 255</code>) &#x2014;
Rescale the input by the given factor. Only has an effect if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.MaskFormerImageProcessor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to normalize the input with mean and standard deviation.`,name:"do_normalize"},{anchor:"transformers.MaskFormerImageProcessor.image_mean",description:`<strong>image_mean</strong> (<code>int</code>, <em>optional</em>, defaults to <code>[0.485, 0.456, 0.406]</code>) &#x2014;
The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.`,name:"image_mean"},{anchor:"transformers.MaskFormerImageProcessor.image_std",description:`<strong>image_std</strong> (<code>int</code>, <em>optional</em>, defaults to <code>[0.229, 0.224, 0.225]</code>) &#x2014;
The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
ImageNet std.`,name:"image_std"},{anchor:"transformers.MaskFormerImageProcessor.ignore_index",description:`<strong>ignore_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
denoted with 0 (background) will be replaced with <code>ignore_index</code>.`,name:"ignore_index"},{anchor:"transformers.MaskFormerImageProcessor.do_reduce_labels",description:`<strong>do_reduce_labels</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
The background label will be replaced by <code>ignore_index</code>.`,name:"do_reduce_labels"},{anchor:"transformers.MaskFormerImageProcessor.num_labels",description:`<strong>num_labels</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The number of labels in the segmentation map.`,name:"num_labels"},{anchor:"transformers.MaskFormerImageProcessor.pad_size",description:`<strong>pad_size</strong> (<code>Dict[str, int]</code>, <em>optional</em>) &#x2014;
The size <code>{&quot;height&quot;: int, &quot;width&quot; int}</code> to pad the images to. Must be larger than any image size
provided for preprocessing. If <code>pad_size</code> is not provided, images will be padded to the largest
height and width in the batch.`,name:"pad_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L397"}}),Ce=new w({props:{name:"preprocess",anchor:"transformers.MaskFormerImageProcessor.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"segmentation_maps",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"},{name:"instance_id_to_semantic_id",val:": typing.Optional[dict[int, int]] = None"},{name:"do_resize",val:": typing.Optional[bool] = None"},{name:"size",val:": typing.Optional[dict[str, int]] = None"},{name:"size_divisor",val:": typing.Optional[int] = None"},{name:"resample",val:": Resampling = None"},{name:"do_rescale",val:": typing.Optional[bool] = None"},{name:"rescale_factor",val:": typing.Optional[float] = None"},{name:"do_normalize",val:": typing.Optional[bool] = None"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"ignore_index",val:": typing.Optional[int] = None"},{name:"do_reduce_labels",val:": typing.Optional[bool] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels_first'>"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"pad_size",val:": typing.Optional[dict[str, int]] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L708"}}),Ze=new w({props:{name:"encode_inputs",anchor:"transformers.MaskFormerImageProcessor.encode_inputs",parameters:[{name:"pixel_values_list",val:": list"},{name:"segmentation_maps",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None"},{name:"instance_id_to_semantic_id",val:": typing.Union[list[dict[int, int]], dict[int, int], NoneType] = None"},{name:"ignore_index",val:": typing.Optional[int] = None"},{name:"do_reduce_labels",val:": bool = False"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"pad_size",val:": typing.Optional[dict[str, int]] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessor.encode_inputs.pixel_values_list",description:`<strong>pixel_values_list</strong> (<code>list[ImageInput]</code>) &#x2014;
List of images (pixel values) to be padded. Each image should be a tensor of shape <code>(channels, height, width)</code>.`,name:"pixel_values_list"},{anchor:"transformers.MaskFormerImageProcessor.encode_inputs.segmentation_maps",description:`<strong>segmentation_maps</strong> (<code>ImageInput</code>, <em>optional</em>) &#x2014;
The corresponding semantic segmentation maps with the pixel-wise annotations.</p>
<p>(<code>bool</code>, <em>optional</em>, defaults to <code>True</code>):
Whether or not to pad images up to the largest image in a batch and create a pixel mask.</p>
<p>If left to the default, will return a pixel mask that is:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>`,name:"segmentation_maps"},{anchor:"transformers.MaskFormerImageProcessor.encode_inputs.instance_id_to_semantic_id",description:`<strong>instance_id_to_semantic_id</strong> (<code>list[dict[int, int]]</code> or <code>dict[int, int]</code>, <em>optional</em>) &#x2014;
A mapping between object instance ids and class ids. If passed, <code>segmentation_maps</code> is treated as an
instance segmentation map where each pixel represents an instance id. Can be provided as a single
dictionary with a global/dataset-level mapping or as a list of dictionaries (one per image), to map
instance ids in each image separately.`,name:"instance_id_to_semantic_id"},{anchor:"transformers.MaskFormerImageProcessor.encode_inputs.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of NumPy arrays. If set to <code>&apos;pt&apos;</code>, return PyTorch <code>torch.Tensor</code>
objects.`,name:"return_tensors"},{anchor:"transformers.MaskFormerImageProcessor.encode_inputs.pad_size",description:`<strong>pad_size</strong> (<code>Dict[str, int]</code>, <em>optional</em>) &#x2014;
The size <code>{&quot;height&quot;: int, &quot;width&quot; int}</code> to pad the images to. Must be larger than any image size
provided for preprocessing. If <code>pad_size</code> is not provided, images will be padded to the largest
height and width in the batch.`,name:"pad_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L903",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>pixel_values</strong> — Pixel values to be fed to a model.</li>
<li><strong>pixel_mask</strong> — Pixel mask to be fed to a model (when <code>=True</code> or if <code>pixel_mask</code> is in
<code>self.model_input_names</code>).</li>
<li><strong>mask_labels</strong> — Optional list of mask labels of shape <code>(labels, height, width)</code> to be fed to a model
(when <code>annotations</code> are provided).</li>
<li><strong>class_labels</strong> — Optional list of class labels of shape <code>(labels)</code> to be fed to a model (when
<code>annotations</code> are provided). They identify the labels of <code>mask_labels</code>, e.g. the label of
<code>mask_labels[i][j]</code> if <code>class_labels[i][j]</code>.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),We=new w({props:{name:"post_process_semantic_segmentation",anchor:"transformers.MaskFormerImageProcessor.post_process_semantic_segmentation",parameters:[{name:"outputs",val:""},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessor.post_process_semantic_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.MaskFormerImageProcessor.post_process_semantic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[tuple[int, int]]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction. If left to None, predictions will not be resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L1066",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of length <code>batch_size</code>, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target_sizes entry (if <code>target_sizes</code> is specified). Each entry of each
<code>torch.Tensor</code> correspond to a semantic class id.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[torch.Tensor]</code></p>
`}}),Je=new w({props:{name:"post_process_instance_segmentation",anchor:"transformers.MaskFormerImageProcessor.post_process_instance_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"},{name:"return_coco_annotation",val:": typing.Optional[bool] = False"},{name:"return_binary_maps",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessor.post_process_instance_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.MaskFormerImageProcessor.post_process_instance_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.MaskFormerImageProcessor.post_process_instance_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.MaskFormerImageProcessor.post_process_instance_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.MaskFormerImageProcessor.post_process_instance_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction. If left to None, predictions will not be resized.`,name:"target_sizes"},{anchor:"transformers.MaskFormerImageProcessor.post_process_instance_segmentation.return_coco_annotation",description:`<strong>return_coco_annotation</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code>, segmentation maps are returned in COCO run-length encoding (RLE) format.`,name:"return_coco_annotation"},{anchor:"transformers.MaskFormerImageProcessor.post_process_instance_segmentation.return_binary_maps",description:`<strong>return_binary_maps</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code>, segmentation maps are returned as a concatenated tensor of binary segmentation maps
(one per detected instance).`,name:"return_binary_maps"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L1116",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — A tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code>, or
<code>list[List]</code> run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
<code>True</code>, or a tensor of shape <code>(num_instances, height, width)</code> if return_binary_maps is set to <code>True</code>.
Set to <code>None</code> if no mask if found above <code>threshold</code>.</li>
<li><strong>segments_info</strong> — A dictionary that contains additional information on each segment.<ul>
<li><strong>id</strong> — An integer representing the <code>segment_id</code>.</li>
<li><strong>label_id</strong> — An integer representing the label / semantic class id corresponding to <code>segment_id</code>.</li>
<li><strong>score</strong> — Prediction score of segment with <code>segment_id</code>.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),Re=new w({props:{name:"post_process_panoptic_segmentation",anchor:"transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"label_ids_to_fuse",val:": typing.Optional[set[int]] = None"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation.outputs",description:`<strong>outputs</strong> (<code>MaskFormerForInstanceSegmentationOutput</code>) &#x2014;
The outputs from <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>.`,name:"outputs"},{anchor:"transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation.label_ids_to_fuse",description:`<strong>label_ids_to_fuse</strong> (<code>Set[int]</code>, <em>optional</em>) &#x2014;
The labels in this state will have all their instances be fused together. For instance we could say
there can only be one sky in an image, but several persons, so the label ID for sky would be in that
set, but not the one for person.`,name:"label_ids_to_fuse"},{anchor:"transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction in batch. If left to None, predictions will not be
resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L1232",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — a tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code>, set
to <code>None</code> if no mask if found above <code>threshold</code>. If <code>target_sizes</code> is specified, segmentation is resized
to the corresponding <code>target_sizes</code> entry.</li>
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
`}}),Le=new B({props:{title:"MaskFormerImageProcessorFast",local:"transformers.MaskFormerImageProcessorFast",headingTag:"h2"}}),Be=new w({props:{name:"class transformers.MaskFormerImageProcessorFast",anchor:"transformers.MaskFormerImageProcessorFast",parameters:[{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.maskformer.image_processing_maskformer_fast.MaskFormerFastImageProcessorKwargs]"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L143"}}),Se=new w({props:{name:"preprocess",anchor:"transformers.MaskFormerImageProcessorFast.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"},{name:"segmentation_maps",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"},{name:"instance_id_to_semantic_id",val:": typing.Union[list[dict[int, int]], dict[int, int], NoneType] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.maskformer.image_processing_maskformer_fast.MaskFormerFastImageProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.images",description:`<strong>images</strong> (<code>Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[&apos;PIL.Image.Image&apos;], list[numpy.ndarray], list[&apos;torch.Tensor&apos;]]</code>) &#x2014;
Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
passing in images with pixel values between 0 and 1, set <code>do_rescale=False</code>.`,name:"images"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.segmentation_maps",description:`<strong>segmentation_maps</strong> (<code>ImageInput</code>, <em>optional</em>) &#x2014;
The segmentation maps.`,name:"segmentation_maps"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.instance_id_to_semantic_id",description:`<strong>instance_id_to_semantic_id</strong> (<code>Union[list[dict[int, int]], dict[int, int]]</code>, <em>optional</em>) &#x2014;
A mapping from instance IDs to semantic IDs.`,name:"instance_id_to_semantic_id"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.do_resize",description:`<strong>do_resize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to resize the image.`,name:"do_resize"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.size",description:`<strong>size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
Describes the maximum input dimensions to the model.`,name:"size"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.default_to_square",description:`<strong>default_to_square</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to default to a square image when resizing, if size is an int.`,name:"default_to_square"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.resample",description:`<strong>resample</strong> (<code>Union[PILImageResampling, F.InterpolationMode, NoneType]</code>) &#x2014;
Resampling filter to use if resizing the image. This can be one of the enum <code>PILImageResampling</code>. Only
has an effect if <code>do_resize</code> is set to <code>True</code>.`,name:"resample"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.do_center_crop",description:`<strong>do_center_crop</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to center crop the image.`,name:"do_center_crop"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.crop_size",description:`<strong>crop_size</strong> (<code>dict[str, int]</code>, <em>optional</em>) &#x2014;
Size of the output image after applying <code>center_crop</code>.`,name:"crop_size"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>Union[int, float, NoneType]</code>) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>Union[float, list[float], NoneType]</code>) &#x2014;
Image mean to use for normalization. Only has an effect if <code>do_normalize</code> is set to <code>True</code>.`,name:"image_mean"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.image_std",description:`<strong>image_std</strong> (<code>Union[float, list[float], NoneType]</code>) &#x2014;
Image standard deviation to use for normalization. Only has an effect if <code>do_normalize</code> is set to
<code>True</code>.`,name:"image_std"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.return_tensors",description:"<strong>return_tensors</strong> (<code>Union[str, ~utils.generic.TensorType, NoneType]</code>) &#x2014;\nReturns stacked tensors if set to `pt, otherwise returns a list of tensors.",name:"return_tensors"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.data_format",description:`<strong>data_format</strong> (<code>~image_utils.ChannelDimension</code>, <em>optional</em>) &#x2014;
Only <code>ChannelDimension.FIRST</code> is supported. Added for compatibility with slow processors.`,name:"data_format"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>Union[str, ~image_utils.ChannelDimension, NoneType]</code>) &#x2014;
The channel dimension format for the input image. If unset, the channel dimension format is inferred
from the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.</li>
<li><code>&quot;none&quot;</code> or <code>ChannelDimension.NONE</code>: image in (height, width) format.</li>
</ul>`,name:"input_data_format"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.device",description:`<strong>device</strong> (<code>torch.device</code>, <em>optional</em>) &#x2014;
The device to process the images on. If unset, the device is inferred from the input images.`,name:"device"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.disable_grouping",description:`<strong>disable_grouping</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to disable grouping of images by size to process them individually and not in batches.
If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
empirical observations, as detailed here: <a href="https://github.com/huggingface/transformers/pull/38157" rel="nofollow">https://github.com/huggingface/transformers/pull/38157</a>`,name:"disable_grouping"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.size_divisor",description:`<strong>size_divisor</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Some backbones need images divisible by a certain number. If not passed, it defaults to the value used in
Swin Transformer.`,name:"size_divisor"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.ignore_index",description:`<strong>ignore_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
denoted with 0 (background) will be replaced with <code>ignore_index</code>.`,name:"ignore_index"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.do_reduce_labels",description:`<strong>do_reduce_labels</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
The background label will be replaced by <code>ignore_index</code>.`,name:"do_reduce_labels"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.num_labels",description:`<strong>num_labels</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The number of labels in the segmentation map.`,name:"num_labels"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.do_pad",description:`<strong>do_pad</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Controls whether to pad the image. Can be overridden by the <code>do_pad</code> parameter in the <code>preprocess</code>
method. If <code>True</code>, padding will be applied to the bottom and right of the image with zeros.
If <code>pad_size</code> is provided, the image will be padded to the specified dimensions.
Otherwise, the image will be padded to the maximum height and width of the batch.`,name:"do_pad"},{anchor:"transformers.MaskFormerImageProcessorFast.preprocess.pad_size",description:`<strong>pad_size</strong> (<code>Dict[str, int]</code>, <em>optional</em>) &#x2014;
The size <code>{&quot;height&quot;: int, &quot;width&quot; int}</code> to pad the images to. Must be larger than any image size
provided for preprocessing. If <code>pad_size</code> is not provided, images will be padded to the largest
height and width in the batch.`,name:"pad_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L283",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<ul>
<li><strong>data</strong> (<code>dict</code>) — Dictionary of lists/arrays/tensors returned by the <strong>call</strong> method (‘pixel_values’, etc.).</li>
<li><strong>tensor_type</strong> (<code>Union[None, str, TensorType]</code>, <em>optional</em>) — You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
initialization.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>&lt;class 'transformers.image_processing_base.BatchFeature'&gt;</code></p>
`}}),Ee=new w({props:{name:"post_process_semantic_segmentation",anchor:"transformers.MaskFormerImageProcessorFast.post_process_semantic_segmentation",parameters:[{name:"outputs",val:""},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessorFast.post_process_semantic_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_semantic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[tuple[int, int]]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction. If left to None, predictions will not be resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L500",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of length <code>batch_size</code>, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target_sizes entry (if <code>target_sizes</code> is specified). Each entry of each
<code>torch.Tensor</code> correspond to a semantic class id.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[torch.Tensor]</code></p>
`}}),He=new w({props:{name:"post_process_instance_segmentation",anchor:"transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"},{name:"return_coco_annotation",val:": typing.Optional[bool] = False"},{name:"return_binary_maps",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction. If left to None, predictions will not be resized.`,name:"target_sizes"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation.return_coco_annotation",description:`<strong>return_coco_annotation</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code>, segmentation maps are returned in COCO run-length encoding (RLE) format.`,name:"return_coco_annotation"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation.return_binary_maps",description:`<strong>return_binary_maps</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code>, segmentation maps are returned as a concatenated tensor of binary segmentation maps
(one per detected instance).`,name:"return_binary_maps"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L551",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — A tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code>, or
<code>list[List]</code> run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
<code>True</code>, or a tensor of shape <code>(num_instances, height, width)</code> if return_binary_maps is set to <code>True</code>.
Set to <code>None</code> if no mask if found above <code>threshold</code>.</li>
<li><strong>segments_info</strong> — A dictionary that contains additional information on each segment.<ul>
<li><strong>id</strong> — An integer representing the <code>segment_id</code>.</li>
<li><strong>label_id</strong> — An integer representing the label / semantic class id corresponding to <code>segment_id</code>.</li>
<li><strong>score</strong> — Prediction score of segment with <code>segment_id</code>.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),Oe=new w({props:{name:"post_process_panoptic_segmentation",anchor:"transformers.MaskFormerImageProcessorFast.post_process_panoptic_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"label_ids_to_fuse",val:": typing.Optional[set[int]] = None"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerImageProcessorFast.post_process_panoptic_segmentation.outputs",description:`<strong>outputs</strong> (<code>MaskFormerForInstanceSegmentationOutput</code>) &#x2014;
The outputs from <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>.`,name:"outputs"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_panoptic_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_panoptic_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_panoptic_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_panoptic_segmentation.label_ids_to_fuse",description:`<strong>label_ids_to_fuse</strong> (<code>Set[int]</code>, <em>optional</em>) &#x2014;
The labels in this state will have all their instances be fused together. For instance we could say
there can only be one sky in an image, but several persons, so the label ID for sky would be in that
set, but not the one for person.`,name:"label_ids_to_fuse"},{anchor:"transformers.MaskFormerImageProcessorFast.post_process_panoptic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction in batch. If left to None, predictions will not be
resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L668",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — a tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code>, set
to <code>None</code> if no mask if found above <code>threshold</code>. If <code>target_sizes</code> is specified, segmentation is resized
to the corresponding <code>target_sizes</code> entry.</li>
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
`}}),Xe=new B({props:{title:"MaskFormerFeatureExtractor",local:"transformers.MaskFormerFeatureExtractor",headingTag:"h2"}}),Ve=new w({props:{name:"class transformers.MaskFormerFeatureExtractor",anchor:"transformers.MaskFormerFeatureExtractor",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/feature_extraction_maskformer.py#L28"}}),qe=new w({props:{name:"__call__",anchor:"transformers.MaskFormerFeatureExtractor.__call__",parameters:[{name:"images",val:""},{name:"segmentation_maps",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L602"}}),Ge=new w({props:{name:"encode_inputs",anchor:"transformers.MaskFormerFeatureExtractor.encode_inputs",parameters:[{name:"pixel_values_list",val:": list"},{name:"segmentation_maps",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None"},{name:"instance_id_to_semantic_id",val:": typing.Union[list[dict[int, int]], dict[int, int], NoneType] = None"},{name:"ignore_index",val:": typing.Optional[int] = None"},{name:"do_reduce_labels",val:": bool = False"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"pad_size",val:": typing.Optional[dict[str, int]] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerFeatureExtractor.encode_inputs.pixel_values_list",description:`<strong>pixel_values_list</strong> (<code>list[ImageInput]</code>) &#x2014;
List of images (pixel values) to be padded. Each image should be a tensor of shape <code>(channels, height, width)</code>.`,name:"pixel_values_list"},{anchor:"transformers.MaskFormerFeatureExtractor.encode_inputs.segmentation_maps",description:`<strong>segmentation_maps</strong> (<code>ImageInput</code>, <em>optional</em>) &#x2014;
The corresponding semantic segmentation maps with the pixel-wise annotations.</p>
<p>(<code>bool</code>, <em>optional</em>, defaults to <code>True</code>):
Whether or not to pad images up to the largest image in a batch and create a pixel mask.</p>
<p>If left to the default, will return a pixel mask that is:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>`,name:"segmentation_maps"},{anchor:"transformers.MaskFormerFeatureExtractor.encode_inputs.instance_id_to_semantic_id",description:`<strong>instance_id_to_semantic_id</strong> (<code>list[dict[int, int]]</code> or <code>dict[int, int]</code>, <em>optional</em>) &#x2014;
A mapping between object instance ids and class ids. If passed, <code>segmentation_maps</code> is treated as an
instance segmentation map where each pixel represents an instance id. Can be provided as a single
dictionary with a global/dataset-level mapping or as a list of dictionaries (one per image), to map
instance ids in each image separately.`,name:"instance_id_to_semantic_id"},{anchor:"transformers.MaskFormerFeatureExtractor.encode_inputs.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of NumPy arrays. If set to <code>&apos;pt&apos;</code>, return PyTorch <code>torch.Tensor</code>
objects.`,name:"return_tensors"},{anchor:"transformers.MaskFormerFeatureExtractor.encode_inputs.pad_size",description:`<strong>pad_size</strong> (<code>Dict[str, int]</code>, <em>optional</em>) &#x2014;
The size <code>{&quot;height&quot;: int, &quot;width&quot; int}</code> to pad the images to. Must be larger than any image size
provided for preprocessing. If <code>pad_size</code> is not provided, images will be padded to the largest
height and width in the batch.`,name:"pad_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L903",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>pixel_values</strong> — Pixel values to be fed to a model.</li>
<li><strong>pixel_mask</strong> — Pixel mask to be fed to a model (when <code>=True</code> or if <code>pixel_mask</code> is in
<code>self.model_input_names</code>).</li>
<li><strong>mask_labels</strong> — Optional list of mask labels of shape <code>(labels, height, width)</code> to be fed to a model
(when <code>annotations</code> are provided).</li>
<li><strong>class_labels</strong> — Optional list of class labels of shape <code>(labels)</code> to be fed to a model (when
<code>annotations</code> are provided). They identify the labels of <code>mask_labels</code>, e.g. the label of
<code>mask_labels[i][j]</code> if <code>class_labels[i][j]</code>.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),De=new w({props:{name:"post_process_semantic_segmentation",anchor:"transformers.MaskFormerFeatureExtractor.post_process_semantic_segmentation",parameters:[{name:"outputs",val:""},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerFeatureExtractor.post_process_semantic_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_semantic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[tuple[int, int]]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction. If left to None, predictions will not be resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L1066",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of length <code>batch_size</code>, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target_sizes entry (if <code>target_sizes</code> is specified). Each entry of each
<code>torch.Tensor</code> correspond to a semantic class id.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[torch.Tensor]</code></p>
`}}),Ae=new w({props:{name:"post_process_instance_segmentation",anchor:"transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"},{name:"return_coco_annotation",val:": typing.Optional[bool] = False"},{name:"return_binary_maps",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation.outputs",description:`<strong>outputs</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>) &#x2014;
Raw outputs of the model.`,name:"outputs"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction. If left to None, predictions will not be resized.`,name:"target_sizes"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation.return_coco_annotation",description:`<strong>return_coco_annotation</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code>, segmentation maps are returned in COCO run-length encoding (RLE) format.`,name:"return_coco_annotation"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_instance_segmentation.return_binary_maps",description:`<strong>return_binary_maps</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code>, segmentation maps are returned as a concatenated tensor of binary segmentation maps
(one per detected instance).`,name:"return_binary_maps"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L1116",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — A tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code>, or
<code>list[List]</code> run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
<code>True</code>, or a tensor of shape <code>(num_instances, height, width)</code> if return_binary_maps is set to <code>True</code>.
Set to <code>None</code> if no mask if found above <code>threshold</code>.</li>
<li><strong>segments_info</strong> — A dictionary that contains additional information on each segment.<ul>
<li><strong>id</strong> — An integer representing the <code>segment_id</code>.</li>
<li><strong>label_id</strong> — An integer representing the label / semantic class id corresponding to <code>segment_id</code>.</li>
<li><strong>score</strong> — Prediction score of segment with <code>segment_id</code>.</li>
</ul></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Dict]</code></p>
`}}),Ye=new w({props:{name:"post_process_panoptic_segmentation",anchor:"transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation",parameters:[{name:"outputs",val:""},{name:"threshold",val:": float = 0.5"},{name:"mask_threshold",val:": float = 0.5"},{name:"overlap_mask_area_threshold",val:": float = 0.8"},{name:"label_ids_to_fuse",val:": typing.Optional[set[int]] = None"},{name:"target_sizes",val:": typing.Optional[list[tuple[int, int]]] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation.outputs",description:`<strong>outputs</strong> (<code>MaskFormerForInstanceSegmentationOutput</code>) &#x2014;
The outputs from <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation">MaskFormerForInstanceSegmentation</a>.`,name:"outputs"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation.threshold",description:`<strong>threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The probability score threshold to keep predicted instance masks.`,name:"threshold"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation.mask_threshold",description:`<strong>mask_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
Threshold to use when turning the predicted masks into binary values.`,name:"mask_threshold"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation.overlap_mask_area_threshold",description:`<strong>overlap_mask_area_threshold</strong> (<code>float</code>, <em>optional</em>, defaults to 0.8) &#x2014;
The overlap mask area threshold to merge or discard small disconnected parts within each binary
instance mask.`,name:"overlap_mask_area_threshold"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation.label_ids_to_fuse",description:`<strong>label_ids_to_fuse</strong> (<code>Set[int]</code>, <em>optional</em>) &#x2014;
The labels in this state will have all their instances be fused together. For instance we could say
there can only be one sky in an image, but several persons, so the label ID for sky would be in that
set, but not the one for person.`,name:"label_ids_to_fuse"},{anchor:"transformers.MaskFormerFeatureExtractor.post_process_panoptic_segmentation.target_sizes",description:`<strong>target_sizes</strong> (<code>list[Tuple]</code>, <em>optional</em>) &#x2014;
List of length (batch_size), where each list item (<code>tuple[int, int]]</code>) corresponds to the requested
final size (height, width) of each prediction in batch. If left to None, predictions will not be
resized.`,name:"target_sizes"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/image_processing_maskformer.py#L1232",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of dictionaries, one per image, each dictionary containing two keys:</p>
<ul>
<li><strong>segmentation</strong> — a tensor of shape <code>(height, width)</code> where each pixel represents a <code>segment_id</code>, set
to <code>None</code> if no mask if found above <code>threshold</code>. If <code>target_sizes</code> is specified, segmentation is resized
to the corresponding <code>target_sizes</code> entry.</li>
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
`}}),Qe=new B({props:{title:"MaskFormerModel",local:"transformers.MaskFormerModel",headingTag:"h2"}}),Ke=new w({props:{name:"class transformers.MaskFormerModel",anchor:"transformers.MaskFormerModel",parameters:[{name:"config",val:": MaskFormerConfig"}],parametersDescription:[{anchor:"transformers.MaskFormerModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig">MaskFormerConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/modeling_maskformer.py#L1479"}}),et=new w({props:{name:"forward",anchor:"transformers.MaskFormerModel.forward",parameters:[{name:"pixel_values",val:": Tensor"},{name:"pixel_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor">MaskFormerImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerFeatureExtractor.__call__">MaskFormerImageProcessor.<strong>call</strong>()</a> for details (<code>processor_class</code> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor">MaskFormerImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.MaskFormerModel.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.MaskFormerModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MaskFormerModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MaskFormerModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/modeling_maskformer.py#L1489",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput"
>transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig"
>MaskFormerConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) — Last hidden states (final feature map) of the last stage of the encoder model (backbone).</p>
</li>
<li>
<p><strong>pixel_decoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) — Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).</p>
</li>
<li>
<p><strong>transformer_decoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Last hidden states (final feature map) of the last stage of the transformer decoder model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, num_channels, height, width)</code>. Hidden-states (also called feature maps) of the encoder
model at the output of each stage.</p>
</li>
<li>
<p><strong>pixel_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, num_channels, height, width)</code>. Hidden-states (also called feature maps) of the pixel
decoder model at the output of each stage.</p>
</li>
<li>
<p><strong>transformer_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, sequence_length, hidden_size)</code>. Hidden-states (also called feature maps) of the
transformer decoder at the output of each stage.</p>
</li>
<li>
<p><strong>hidden_states</strong> <code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> containing <code>encoder_hidden_states</code>, <code>pixel_decoder_hidden_states</code> and
<code>decoder_hidden_states</code></p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput"
>transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ie=new Ps({props:{$$slots:{default:[jn]},$$scope:{ctx:N}}}),de=new Po({props:{anchor:"transformers.MaskFormerModel.forward.example",$$slots:{default:[Nn]},$$scope:{ctx:N}}}),tt=new B({props:{title:"MaskFormerForInstanceSegmentation",local:"transformers.MaskFormerForInstanceSegmentation",headingTag:"h2"}}),ot=new w({props:{name:"class transformers.MaskFormerForInstanceSegmentation",anchor:"transformers.MaskFormerForInstanceSegmentation",parameters:[{name:"config",val:": MaskFormerConfig"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/modeling_maskformer.py#L1575"}}),st=new w({props:{name:"forward",anchor:"transformers.MaskFormerForInstanceSegmentation.forward",parameters:[{name:"pixel_values",val:": Tensor"},{name:"mask_labels",val:": typing.Optional[list[torch.Tensor]] = None"},{name:"class_labels",val:": typing.Optional[list[torch.Tensor]] = None"},{name:"pixel_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_auxiliary_logits",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor">MaskFormerImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerFeatureExtractor.__call__">MaskFormerImageProcessor.<strong>call</strong>()</a> for details (<code>processor_class</code> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerImageProcessor">MaskFormerImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.mask_labels",description:`<strong>mask_labels</strong> (<code>list[torch.Tensor]</code>, <em>optional</em>) &#x2014;
List of mask labels of shape <code>(num_labels, height, width)</code> to be fed to a model`,name:"mask_labels"},{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.class_labels",description:`<strong>class_labels</strong> (<code>list[torch.LongTensor]</code>, <em>optional</em>) &#x2014;
list of target class labels of shape <code>(num_labels, height, width)</code> to be fed to a model. They identify the
labels of <code>mask_labels</code>, e.g. the label of <code>mask_labels[i][j]</code> if <code>class_labels[i][j]</code>.`,name:"class_labels"},{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, height, width)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.output_auxiliary_logits",description:`<strong>output_auxiliary_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to output auxiliary logits.`,name:"output_auxiliary_logits"},{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/maskformer/modeling_maskformer.py#L1657",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput"
>transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerConfig"
>MaskFormerConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.Tensor</code>, <em>optional</em>) — The computed loss, returned when labels are present.</p>
</li>
<li>
<p><strong>class_queries_logits</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — A tensor of shape <code>(batch_size, num_queries, num_labels + 1)</code> representing the proposed classes for each
query. Note the <code>+ 1</code> is needed because we incorporate the null class.</p>
</li>
<li>
<p><strong>masks_queries_logits</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — A tensor of shape <code>(batch_size, num_queries, height, width)</code> representing the proposed masks for each
query.</p>
</li>
<li>
<p><strong>auxiliary_logits</strong> (<code>Dict[str, torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_auxiliary_logits=True</code>) — Dictionary containing auxiliary predictions for each decoder layer when auxiliary losses are enabled.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) — Last hidden states (final feature map) of the last stage of the encoder model (backbone).</p>
</li>
<li>
<p><strong>pixel_decoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, height, width)</code>) — Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).</p>
</li>
<li>
<p><strong>transformer_decoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Last hidden states (final feature map) of the last stage of the transformer decoder model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, num_channels, height, width)</code>. Hidden-states (also called feature maps) of the encoder
model at the output of each stage.</p>
</li>
<li>
<p><strong>pixel_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, num_channels, height, width)</code>. Hidden-states (also called feature maps) of the pixel
decoder model at the output of each stage.</p>
</li>
<li>
<p><strong>transformer_decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each stage) of
shape <code>(batch_size, sequence_length, hidden_size)</code>. Hidden-states of the transformer decoder at the output
of each stage.</p>
</li>
<li>
<p><strong>hidden_states</strong> <code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> containing <code>encoder_hidden_states</code>, <code>pixel_decoder_hidden_states</code> and
<code>decoder_hidden_states</code>.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput"
>transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ce=new Ps({props:{$$slots:{default:[Pn]},$$scope:{ctx:N}}}),le=new Po({props:{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.example",$$slots:{default:[Un]},$$scope:{ctx:N}}}),me=new Po({props:{anchor:"transformers.MaskFormerForInstanceSegmentation.forward.example-2",$$slots:{default:[Cn]},$$scope:{ctx:N}}}),nt=new In({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/maskformer.md"}}),{c(){d=a("meta"),v=s(),b=a("p"),k=s(),y=a("p"),y.innerHTML=l,x=s(),m(ge.$$.fragment),Ht=s(),G=a("div"),G.innerHTML=Us,Ot=s(),m(D.$$.fragment),Xt=s(),m(he.$$.fragment),Vt=s(),fe=a("p"),fe.innerHTML=Cs,qt=s(),ue=a("p"),ue.textContent=Zs,Gt=s(),_e=a("p"),_e.innerHTML=Ws,Dt=s(),be=a("p"),be.innerHTML=Js,At=s(),ke=a("img"),Yt=s(),Me=a("p"),Me.innerHTML=Ls,Qt=s(),m(ye.$$.fragment),Kt=s(),Fe=a("ul"),Fe.innerHTML=Bs,eo=s(),m(ve.$$.fragment),to=s(),m(we.$$.fragment),oo=s(),Te=a("ul"),Te.innerHTML=Ss,so=s(),m(xe.$$.fragment),no=s(),O=a("div"),m(Ie.$$.fragment),Co=s(),dt=a("p"),dt.innerHTML=Es,ro=s(),L=a("div"),m($e.$$.fragment),Zo=s(),ct=a("p"),ct.innerHTML=Hs,Wo=s(),lt=a("p"),lt.innerHTML=Os,ao=s(),m(ze.$$.fragment),io=s(),$=a("div"),m(je.$$.fragment),Jo=s(),mt=a("p"),mt.innerHTML=Xs,Ro=s(),pt=a("p"),pt.innerHTML=Vs,Lo=s(),gt=a("p"),gt.innerHTML=qs,Bo=s(),m(A.$$.fragment),So=s(),Y=a("div"),m(Ne.$$.fragment),Eo=s(),ht=a("p"),ht.innerHTML=Gs,co=s(),m(Pe.$$.fragment),lo=s(),T=a("div"),m(Ue.$$.fragment),Ho=s(),ft=a("p"),ft.textContent=Ds,Oo=s(),ut=a("p"),ut.innerHTML=As,Xo=s(),_t=a("div"),m(Ce.$$.fragment),Vo=s(),S=a("div"),m(Ze.$$.fragment),qo=s(),bt=a("p"),bt.innerHTML=Ys,Go=s(),kt=a("p"),kt.innerHTML=Qs,Do=s(),Q=a("div"),m(We.$$.fragment),Ao=s(),Mt=a("p"),Mt.innerHTML=Ks,Yo=s(),K=a("div"),m(Je.$$.fragment),Qo=s(),yt=a("p"),yt.innerHTML=en,Ko=s(),ee=a("div"),m(Re.$$.fragment),es=s(),Ft=a("p"),Ft.innerHTML=tn,mo=s(),m(Le.$$.fragment),po=s(),z=a("div"),m(Be.$$.fragment),ts=s(),vt=a("p"),vt.textContent=on,os=s(),wt=a("div"),m(Se.$$.fragment),ss=s(),te=a("div"),m(Ee.$$.fragment),ns=s(),Tt=a("p"),Tt.innerHTML=sn,rs=s(),oe=a("div"),m(He.$$.fragment),as=s(),xt=a("p"),xt.innerHTML=nn,is=s(),se=a("div"),m(Oe.$$.fragment),ds=s(),It=a("p"),It.innerHTML=rn,go=s(),m(Xe.$$.fragment),ho=s(),j=a("div"),m(Ve.$$.fragment),cs=s(),$t=a("div"),m(qe.$$.fragment),ls=s(),E=a("div"),m(Ge.$$.fragment),ms=s(),zt=a("p"),zt.innerHTML=an,ps=s(),jt=a("p"),jt.innerHTML=dn,gs=s(),ne=a("div"),m(De.$$.fragment),hs=s(),Nt=a("p"),Nt.innerHTML=cn,fs=s(),re=a("div"),m(Ae.$$.fragment),us=s(),Pt=a("p"),Pt.innerHTML=ln,_s=s(),ae=a("div"),m(Ye.$$.fragment),bs=s(),Ut=a("p"),Ut.innerHTML=mn,fo=s(),m(Qe.$$.fragment),uo=s(),U=a("div"),m(Ke.$$.fragment),ks=s(),Ct=a("p"),Ct.textContent=pn,Ms=s(),Zt=a("p"),Zt.innerHTML=gn,ys=s(),Wt=a("p"),Wt.innerHTML=hn,Fs=s(),Z=a("div"),m(et.$$.fragment),vs=s(),Jt=a("p"),Jt.innerHTML=fn,ws=s(),m(ie.$$.fragment),Ts=s(),m(de.$$.fragment),_o=s(),m(tt.$$.fragment),bo=s(),X=a("div"),m(ot.$$.fragment),xs=s(),P=a("div"),m(st.$$.fragment),Is=s(),Rt=a("p"),Rt.innerHTML=un,$s=s(),m(ce.$$.fragment),zs=s(),Lt=a("p"),Lt.textContent=_n,js=s(),m(le.$$.fragment),Ns=s(),m(me.$$.fragment),ko=s(),m(nt.$$.fragment),Mo=s(),St=a("p"),this.h()},l(e){const r=Tn("svelte-u9bgzb",document.head);d=i(r,"META",{name:!0,content:!0}),r.forEach(o),v=n(e),b=i(e,"P",{}),F(b).forEach(o),k=n(e),y=i(e,"P",{"data-svelte-h":!0}),_(y)!=="svelte-v88miz"&&(y.innerHTML=l),x=n(e),p(ge.$$.fragment,e),Ht=n(e),G=i(e,"DIV",{class:!0,"data-svelte-h":!0}),_(G)!=="svelte-13t8s2t"&&(G.innerHTML=Us),Ot=n(e),p(D.$$.fragment,e),Xt=n(e),p(he.$$.fragment,e),Vt=n(e),fe=i(e,"P",{"data-svelte-h":!0}),_(fe)!=="svelte-1kp7b0i"&&(fe.innerHTML=Cs),qt=n(e),ue=i(e,"P",{"data-svelte-h":!0}),_(ue)!=="svelte-vfdo9a"&&(ue.textContent=Zs),Gt=n(e),_e=i(e,"P",{"data-svelte-h":!0}),_(_e)!=="svelte-lfolgg"&&(_e.innerHTML=Ws),Dt=n(e),be=i(e,"P",{"data-svelte-h":!0}),_(be)!=="svelte-mgm2yv"&&(be.innerHTML=Js),At=n(e),ke=i(e,"IMG",{width:!0,src:!0}),Yt=n(e),Me=i(e,"P",{"data-svelte-h":!0}),_(Me)!=="svelte-1j464j"&&(Me.innerHTML=Ls),Qt=n(e),p(ye.$$.fragment,e),Kt=n(e),Fe=i(e,"UL",{"data-svelte-h":!0}),_(Fe)!=="svelte-1ilfc9x"&&(Fe.innerHTML=Bs),eo=n(e),p(ve.$$.fragment,e),to=n(e),p(we.$$.fragment,e),oo=n(e),Te=i(e,"UL",{"data-svelte-h":!0}),_(Te)!=="svelte-1pmi5x7"&&(Te.innerHTML=Ss),so=n(e),p(xe.$$.fragment,e),no=n(e),O=i(e,"DIV",{class:!0});var rt=F(O);p(Ie.$$.fragment,rt),Co=n(rt),dt=i(rt,"P",{"data-svelte-h":!0}),_(dt)!=="svelte-11mvty7"&&(dt.innerHTML=Es),rt.forEach(o),ro=n(e),L=i(e,"DIV",{class:!0});var V=F(L);p($e.$$.fragment,V),Zo=n(V),ct=i(V,"P",{"data-svelte-h":!0}),_(ct)!=="svelte-1gyg194"&&(ct.innerHTML=Hs),Wo=n(V),lt=i(V,"P",{"data-svelte-h":!0}),_(lt)!=="svelte-1y9tnpc"&&(lt.innerHTML=Os),V.forEach(o),ao=n(e),p(ze.$$.fragment,e),io=n(e),$=i(e,"DIV",{class:!0});var C=F($);p(je.$$.fragment,C),Jo=n(C),mt=i(C,"P",{"data-svelte-h":!0}),_(mt)!=="svelte-eg2yi2"&&(mt.innerHTML=Xs),Ro=n(C),pt=i(C,"P",{"data-svelte-h":!0}),_(pt)!=="svelte-1ek1ss9"&&(pt.innerHTML=Vs),Lo=n(C),gt=i(C,"P",{"data-svelte-h":!0}),_(gt)!=="svelte-l3soxz"&&(gt.innerHTML=qs),Bo=n(C),p(A.$$.fragment,C),So=n(C),Y=i(C,"DIV",{class:!0});var at=F(Y);p(Ne.$$.fragment,at),Eo=n(at),ht=i(at,"P",{"data-svelte-h":!0}),_(ht)!=="svelte-1gnb91y"&&(ht.innerHTML=Gs),at.forEach(o),C.forEach(o),co=n(e),p(Pe.$$.fragment,e),lo=n(e),T=i(e,"DIV",{class:!0});var I=F(T);p(Ue.$$.fragment,I),Ho=n(I),ft=i(I,"P",{"data-svelte-h":!0}),_(ft)!=="svelte-1fb5les"&&(ft.textContent=Ds),Oo=n(I),ut=i(I,"P",{"data-svelte-h":!0}),_(ut)!=="svelte-1nyj7p5"&&(ut.innerHTML=As),Xo=n(I),_t=i(I,"DIV",{class:!0});var Et=F(_t);p(Ce.$$.fragment,Et),Et.forEach(o),Vo=n(I),S=i(I,"DIV",{class:!0});var q=F(S);p(Ze.$$.fragment,q),qo=n(q),bt=i(q,"P",{"data-svelte-h":!0}),_(bt)!=="svelte-1tetyua"&&(bt.innerHTML=Ys),Go=n(q),kt=i(q,"P",{"data-svelte-h":!0}),_(kt)!=="svelte-1r85oma"&&(kt.innerHTML=Qs),q.forEach(o),Do=n(I),Q=i(I,"DIV",{class:!0});var Fo=F(Q);p(We.$$.fragment,Fo),Ao=n(Fo),Mt=i(Fo,"P",{"data-svelte-h":!0}),_(Mt)!=="svelte-1m56dtt"&&(Mt.innerHTML=Ks),Fo.forEach(o),Yo=n(I),K=i(I,"DIV",{class:!0});var vo=F(K);p(Je.$$.fragment,vo),Qo=n(vo),yt=i(vo,"P",{"data-svelte-h":!0}),_(yt)!=="svelte-1hyddm1"&&(yt.innerHTML=en),vo.forEach(o),Ko=n(I),ee=i(I,"DIV",{class:!0});var wo=F(ee);p(Re.$$.fragment,wo),es=n(wo),Ft=i(wo,"P",{"data-svelte-h":!0}),_(Ft)!=="svelte-pw2uru"&&(Ft.innerHTML=tn),wo.forEach(o),I.forEach(o),mo=n(e),p(Le.$$.fragment,e),po=n(e),z=i(e,"DIV",{class:!0});var W=F(z);p(Be.$$.fragment,W),ts=n(W),vt=i(W,"P",{"data-svelte-h":!0}),_(vt)!=="svelte-1dp9hfz"&&(vt.textContent=on),os=n(W),wt=i(W,"DIV",{class:!0});var bn=F(wt);p(Se.$$.fragment,bn),bn.forEach(o),ss=n(W),te=i(W,"DIV",{class:!0});var To=F(te);p(Ee.$$.fragment,To),ns=n(To),Tt=i(To,"P",{"data-svelte-h":!0}),_(Tt)!=="svelte-1m56dtt"&&(Tt.innerHTML=sn),To.forEach(o),rs=n(W),oe=i(W,"DIV",{class:!0});var xo=F(oe);p(He.$$.fragment,xo),as=n(xo),xt=i(xo,"P",{"data-svelte-h":!0}),_(xt)!=="svelte-1hyddm1"&&(xt.innerHTML=nn),xo.forEach(o),is=n(W),se=i(W,"DIV",{class:!0});var Io=F(se);p(Oe.$$.fragment,Io),ds=n(Io),It=i(Io,"P",{"data-svelte-h":!0}),_(It)!=="svelte-pw2uru"&&(It.innerHTML=rn),Io.forEach(o),W.forEach(o),go=n(e),p(Xe.$$.fragment,e),ho=n(e),j=i(e,"DIV",{class:!0});var J=F(j);p(Ve.$$.fragment,J),cs=n(J),$t=i(J,"DIV",{class:!0});var kn=F($t);p(qe.$$.fragment,kn),kn.forEach(o),ls=n(J),E=i(J,"DIV",{class:!0});var Bt=F(E);p(Ge.$$.fragment,Bt),ms=n(Bt),zt=i(Bt,"P",{"data-svelte-h":!0}),_(zt)!=="svelte-1tetyua"&&(zt.innerHTML=an),ps=n(Bt),jt=i(Bt,"P",{"data-svelte-h":!0}),_(jt)!=="svelte-1r85oma"&&(jt.innerHTML=dn),Bt.forEach(o),gs=n(J),ne=i(J,"DIV",{class:!0});var $o=F(ne);p(De.$$.fragment,$o),hs=n($o),Nt=i($o,"P",{"data-svelte-h":!0}),_(Nt)!=="svelte-1m56dtt"&&(Nt.innerHTML=cn),$o.forEach(o),fs=n(J),re=i(J,"DIV",{class:!0});var zo=F(re);p(Ae.$$.fragment,zo),us=n(zo),Pt=i(zo,"P",{"data-svelte-h":!0}),_(Pt)!=="svelte-1hyddm1"&&(Pt.innerHTML=ln),zo.forEach(o),_s=n(J),ae=i(J,"DIV",{class:!0});var jo=F(ae);p(Ye.$$.fragment,jo),bs=n(jo),Ut=i(jo,"P",{"data-svelte-h":!0}),_(Ut)!=="svelte-pw2uru"&&(Ut.innerHTML=mn),jo.forEach(o),J.forEach(o),fo=n(e),p(Qe.$$.fragment,e),uo=n(e),U=i(e,"DIV",{class:!0});var H=F(U);p(Ke.$$.fragment,H),ks=n(H),Ct=i(H,"P",{"data-svelte-h":!0}),_(Ct)!=="svelte-1e0z7at"&&(Ct.textContent=pn),Ms=n(H),Zt=i(H,"P",{"data-svelte-h":!0}),_(Zt)!=="svelte-q52n56"&&(Zt.innerHTML=gn),ys=n(H),Wt=i(H,"P",{"data-svelte-h":!0}),_(Wt)!=="svelte-hswkmf"&&(Wt.innerHTML=hn),Fs=n(H),Z=i(H,"DIV",{class:!0});var pe=F(Z);p(et.$$.fragment,pe),vs=n(pe),Jt=i(pe,"P",{"data-svelte-h":!0}),_(Jt)!=="svelte-6759mi"&&(Jt.innerHTML=fn),ws=n(pe),p(ie.$$.fragment,pe),Ts=n(pe),p(de.$$.fragment,pe),pe.forEach(o),H.forEach(o),_o=n(e),p(tt.$$.fragment,e),bo=n(e),X=i(e,"DIV",{class:!0});var No=F(X);p(ot.$$.fragment,No),xs=n(No),P=i(No,"DIV",{class:!0});var R=F(P);p(st.$$.fragment,R),Is=n(R),Rt=i(R,"P",{"data-svelte-h":!0}),_(Rt)!=="svelte-1khd5xe"&&(Rt.innerHTML=un),$s=n(R),p(ce.$$.fragment,R),zs=n(R),Lt=i(R,"P",{"data-svelte-h":!0}),_(Lt)!=="svelte-kvfsh7"&&(Lt.textContent=_n),js=n(R),p(le.$$.fragment,R),Ns=n(R),p(me.$$.fragment,R),R.forEach(o),No.forEach(o),ko=n(e),p(nt.$$.fragment,e),Mo=n(e),St=i(e,"P",{}),F(St).forEach(o),this.h()},h(){M(d,"name","hf:doc:metadata"),M(d,"content",Wn),M(G,"class","flex flex-wrap space-x-1"),M(ke,"width","600"),yn(ke.src,Rs="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/maskformer_architecture.png")||M(ke,"src",Rs),M(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(_t,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(wt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M($t,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,r){t(document.head,d),c(e,v,r),c(e,b,r),c(e,k,r),c(e,y,r),c(e,x,r),g(ge,e,r),c(e,Ht,r),c(e,G,r),c(e,Ot,r),g(D,e,r),c(e,Xt,r),g(he,e,r),c(e,Vt,r),c(e,fe,r),c(e,qt,r),c(e,ue,r),c(e,Gt,r),c(e,_e,r),c(e,Dt,r),c(e,be,r),c(e,At,r),c(e,ke,r),c(e,Yt,r),c(e,Me,r),c(e,Qt,r),g(ye,e,r),c(e,Kt,r),c(e,Fe,r),c(e,eo,r),g(ve,e,r),c(e,to,r),g(we,e,r),c(e,oo,r),c(e,Te,r),c(e,so,r),g(xe,e,r),c(e,no,r),c(e,O,r),g(Ie,O,null),t(O,Co),t(O,dt),c(e,ro,r),c(e,L,r),g($e,L,null),t(L,Zo),t(L,ct),t(L,Wo),t(L,lt),c(e,ao,r),g(ze,e,r),c(e,io,r),c(e,$,r),g(je,$,null),t($,Jo),t($,mt),t($,Ro),t($,pt),t($,Lo),t($,gt),t($,Bo),g(A,$,null),t($,So),t($,Y),g(Ne,Y,null),t(Y,Eo),t(Y,ht),c(e,co,r),g(Pe,e,r),c(e,lo,r),c(e,T,r),g(Ue,T,null),t(T,Ho),t(T,ft),t(T,Oo),t(T,ut),t(T,Xo),t(T,_t),g(Ce,_t,null),t(T,Vo),t(T,S),g(Ze,S,null),t(S,qo),t(S,bt),t(S,Go),t(S,kt),t(T,Do),t(T,Q),g(We,Q,null),t(Q,Ao),t(Q,Mt),t(T,Yo),t(T,K),g(Je,K,null),t(K,Qo),t(K,yt),t(T,Ko),t(T,ee),g(Re,ee,null),t(ee,es),t(ee,Ft),c(e,mo,r),g(Le,e,r),c(e,po,r),c(e,z,r),g(Be,z,null),t(z,ts),t(z,vt),t(z,os),t(z,wt),g(Se,wt,null),t(z,ss),t(z,te),g(Ee,te,null),t(te,ns),t(te,Tt),t(z,rs),t(z,oe),g(He,oe,null),t(oe,as),t(oe,xt),t(z,is),t(z,se),g(Oe,se,null),t(se,ds),t(se,It),c(e,go,r),g(Xe,e,r),c(e,ho,r),c(e,j,r),g(Ve,j,null),t(j,cs),t(j,$t),g(qe,$t,null),t(j,ls),t(j,E),g(Ge,E,null),t(E,ms),t(E,zt),t(E,ps),t(E,jt),t(j,gs),t(j,ne),g(De,ne,null),t(ne,hs),t(ne,Nt),t(j,fs),t(j,re),g(Ae,re,null),t(re,us),t(re,Pt),t(j,_s),t(j,ae),g(Ye,ae,null),t(ae,bs),t(ae,Ut),c(e,fo,r),g(Qe,e,r),c(e,uo,r),c(e,U,r),g(Ke,U,null),t(U,ks),t(U,Ct),t(U,Ms),t(U,Zt),t(U,ys),t(U,Wt),t(U,Fs),t(U,Z),g(et,Z,null),t(Z,vs),t(Z,Jt),t(Z,ws),g(ie,Z,null),t(Z,Ts),g(de,Z,null),c(e,_o,r),g(tt,e,r),c(e,bo,r),c(e,X,r),g(ot,X,null),t(X,xs),t(X,P),g(st,P,null),t(P,Is),t(P,Rt),t(P,$s),g(ce,P,null),t(P,zs),t(P,Lt),t(P,js),g(le,P,null),t(P,Ns),g(me,P,null),c(e,ko,r),g(nt,e,r),c(e,Mo,r),c(e,St,r),yo=!0},p(e,[r]){const rt={};r&2&&(rt.$$scope={dirty:r,ctx:e}),D.$set(rt);const V={};r&2&&(V.$$scope={dirty:r,ctx:e}),A.$set(V);const C={};r&2&&(C.$$scope={dirty:r,ctx:e}),ie.$set(C);const at={};r&2&&(at.$$scope={dirty:r,ctx:e}),de.$set(at);const I={};r&2&&(I.$$scope={dirty:r,ctx:e}),ce.$set(I);const Et={};r&2&&(Et.$$scope={dirty:r,ctx:e}),le.$set(Et);const q={};r&2&&(q.$$scope={dirty:r,ctx:e}),me.$set(q)},i(e){yo||(h(ge.$$.fragment,e),h(D.$$.fragment,e),h(he.$$.fragment,e),h(ye.$$.fragment,e),h(ve.$$.fragment,e),h(we.$$.fragment,e),h(xe.$$.fragment,e),h(Ie.$$.fragment,e),h($e.$$.fragment,e),h(ze.$$.fragment,e),h(je.$$.fragment,e),h(A.$$.fragment,e),h(Ne.$$.fragment,e),h(Pe.$$.fragment,e),h(Ue.$$.fragment,e),h(Ce.$$.fragment,e),h(Ze.$$.fragment,e),h(We.$$.fragment,e),h(Je.$$.fragment,e),h(Re.$$.fragment,e),h(Le.$$.fragment,e),h(Be.$$.fragment,e),h(Se.$$.fragment,e),h(Ee.$$.fragment,e),h(He.$$.fragment,e),h(Oe.$$.fragment,e),h(Xe.$$.fragment,e),h(Ve.$$.fragment,e),h(qe.$$.fragment,e),h(Ge.$$.fragment,e),h(De.$$.fragment,e),h(Ae.$$.fragment,e),h(Ye.$$.fragment,e),h(Qe.$$.fragment,e),h(Ke.$$.fragment,e),h(et.$$.fragment,e),h(ie.$$.fragment,e),h(de.$$.fragment,e),h(tt.$$.fragment,e),h(ot.$$.fragment,e),h(st.$$.fragment,e),h(ce.$$.fragment,e),h(le.$$.fragment,e),h(me.$$.fragment,e),h(nt.$$.fragment,e),yo=!0)},o(e){f(ge.$$.fragment,e),f(D.$$.fragment,e),f(he.$$.fragment,e),f(ye.$$.fragment,e),f(ve.$$.fragment,e),f(we.$$.fragment,e),f(xe.$$.fragment,e),f(Ie.$$.fragment,e),f($e.$$.fragment,e),f(ze.$$.fragment,e),f(je.$$.fragment,e),f(A.$$.fragment,e),f(Ne.$$.fragment,e),f(Pe.$$.fragment,e),f(Ue.$$.fragment,e),f(Ce.$$.fragment,e),f(Ze.$$.fragment,e),f(We.$$.fragment,e),f(Je.$$.fragment,e),f(Re.$$.fragment,e),f(Le.$$.fragment,e),f(Be.$$.fragment,e),f(Se.$$.fragment,e),f(Ee.$$.fragment,e),f(He.$$.fragment,e),f(Oe.$$.fragment,e),f(Xe.$$.fragment,e),f(Ve.$$.fragment,e),f(qe.$$.fragment,e),f(Ge.$$.fragment,e),f(De.$$.fragment,e),f(Ae.$$.fragment,e),f(Ye.$$.fragment,e),f(Qe.$$.fragment,e),f(Ke.$$.fragment,e),f(et.$$.fragment,e),f(ie.$$.fragment,e),f(de.$$.fragment,e),f(tt.$$.fragment,e),f(ot.$$.fragment,e),f(st.$$.fragment,e),f(ce.$$.fragment,e),f(le.$$.fragment,e),f(me.$$.fragment,e),f(nt.$$.fragment,e),yo=!1},d(e){e&&(o(v),o(b),o(k),o(y),o(x),o(Ht),o(G),o(Ot),o(Xt),o(Vt),o(fe),o(qt),o(ue),o(Gt),o(_e),o(Dt),o(be),o(At),o(ke),o(Yt),o(Me),o(Qt),o(Kt),o(Fe),o(eo),o(to),o(oo),o(Te),o(so),o(no),o(O),o(ro),o(L),o(ao),o(io),o($),o(co),o(lo),o(T),o(mo),o(po),o(z),o(go),o(ho),o(j),o(fo),o(uo),o(U),o(_o),o(bo),o(X),o(ko),o(Mo),o(St)),o(d),u(ge,e),u(D,e),u(he,e),u(ye,e),u(ve,e),u(we,e),u(xe,e),u(Ie),u($e),u(ze,e),u(je),u(A),u(Ne),u(Pe,e),u(Ue),u(Ce),u(Ze),u(We),u(Je),u(Re),u(Le,e),u(Be),u(Se),u(Ee),u(He),u(Oe),u(Xe,e),u(Ve),u(qe),u(Ge),u(De),u(Ae),u(Ye),u(Qe,e),u(Ke),u(et),u(ie),u(de),u(tt,e),u(ot),u(st),u(ce),u(le),u(me),u(nt,e)}}}const Wn='{"title":"MaskFormer","local":"maskformer","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"MaskFormer specific outputs","local":"transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput","sections":[],"depth":2},{"title":"MaskFormerConfig","local":"transformers.MaskFormerConfig","sections":[],"depth":2},{"title":"MaskFormerImageProcessor","local":"transformers.MaskFormerImageProcessor","sections":[],"depth":2},{"title":"MaskFormerImageProcessorFast","local":"transformers.MaskFormerImageProcessorFast","sections":[],"depth":2},{"title":"MaskFormerFeatureExtractor","local":"transformers.MaskFormerFeatureExtractor","sections":[],"depth":2},{"title":"MaskFormerModel","local":"transformers.MaskFormerModel","sections":[],"depth":2},{"title":"MaskFormerForInstanceSegmentation","local":"transformers.MaskFormerForInstanceSegmentation","sections":[],"depth":2}],"depth":1}';function Jn(N){return Fn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Vn extends vn{constructor(d){super(),wn(this,d,Jn,Zn,Mn,{})}}export{Vn as component};
