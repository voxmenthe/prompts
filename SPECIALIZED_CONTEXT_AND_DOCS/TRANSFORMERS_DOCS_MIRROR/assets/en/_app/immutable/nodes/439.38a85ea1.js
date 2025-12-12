import{s as vo,o as wo,n as I}from"../chunks/scheduler.18a86fab.js";import{S as Jo,i as ko,g as p,s as i,r as h,A as Bo,h as u,f as a,c as l,j as W,x as y,u as g,k as C,l as Uo,y as d,a as c,v as f,d as _,t as M,w as b}from"../chunks/index.98837b22.js";import{T as Qe}from"../chunks/Tip.77304350.js";import{D as N}from"../chunks/Docstring.a1ef7999.js";import{C as _e}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ye}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ee,E as jo}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Vo,a as Wo}from"../chunks/HfOption.6641485e.js";function Co(v){let t,m=`This model was contributed by <a href="https://huggingface.co/gchhablani" rel="nofollow">gchhablani</a>.
Click on the VisualBERT models in the right sidebar for more examples of how to apply VisualBERT to different image and language tasks.`;return{c(){t=p("p"),t.innerHTML=m},l(o){t=u(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-1xxojws"&&(t.innerHTML=m)},m(o,r){c(o,t,r)},p:I,d(o){o&&a(t)}}}function Io(v){let t,m;return t=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFpbXBvcnQlMjB0b3JjaHZpc2lvbiUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWltcG9ydCUyMG51bXB5JTIwYXMlMjBucCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBBdXRvVG9rZW5pemVyJTJDJTIwVmlzdWFsQmVydEZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwcmVxdWVzdHMlMEFmcm9tJTIwaW8lMjBpbXBvcnQlMjBCeXRlc0lPJTBBJTBBZGVmJTIwZ2V0X3Zpc3VhbF9lbWJlZGRpbmdzX3NpbXBsZShpbWFnZSUyQyUyMGRldmljZSUzRE5vbmUpJTNBJTBBJTIwJTIwJTIwJTIwJTBBJTIwJTIwJTIwJTIwbW9kZWwlMjAlM0QlMjB0b3JjaHZpc2lvbi5tb2RlbHMucmVzbmV0NTAocHJldHJhaW5lZCUzRFRydWUpJTBBJTIwJTIwJTIwJTIwbW9kZWwlMjAlM0QlMjB0b3JjaC5ubi5TZXF1ZW50aWFsKCpsaXN0KG1vZGVsLmNoaWxkcmVuKCkpJTVCJTNBLTElNUQpJTBBJTIwJTIwJTIwJTIwbW9kZWwudG8oZGV2aWNlKSUwQSUyMCUyMCUyMCUyMG1vZGVsLmV2YWwoKSUwQSUyMCUyMCUyMCUyMCUwQSUyMCUyMCUyMCUyMHRyYW5zZm9ybSUyMCUzRCUyMHRvcmNodmlzaW9uLnRyYW5zZm9ybXMuQ29tcG9zZSglNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjB0b3JjaHZpc2lvbi50cmFuc2Zvcm1zLlJlc2l6ZSgyNTYpJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwdG9yY2h2aXNpb24udHJhbnNmb3Jtcy5DZW50ZXJDcm9wKDIyNCklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjB0b3JjaHZpc2lvbi50cmFuc2Zvcm1zLlRvVGVuc29yKCklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjB0b3JjaHZpc2lvbi50cmFuc2Zvcm1zLk5vcm1hbGl6ZSglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBtZWFuJTNEJTVCMC40ODUlMkMlMjAwLjQ1NiUyQyUyMDAuNDA2JTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwc3RkJTNEJTVCMC4yMjklMkMlMjAwLjIyNCUyQyUyMDAuMjI1JTVEJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwKSUwQSUyMCUyMCUyMCUyMCU1RCklMEElMjAlMjAlMjAlMjAlMEElMjAlMjAlMjAlMjBpZiUyMGlzaW5zdGFuY2UoaW1hZ2UlMkMlMjBzdHIpJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwaW1hZ2UlMjAlM0QlMjBJbWFnZS5vcGVuKGltYWdlKS5jb252ZXJ0KCdSR0InKSUwQSUyMCUyMCUyMCUyMGVsaWYlMjBpc2luc3RhbmNlKGltYWdlJTJDJTIwSW1hZ2UuSW1hZ2UpJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwaW1hZ2UlMjAlM0QlMjBpbWFnZS5jb252ZXJ0KCdSR0InKSUwQSUyMCUyMCUyMCUyMGVsc2UlM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjByYWlzZSUyMFZhbHVlRXJyb3IoJTIySW1hZ2UlMjBtdXN0JTIwYmUlMjBhJTIwUElMJTIwSW1hZ2UlMjBvciUyMHBhdGglMjB0byUyMGltYWdlJTIwZmlsZSUyMiklMEElMjAlMjAlMjAlMjAlMEElMjAlMjAlMjAlMjBpbWFnZV90ZW5zb3IlMjAlM0QlMjB0cmFuc2Zvcm0oaW1hZ2UpLnVuc3F1ZWV6ZSgwKS50byhkZXZpY2UpJTBBJTIwJTIwJTIwJTIwJTBBJTIwJTIwJTIwJTIwd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGZlYXR1cmVzJTIwJTNEJTIwbW9kZWwoaW1hZ2VfdGVuc29yKSUwQSUyMCUyMCUyMCUyMCUwQSUyMCUyMCUyMCUyMGJhdGNoX3NpemUlMjAlM0QlMjBmZWF0dXJlcy5zaGFwZSU1QjAlNUQlMEElMjAlMjAlMjAlMjBmZWF0dXJlX2RpbSUyMCUzRCUyMGZlYXR1cmVzLnNoYXBlJTVCMSU1RCUwQSUyMCUyMCUyMCUyMHZpc3VhbF9zZXFfbGVuZ3RoJTIwJTNEJTIwMTAlMEElMjAlMjAlMjAlMjAlMEElMjAlMjAlMjAlMjB2aXN1YWxfZW1iZWRzJTIwJTNEJTIwZmVhdHVyZXMuc3F1ZWV6ZSgtMSkuc3F1ZWV6ZSgtMSkudW5zcXVlZXplKDEpLmV4cGFuZChiYXRjaF9zaXplJTJDJTIwdmlzdWFsX3NlcV9sZW5ndGglMkMlMjBmZWF0dXJlX2RpbSklMEElMjAlMjAlMjAlMjAlMEElMjAlMjAlMjAlMjByZXR1cm4lMjB2aXN1YWxfZW1iZWRzJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFZpc3VhbEJlcnRGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIydWNsYW5scCUyRnZpc3VhbGJlcnQtdnFhLWNvY28tcHJlJTIyKSUwQSUwQXJlc3BvbnNlJTIwJTNEJTIwcmVxdWVzdHMuZ2V0KCUyMmh0dHBzJTNBJTJGJTJGaHVnZ2luZ2ZhY2UuY28lMkZkYXRhc2V0cyUyRmh1Z2dpbmdmYWNlJTJGZG9jdW1lbnRhdGlvbi1pbWFnZXMlMkZyZXNvbHZlJTJGbWFpbiUyRnBpcGVsaW5lLWNhdC1jaG9uay5qcGVnJTIyKSUwQWltYWdlJTIwJTNEJTIwSW1hZ2Uub3BlbihCeXRlc0lPKHJlc3BvbnNlLmNvbnRlbnQpKSUwQSUyMCUyMCUyMCUyMCUwQXZpc3VhbF9lbWJlZHMlMjAlM0QlMjBnZXRfdmlzdWFsX2VtYmVkZGluZ3Nfc2ltcGxlKGltYWdlKSUwQSUyMCUyMCUyMCUyMCUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJXaGF0JTIwaXMlMjBzaG93biUyMGluJTIwdGhpcyUyMGltYWdlJTNGJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMjAlMjAlMjAlMjAlMEF2aXN1YWxfdG9rZW5fdHlwZV9pZHMlMjAlM0QlMjB0b3JjaC5vbmVzKHZpc3VhbF9lbWJlZHMuc2hhcGUlNUIlM0EtMSU1RCUyQyUyMGR0eXBlJTNEdG9yY2gubG9uZyklMEF2aXN1YWxfYXR0ZW50aW9uX21hc2slMjAlM0QlMjB0b3JjaC5vbmVzKHZpc3VhbF9lbWJlZHMuc2hhcGUlNUIlM0EtMSU1RCUyQyUyMGR0eXBlJTNEdG9yY2guZmxvYXQpJTBBJTIwJTIwJTIwJTIwJTBBaW5wdXRzLnVwZGF0ZSglN0IlMEElMjAlMjAlMjAlMjAlMjJ2aXN1YWxfZW1iZWRzJTIyJTNBJTIwdmlzdWFsX2VtYmVkcyUyQyUwQSUyMCUyMCUyMCUyMCUyMnZpc3VhbF90b2tlbl90eXBlX2lkcyUyMiUzQSUyMHZpc3VhbF90b2tlbl90eXBlX2lkcyUyQyUwQSUyMCUyMCUyMCUyMCUyMnZpc3VhbF9hdHRlbnRpb25fbWFzayUyMiUzQSUyMHZpc3VhbF9hdHRlbnRpb25fbWFzayUyQyUwQSU3RCklMEElMjAlMjAlMjAlMjAlMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTIwJTIwJTIwJTIwcHJlZGljdGVkX2Fuc3dlcl9pZHglMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKS5pdGVtKCklMEElMEFwcmludChmJTIyUHJlZGljdGVkJTIwYW5zd2VyJTNBJTIwJTdCcHJlZGljdGVkX2Fuc3dlcl9pZHglN0QlMjIp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">import</span> torchvision
<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, VisualBertForQuestionAnswering
<span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">from</span> io <span class="hljs-keyword">import</span> BytesIO

<span class="hljs-keyword">def</span> <span class="hljs-title function_">get_visual_embeddings_simple</span>(<span class="hljs-params">image, device=<span class="hljs-literal">None</span></span>):
    
    model = torchvision.models.resnet50(pretrained=<span class="hljs-literal">True</span>)
    model = torch.nn.Sequential(*<span class="hljs-built_in">list</span>(model.children())[:-<span class="hljs-number">1</span>])
    model.to(device)
    model.<span class="hljs-built_in">eval</span>()
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(<span class="hljs-number">256</span>),
        torchvision.transforms.CenterCrop(<span class="hljs-number">224</span>),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[<span class="hljs-number">0.485</span>, <span class="hljs-number">0.456</span>, <span class="hljs-number">0.406</span>],
            std=[<span class="hljs-number">0.229</span>, <span class="hljs-number">0.224</span>, <span class="hljs-number">0.225</span>]
        )
    ])
    
    <span class="hljs-keyword">if</span> <span class="hljs-built_in">isinstance</span>(image, <span class="hljs-built_in">str</span>):
        image = Image.<span class="hljs-built_in">open</span>(image).convert(<span class="hljs-string">&#x27;RGB&#x27;</span>)
    <span class="hljs-keyword">elif</span> <span class="hljs-built_in">isinstance</span>(image, Image.Image):
        image = image.convert(<span class="hljs-string">&#x27;RGB&#x27;</span>)
    <span class="hljs-keyword">else</span>:
        <span class="hljs-keyword">raise</span> ValueError(<span class="hljs-string">&quot;Image must be a PIL Image or path to image file&quot;</span>)
    
    image_tensor = transform(image).unsqueeze(<span class="hljs-number">0</span>).to(device)
    
    <span class="hljs-keyword">with</span> torch.no_grad():
        features = model(image_tensor)
    
    batch_size = features.shape[<span class="hljs-number">0</span>]
    feature_dim = features.shape[<span class="hljs-number">1</span>]
    visual_seq_length = <span class="hljs-number">10</span>
    
    visual_embeds = features.squeeze(-<span class="hljs-number">1</span>).squeeze(-<span class="hljs-number">1</span>).unsqueeze(<span class="hljs-number">1</span>).expand(batch_size, visual_seq_length, feature_dim)
    
    <span class="hljs-keyword">return</span> visual_embeds

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = VisualBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;uclanlp/visualbert-vqa-coco-pre&quot;</span>)

response = requests.get(<span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>)
image = Image.<span class="hljs-built_in">open</span>(BytesIO(response.content))
    
visual_embeds = get_visual_embeddings_simple(image)
    
inputs = tokenizer(<span class="hljs-string">&quot;What is shown in this image?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
    
visual_token_type_ids = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.<span class="hljs-built_in">float</span>)
    
inputs.update({
    <span class="hljs-string">&quot;visual_embeds&quot;</span>: visual_embeds,
    <span class="hljs-string">&quot;visual_token_type_ids&quot;</span>: visual_token_type_ids,
    <span class="hljs-string">&quot;visual_attention_mask&quot;</span>: visual_attention_mask,
})
    
<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_answer_idx = logits.argmax(-<span class="hljs-number">1</span>).item()

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Predicted answer: <span class="hljs-subst">{predicted_answer_idx}</span>&quot;</span>)`,wrap:!1}}),{c(){h(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,r){f(t,o,r),m=!0},p:I,i(o){m||(_(t.$$.fragment,o),m=!0)},o(o){M(t.$$.fragment,o),m=!1},d(o){b(t,o)}}}function zo(v){let t,m;return t=new Wo({props:{id:"usage",option:"AutoModel",$$slots:{default:[Io]},$$scope:{ctx:v}}}),{c(){h(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,r){f(t,o,r),m=!0},p(o,r){const T={};r&2&&(T.$$scope={dirty:r,ctx:o}),t.$set(T)},i(o){m||(_(t.$$.fragment,o),m=!0)},o(o){M(t.$$.fragment,o),m=!1},d(o){b(t,o)}}}function Fo(v){let t,m="Example:",o,r,T;return r=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFZpc3VhbEJlcnRDb25maWclMkMlMjBWaXN1YWxCZXJ0TW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwVmlzdWFsQkVSVCUyMHZpc3VhbGJlcnQtdnFhLWNvY28tcHJlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFZpc3VhbEJlcnRDb25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMnVjbGFubHAlMkZ2aXN1YWxiZXJ0LXZxYS1jb2NvLXByZSUyMiklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMHZpc3VhbGJlcnQtdnFhLWNvY28tcHJlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBWaXN1YWxCZXJ0TW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> VisualBertConfig, VisualBertModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a VisualBERT visualbert-vqa-coco-pre style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = VisualBertConfig.from_pretrained(<span class="hljs-string">&quot;uclanlp/visualbert-vqa-coco-pre&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the visualbert-vqa-coco-pre style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = VisualBertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=m,o=i(),h(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=m),o=l(n),g(r.$$.fragment,n)},m(n,w){c(n,t,w),c(n,o,w),f(r,n,w),T=!0},p:I,i(n){T||(_(r.$$.fragment,n),T=!0)},o(n){M(r.$$.fragment,n),T=!1},d(n){n&&(a(t),a(o)),b(r,n)}}}function Zo(v){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=m},l(o){t=u(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,r){c(o,t,r)},p:I,d(o){o&&a(t)}}}function $o(v){let t,m="Example:",o,r,T;return r=new _e({props:{code:"JTIzJTIwQXNzdW1wdGlvbiUzQSUyMCpnZXRfdmlzdWFsX2VtYmVkZGluZ3MoaW1hZ2UpKiUyMGdldHMlMjB0aGUlMjB2aXN1YWwlMjBlbWJlZGRpbmdzJTIwb2YlMjB0aGUlMjBpbWFnZS4lMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFZpc3VhbEJlcnRNb2RlbCUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFZpc3VhbEJlcnRNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIydWNsYW5scCUyRnZpc3VhbGJlcnQtdnFhLWNvY28tcHJlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBdmlzdWFsX2VtYmVkcyUyMCUzRCUyMGdldF92aXN1YWxfZW1iZWRkaW5ncyhpbWFnZSkudW5zcXVlZXplKDApJTBBdmlzdWFsX3Rva2VuX3R5cGVfaWRzJTIwJTNEJTIwdG9yY2gub25lcyh2aXN1YWxfZW1iZWRzLnNoYXBlJTVCJTNBLTElNUQlMkMlMjBkdHlwZSUzRHRvcmNoLmxvbmcpJTBBdmlzdWFsX2F0dGVudGlvbl9tYXNrJTIwJTNEJTIwdG9yY2gub25lcyh2aXN1YWxfZW1iZWRzLnNoYXBlJTVCJTNBLTElNUQlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0KSUwQSUwQWlucHV0cy51cGRhdGUoJTBBJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydmlzdWFsX2VtYmVkcyUyMiUzQSUyMHZpc3VhbF9lbWJlZHMlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ2aXN1YWxfdG9rZW5fdHlwZV9pZHMlMjIlM0ElMjB2aXN1YWxfdG9rZW5fdHlwZV9pZHMlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ2aXN1YWxfYXR0ZW50aW9uX21hc2slMjIlM0ElMjB2aXN1YWxfYXR0ZW50aW9uX21hc2slMkMlMEElMjAlMjAlMjAlMjAlN0QlMEEpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-comment"># Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.</span>
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, VisualBertModel
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = VisualBertModel.from_pretrained(<span class="hljs-string">&quot;uclanlp/visualbert-vqa-coco-pre&quot;</span>)

inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
visual_embeds = get_visual_embeddings(image).unsqueeze(<span class="hljs-number">0</span>)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.<span class="hljs-built_in">float</span>)

inputs.update(
    {
        <span class="hljs-string">&quot;visual_embeds&quot;</span>: visual_embeds,
        <span class="hljs-string">&quot;visual_token_type_ids&quot;</span>: visual_token_type_ids,
        <span class="hljs-string">&quot;visual_attention_mask&quot;</span>: visual_attention_mask,
    }
)

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=p("p"),t.textContent=m,o=i(),h(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=m),o=l(n),g(r.$$.fragment,n)},m(n,w){c(n,t,w),c(n,o,w),f(r,n,w),T=!0},p:I,i(n){T||(_(r.$$.fragment,n),T=!0)},o(n){M(r.$$.fragment,n),T=!1},d(n){n&&(a(t),a(o)),b(r,n)}}}function xo(v){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=m},l(o){t=u(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,r){c(o,t,r)},p:I,d(o){o&&a(t)}}}function Ro(v){let t,m="Example:",o,r,T;return r=new _e({props:{code:"JTIzJTIwQXNzdW1wdGlvbiUzQSUyMCpnZXRfdmlzdWFsX2VtYmVkZGluZ3MoaW1hZ2UpKiUyMGdldHMlMjB0aGUlMjB2aXN1YWwlMjBlbWJlZGRpbmdzJTIwb2YlMjB0aGUlMjBpbWFnZSUyMGluJTIwdGhlJTIwYmF0Y2guJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBWaXN1YWxCZXJ0Rm9yUHJlVHJhaW5pbmclMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwVmlzdWFsQmVydEZvclByZVRyYWluaW5nLmZyb21fcHJldHJhaW5lZCglMjJ1Y2xhbmxwJTJGdmlzdWFsYmVydC12cWEtY29jby1wcmUlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTVCTUFTSyU1RC4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXZpc3VhbF9lbWJlZHMlMjAlM0QlMjBnZXRfdmlzdWFsX2VtYmVkZGluZ3MoaW1hZ2UpLnVuc3F1ZWV6ZSgwKSUwQXZpc3VhbF90b2tlbl90eXBlX2lkcyUyMCUzRCUyMHRvcmNoLm9uZXModmlzdWFsX2VtYmVkcy5zaGFwZSU1QiUzQS0xJTVEJTJDJTIwZHR5cGUlM0R0b3JjaC5sb25nKSUwQXZpc3VhbF9hdHRlbnRpb25fbWFzayUyMCUzRCUyMHRvcmNoLm9uZXModmlzdWFsX2VtYmVkcy5zaGFwZSU1QiUzQS0xJTVEJTJDJTIwZHR5cGUlM0R0b3JjaC5mbG9hdCklMEElMEFpbnB1dHMudXBkYXRlKCUwQSUyMCUyMCUyMCUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnZpc3VhbF9lbWJlZHMlMjIlM0ElMjB2aXN1YWxfZW1iZWRzJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydmlzdWFsX3Rva2VuX3R5cGVfaWRzJTIyJTNBJTIwdmlzdWFsX3Rva2VuX3R5cGVfaWRzJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydmlzdWFsX2F0dGVudGlvbl9tYXNrJTIyJTNBJTIwdmlzdWFsX2F0dGVudGlvbl9tYXNrJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTBBKSUwQW1heF9sZW5ndGglMjAlM0QlMjBpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQuc2hhcGUlNUItMSU1RCUyMCUyQiUyMHZpc3VhbF9lbWJlZHMuc2hhcGUlNUItMiU1RCUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEJTIybWF4X2xlbmd0aCUyMiUyQyUyMG1heF9sZW5ndGglM0RtYXhfbGVuZ3RoJTBBKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQXNlbnRlbmNlX2ltYWdlX2xhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigxKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBCYXRjaF9zaXplJTBBJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzJTJDJTIwc2VudGVuY2VfaW1hZ2VfbGFiZWxzJTNEc2VudGVuY2VfaW1hZ2VfbGFiZWxzKSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMucHJlZGljdGlvbl9sb2dpdHMlMEFzZXFfcmVsYXRpb25zaGlwX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMuc2VxX3JlbGF0aW9uc2hpcF9sb2dpdHM=",highlighted:`<span class="hljs-comment"># Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.</span>
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, VisualBertForPreTraining

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = VisualBertForPreTraining.from_pretrained(<span class="hljs-string">&quot;uclanlp/visualbert-vqa-coco-pre&quot;</span>)

inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
visual_embeds = get_visual_embeddings(image).unsqueeze(<span class="hljs-number">0</span>)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.<span class="hljs-built_in">float</span>)

inputs.update(
    {
        <span class="hljs-string">&quot;visual_embeds&quot;</span>: visual_embeds,
        <span class="hljs-string">&quot;visual_token_type_ids&quot;</span>: visual_token_type_ids,
        <span class="hljs-string">&quot;visual_attention_mask&quot;</span>: visual_attention_mask,
    }
)
max_length = inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape[-<span class="hljs-number">1</span>] + visual_embeds.shape[-<span class="hljs-number">2</span>]
labels = tokenizer(
    <span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-string">&quot;max_length&quot;</span>, max_length=max_length
)[<span class="hljs-string">&quot;input_ids&quot;</span>]
sentence_image_labels = torch.tensor(<span class="hljs-number">1</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># Batch_size</span>


outputs = model(**inputs, labels=labels, sentence_image_labels=sentence_image_labels)
loss = outputs.loss
prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=m,o=i(),h(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=m),o=l(n),g(r.$$.fragment,n)},m(n,w){c(n,t,w),c(n,o,w),f(r,n,w),T=!0},p:I,i(n){T||(_(r.$$.fragment,n),T=!0)},o(n){M(r.$$.fragment,n),T=!1},d(n){n&&(a(t),a(o)),b(r,n)}}}function Xo(v){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=m},l(o){t=u(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,r){c(o,t,r)},p:I,d(o){o&&a(t)}}}function No(v){let t,m="Example:",o,r,T;return r=new _e({props:{code:"JTIzJTIwQXNzdW1wdGlvbiUzQSUyMCpnZXRfdmlzdWFsX2VtYmVkZGluZ3MoaW1hZ2UpKiUyMGdldHMlMjB0aGUlMjB2aXN1YWwlMjBlbWJlZGRpbmdzJTIwb2YlMjB0aGUlMjBpbWFnZSUyMGluJTIwdGhlJTIwYmF0Y2guJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBWaXN1YWxCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBWaXN1YWxCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMnVjbGFubHAlMkZ2aXN1YWxiZXJ0LXZxYSUyMiklMEElMEF0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwaXMlMjBlYXRpbmclMjB0aGUlMjBhcHBsZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcih0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF2aXN1YWxfZW1iZWRzJTIwJTNEJTIwZ2V0X3Zpc3VhbF9lbWJlZGRpbmdzKGltYWdlKS51bnNxdWVlemUoMCklMEF2aXN1YWxfdG9rZW5fdHlwZV9pZHMlMjAlM0QlMjB0b3JjaC5vbmVzKHZpc3VhbF9lbWJlZHMuc2hhcGUlNUIlM0EtMSU1RCUyQyUyMGR0eXBlJTNEdG9yY2gubG9uZyklMEF2aXN1YWxfYXR0ZW50aW9uX21hc2slMjAlM0QlMjB0b3JjaC5vbmVzKHZpc3VhbF9lbWJlZHMuc2hhcGUlNUIlM0EtMSU1RCUyQyUyMGR0eXBlJTNEdG9yY2guZmxvYXQpJTBBJTBBaW5wdXRzLnVwZGF0ZSglMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ2aXN1YWxfZW1iZWRzJTIyJTNBJTIwdmlzdWFsX2VtYmVkcyUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnZpc3VhbF90b2tlbl90eXBlX2lkcyUyMiUzQSUyMHZpc3VhbF90b2tlbl90eXBlX2lkcyUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnZpc3VhbF9hdHRlbnRpb25fbWFzayUyMiUzQSUyMHZpc3VhbF9hdHRlbnRpb25fbWFzayUyQyUwQSUyMCUyMCUyMCUyMCU3RCUwQSklMEElMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCJTVCMC4wJTJDJTIwMS4wJTVEJTVEKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTJDJTIwTnVtJTIwbGFiZWxzJTIwMiUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBc2NvcmVzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-comment"># Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.</span>
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, VisualBertForQuestionAnswering
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = VisualBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;uclanlp/visualbert-vqa&quot;</span>)

text = <span class="hljs-string">&quot;Who is eating the apple?&quot;</span>
inputs = tokenizer(text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
visual_embeds = get_visual_embeddings(image).unsqueeze(<span class="hljs-number">0</span>)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.<span class="hljs-built_in">float</span>)

inputs.update(
    {
        <span class="hljs-string">&quot;visual_embeds&quot;</span>: visual_embeds,
        <span class="hljs-string">&quot;visual_token_type_ids&quot;</span>: visual_token_type_ids,
        <span class="hljs-string">&quot;visual_attention_mask&quot;</span>: visual_attention_mask,
    }
)

labels = torch.tensor([[<span class="hljs-number">0.0</span>, <span class="hljs-number">1.0</span>]]).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># Batch size 1, Num labels 2</span>

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=m,o=i(),h(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=m),o=l(n),g(r.$$.fragment,n)},m(n,w){c(n,t,w),c(n,o,w),f(r,n,w),T=!0},p:I,i(n){T||(_(r.$$.fragment,n),T=!0)},o(n){M(r.$$.fragment,n),T=!1},d(n){n&&(a(t),a(o)),b(r,n)}}}function Ao(v){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=m},l(o){t=u(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,r){c(o,t,r)},p:I,d(o){o&&a(t)}}}function Go(v){let t,m="Example:",o,r,T;return r=new _e({props:{code:"JTIzJTIwQXNzdW1wdGlvbiUzQSUyMCpnZXRfdmlzdWFsX2VtYmVkZGluZ3MoaW1hZ2UpKiUyMGdldHMlMjB0aGUlMjB2aXN1YWwlMjBlbWJlZGRpbmdzJTIwb2YlMjB0aGUlMjBpbWFnZSUyMGluJTIwdGhlJTIwYmF0Y2guJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBWaXN1YWxCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBWaXN1YWxCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UuZnJvbV9wcmV0cmFpbmVkKCUyMnVjbGFubHAlMkZ2aXN1YWxiZXJ0LXZjciUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBY2hvaWNlMCUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdpdGglMjBhJTIwZm9yayUyMGFuZCUyMGElMjBrbmlmZS4lMjIlMEFjaG9pY2UxJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2hpbGUlMjBoZWxkJTIwaW4lMjB0aGUlMjBoYW5kLiUyMiUwQSUwQXZpc3VhbF9lbWJlZHMlMjAlM0QlMjBnZXRfdmlzdWFsX2VtYmVkZGluZ3MoaW1hZ2UpJTBBJTIzJTIwKGJhdGNoX3NpemUlMkMlMjBudW1fY2hvaWNlcyUyQyUyMHZpc3VhbF9zZXFfbGVuZ3RoJTJDJTIwdmlzdWFsX2VtYmVkZGluZ19kaW0pJTBBdmlzdWFsX2VtYmVkcyUyMCUzRCUyMHZpc3VhbF9lbWJlZHMuZXhwYW5kKDElMkMlMjAyJTJDJTIwKnZpc3VhbF9lbWJlZHMuc2hhcGUpJTBBdmlzdWFsX3Rva2VuX3R5cGVfaWRzJTIwJTNEJTIwdG9yY2gub25lcyh2aXN1YWxfZW1iZWRzLnNoYXBlJTVCJTNBLTElNUQlMkMlMjBkdHlwZSUzRHRvcmNoLmxvbmcpJTBBdmlzdWFsX2F0dGVudGlvbl9tYXNrJTIwJTNEJTIwdG9yY2gub25lcyh2aXN1YWxfZW1iZWRzLnNoYXBlJTVCJTNBLTElNUQlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0KSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigwKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBjaG9pY2UwJTIwaXMlMjBjb3JyZWN0JTIwKGFjY29yZGluZyUyMHRvJTIwV2lraXBlZGlhJTIwJTNCKSklMkMlMjBiYXRjaCUyMHNpemUlMjAxJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTVCJTVCcHJvbXB0JTJDJTIwcHJvbXB0JTVEJTJDJTIwJTVCY2hvaWNlMCUyQyUyMGNob2ljZTElNUQlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBJTIzJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBaW5wdXRzX2RpY3QlMjAlM0QlMjAlN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUwQWlucHV0c19kaWN0LnVwZGF0ZSglMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ2aXN1YWxfZW1iZWRzJTIyJTNBJTIwdmlzdWFsX2VtYmVkcyUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnZpc3VhbF9hdHRlbnRpb25fbWFzayUyMiUzQSUyMHZpc3VhbF9hdHRlbnRpb25fbWFzayUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnZpc3VhbF90b2tlbl90eXBlX2lkcyUyMiUzQSUyMHZpc3VhbF90b2tlbl90eXBlX2lkcyUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmxhYmVscyUyMiUzQSUyMGxhYmVscyUyQyUwQSUyMCUyMCUyMCUyMCU3RCUwQSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHNfZGljdCklMEElMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-comment"># Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.</span>
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, VisualBertForMultipleChoice
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = VisualBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;uclanlp/visualbert-vcr&quot;</span>)

prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>

visual_embeds = get_visual_embeddings(image)
<span class="hljs-comment"># (batch_size, num_choices, visual_seq_length, visual_embedding_dim)</span>
visual_embeds = visual_embeds.expand(<span class="hljs-number">1</span>, <span class="hljs-number">2</span>, *visual_embeds.shape)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.<span class="hljs-built_in">float</span>)

labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-comment"># batch size is 1</span>
inputs_dict = {k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}
inputs_dict.update(
    {
        <span class="hljs-string">&quot;visual_embeds&quot;</span>: visual_embeds,
        <span class="hljs-string">&quot;visual_attention_mask&quot;</span>: visual_attention_mask,
        <span class="hljs-string">&quot;visual_token_type_ids&quot;</span>: visual_token_type_ids,
        <span class="hljs-string">&quot;labels&quot;</span>: labels,
    }
)
outputs = model(**inputs_dict)

loss = outputs.loss
logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=m,o=i(),h(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=m),o=l(n),g(r.$$.fragment,n)},m(n,w){c(n,t,w),c(n,o,w),f(r,n,w),T=!0},p:I,i(n){T||(_(r.$$.fragment,n),T=!0)},o(n){M(r.$$.fragment,n),T=!1},d(n){n&&(a(t),a(o)),b(r,n)}}}function Ho(v){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=m},l(o){t=u(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,r){c(o,t,r)},p:I,d(o){o&&a(t)}}}function qo(v){let t,m="Example:",o,r,T;return r=new _e({props:{code:"JTIzJTIwQXNzdW1wdGlvbiUzQSUyMCpnZXRfdmlzdWFsX2VtYmVkZGluZ3MoaW1hZ2UpKiUyMGdldHMlMjB0aGUlMjB2aXN1YWwlMjBlbWJlZGRpbmdzJTIwb2YlMjB0aGUlMjBpbWFnZSUyMGluJTIwdGhlJTIwYmF0Y2guJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBWaXN1YWxCZXJ0Rm9yVmlzdWFsUmVhc29uaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwVmlzdWFsQmVydEZvclZpc3VhbFJlYXNvbmluZy5mcm9tX3ByZXRyYWluZWQoJTIydWNsYW5scCUyRnZpc3VhbGJlcnQtbmx2cjIlMjIpJTBBJTBBdGV4dCUyMCUzRCUyMCUyMldobyUyMGlzJTIwZWF0aW5nJTIwdGhlJTIwYXBwbGUlM0YlMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIodGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBdmlzdWFsX2VtYmVkcyUyMCUzRCUyMGdldF92aXN1YWxfZW1iZWRkaW5ncyhpbWFnZSkudW5zcXVlZXplKDApJTBBdmlzdWFsX3Rva2VuX3R5cGVfaWRzJTIwJTNEJTIwdG9yY2gub25lcyh2aXN1YWxfZW1iZWRzLnNoYXBlJTVCJTNBLTElNUQlMkMlMjBkdHlwZSUzRHRvcmNoLmxvbmcpJTBBdmlzdWFsX2F0dGVudGlvbl9tYXNrJTIwJTNEJTIwdG9yY2gub25lcyh2aXN1YWxfZW1iZWRzLnNoYXBlJTVCJTNBLTElNUQlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0KSUwQSUwQWlucHV0cy51cGRhdGUoJTBBJTIwJTIwJTIwJTIwJTdCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydmlzdWFsX2VtYmVkcyUyMiUzQSUyMHZpc3VhbF9lbWJlZHMlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ2aXN1YWxfdG9rZW5fdHlwZV9pZHMlMjIlM0ElMjB2aXN1YWxfdG9rZW5fdHlwZV9pZHMlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ2aXN1YWxfYXR0ZW50aW9uX21hc2slMjIlM0ElMjB2aXN1YWxfYXR0ZW50aW9uX21hc2slMkMlMEElMjAlMjAlMjAlMjAlN0QlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDEpLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMkMlMjBOdW0lMjBjaG9pY2VzJTIwMiUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBc2NvcmVzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-comment"># Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.</span>
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, VisualBertForVisualReasoning
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = VisualBertForVisualReasoning.from_pretrained(<span class="hljs-string">&quot;uclanlp/visualbert-nlvr2&quot;</span>)

text = <span class="hljs-string">&quot;Who is eating the apple?&quot;</span>
inputs = tokenizer(text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
visual_embeds = get_visual_embeddings(image).unsqueeze(<span class="hljs-number">0</span>)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.<span class="hljs-built_in">float</span>)

inputs.update(
    {
        <span class="hljs-string">&quot;visual_embeds&quot;</span>: visual_embeds,
        <span class="hljs-string">&quot;visual_token_type_ids&quot;</span>: visual_token_type_ids,
        <span class="hljs-string">&quot;visual_attention_mask&quot;</span>: visual_attention_mask,
    }
)

labels = torch.tensor(<span class="hljs-number">1</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># Batch size 1, Num choices 2</span>

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=m,o=i(),h(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=m),o=l(n),g(r.$$.fragment,n)},m(n,w){c(n,t,w),c(n,o,w),f(r,n,w),T=!0},p:I,i(n){T||(_(r.$$.fragment,n),T=!0)},o(n){M(r.$$.fragment,n),T=!1},d(n){n&&(a(t),a(o)),b(r,n)}}}function Eo(v){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=m},l(o){t=u(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,r){c(o,t,r)},p:I,d(o){o&&a(t)}}}function Lo(v){let t,m="Example:",o,r,T;return r=new _e({props:{code:"JTIzJTIwQXNzdW1wdGlvbiUzQSUyMCpnZXRfdmlzdWFsX2VtYmVkZGluZ3MoaW1hZ2UpKiUyMGdldHMlMjB0aGUlMjB2aXN1YWwlMjBlbWJlZGRpbmdzJTIwb2YlMjB0aGUlMjBpbWFnZSUyMGluJTIwdGhlJTIwYmF0Y2guJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBWaXN1YWxCZXJ0Rm9yUmVnaW9uVG9QaHJhc2VBbGlnbm1lbnQlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBWaXN1YWxCZXJ0Rm9yUmVnaW9uVG9QaHJhc2VBbGlnbm1lbnQuZnJvbV9wcmV0cmFpbmVkKCUyMnVjbGFubHAlMkZ2aXN1YWxiZXJ0LXZxYS1jb2NvLXByZSUyMiklMEElMEF0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwaXMlMjBlYXRpbmclMjB0aGUlMjBhcHBsZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcih0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF2aXN1YWxfZW1iZWRzJTIwJTNEJTIwZ2V0X3Zpc3VhbF9lbWJlZGRpbmdzKGltYWdlKS51bnNxdWVlemUoMCklMEF2aXN1YWxfdG9rZW5fdHlwZV9pZHMlMjAlM0QlMjB0b3JjaC5vbmVzKHZpc3VhbF9lbWJlZHMuc2hhcGUlNUIlM0EtMSU1RCUyQyUyMGR0eXBlJTNEdG9yY2gubG9uZyklMEF2aXN1YWxfYXR0ZW50aW9uX21hc2slMjAlM0QlMjB0b3JjaC5vbmVzKHZpc3VhbF9lbWJlZHMuc2hhcGUlNUIlM0EtMSU1RCUyQyUyMGR0eXBlJTNEdG9yY2guZmxvYXQpJTBBcmVnaW9uX3RvX3BocmFzZV9wb3NpdGlvbiUyMCUzRCUyMHRvcmNoLm9uZXMoKDElMkMlMjBpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQuc2hhcGUlNUItMSU1RCUyMCUyQiUyMHZpc3VhbF9lbWJlZHMuc2hhcGUlNUItMiU1RCkpJTBBJTBBaW5wdXRzLnVwZGF0ZSglMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyZWdpb25fdG9fcGhyYXNlX3Bvc2l0aW9uJTIyJTNBJTIwcmVnaW9uX3RvX3BocmFzZV9wb3NpdGlvbiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnZpc3VhbF9lbWJlZHMlMjIlM0ElMjB2aXN1YWxfZW1iZWRzJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydmlzdWFsX3Rva2VuX3R5cGVfaWRzJTIyJTNBJTIwdmlzdWFsX3Rva2VuX3R5cGVfaWRzJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIydmlzdWFsX2F0dGVudGlvbl9tYXNrJTIyJTNBJTIwdmlzdWFsX2F0dGVudGlvbl9tYXNrJTJDJTBBJTIwJTIwJTIwJTIwJTdEJTBBKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLm9uZXMoJTBBJTIwJTIwJTIwJTIwKDElMkMlMjBpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQuc2hhcGUlNUItMSU1RCUyMCUyQiUyMHZpc3VhbF9lbWJlZHMuc2hhcGUlNUItMiU1RCUyQyUyMHZpc3VhbF9lbWJlZHMuc2hhcGUlNUItMiU1RCklMEEpJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBc2NvcmVzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-comment"># Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.</span>
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, VisualBertForRegionToPhraseAlignment
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = VisualBertForRegionToPhraseAlignment.from_pretrained(<span class="hljs-string">&quot;uclanlp/visualbert-vqa-coco-pre&quot;</span>)

text = <span class="hljs-string">&quot;Who is eating the apple?&quot;</span>
inputs = tokenizer(text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
visual_embeds = get_visual_embeddings(image).unsqueeze(<span class="hljs-number">0</span>)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-<span class="hljs-number">1</span>], dtype=torch.<span class="hljs-built_in">float</span>)
region_to_phrase_position = torch.ones((<span class="hljs-number">1</span>, inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape[-<span class="hljs-number">1</span>] + visual_embeds.shape[-<span class="hljs-number">2</span>]))

inputs.update(
    {
        <span class="hljs-string">&quot;region_to_phrase_position&quot;</span>: region_to_phrase_position,
        <span class="hljs-string">&quot;visual_embeds&quot;</span>: visual_embeds,
        <span class="hljs-string">&quot;visual_token_type_ids&quot;</span>: visual_token_type_ids,
        <span class="hljs-string">&quot;visual_attention_mask&quot;</span>: visual_attention_mask,
    }
)

labels = torch.ones(
    (<span class="hljs-number">1</span>, inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape[-<span class="hljs-number">1</span>] + visual_embeds.shape[-<span class="hljs-number">2</span>], visual_embeds.shape[-<span class="hljs-number">2</span>])
)  <span class="hljs-comment"># Batch size 1</span>

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=m,o=i(),h(r.$$.fragment)},l(n){t=u(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=m),o=l(n),g(r.$$.fragment,n)},m(n,w){c(n,t,w),c(n,o,w),f(r,n,w),T=!0},p:I,i(n){T||(_(r.$$.fragment,n),T=!0)},o(n){M(r.$$.fragment,n),T=!1},d(n){n&&(a(t),a(o)),b(r,n)}}}function Qo(v){let t,m,o,r,T,n="<em>This model was released on 2019-08-09 and added to Hugging Face Transformers on 2021-06-02.</em>",w,te,An='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',vt,Me,wt,be,Gn='<a href="https://huggingface.co/papers/1908.03557" rel="nofollow">VisualBERT</a> is a vision-and-language model. It uses an approach called “early fusion”, where inputs are fed together into a single Transformer stack initialized from <a href="./bert">BERT</a>. Self-attention implicitly aligns words with their corresponding image objects. It processes text with visual features from object-detector regions instead of raw pixels.',Jt,ye,Hn='You can find all the original VisualBERT checkpoints under the <a href="https://huggingface.co/uclanlp/models?search=visualbert" rel="nofollow">UCLA NLP</a> organization.',kt,ne,Bt,Te,qn='The example below demonstrates how to answer a question based on an image with the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',Ut,oe,jt,ve,Vt,we,En='<li>Use a fine-tuned checkpoint for downstream tasks, like <code>visualbert-vqa</code> for visual question answering. Otherwise, use one of the pretrained checkpoints.</li> <li>The fine-tuned detector and weights aren’t provided (available in the research projects), but the states can be directly loaded into the detector.</li> <li>The text input is concatenated in front of the visual embeddings in the embedding layer and is expected to be bound by <code>[CLS]</code> and <code>SEP</code> tokens.</li> <li>The segment ids must be set appropriately for the text and visual parts.</li> <li>Use <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a> to encode the text and implement a custom detector/image processor to get the visual embeddings.</li>',Wt,Je,Ct,ke,Ln='<li>Refer to this <a href="https://github.com/huggingface/transformers-research-projects/tree/main/visual_bert" rel="nofollow">notebook</a> for an example of using VisualBERT for visual question answering.</li> <li>Refer to this <a href="https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing" rel="nofollow">notebook</a> for an example of how to generate visual embeddings.</li>',It,Be,zt,z,Ue,St,Se,Qn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel">VisualBertModel</a>. It is used to instantiate an
VisualBERT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the VisualBERT
<a href="https://huggingface.co/uclanlp/visualbert-vqa-coco-pre" rel="nofollow">uclanlp/visualbert-vqa-coco-pre</a> architecture.`,Pt,Pe,Yn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ot,se,Ft,je,Zt,J,Ve,Dt,Oe,Sn=`The model can behave as an encoder (with only self-attention) following the architecture described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention is
all you need</a> by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`,Kt,De,Pn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,en,Ke,On=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,tn,A,We,nn,et,Dn='The <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel">VisualBertModel</a> forward method, overrides the <code>__call__</code> special method.',on,ae,sn,re,$t,Ce,xt,k,Ie,an,tt,Kn=`VisualBert Model with two heads on top as done during the pretraining: a <code>masked language modeling</code> head and a
<code>sentence-image prediction (classification)</code> head.`,rn,nt,eo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ln,ot,to=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,dn,G,ze,cn,st,no='The <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForPreTraining">VisualBertForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',pn,ie,un,le,Rt,Fe,Xt,B,Ze,mn,at,oo=`VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled
output) for VQA.`,hn,rt,so=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,gn,it,ao=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fn,H,$e,_n,lt,ro='The <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForQuestionAnswering">VisualBertForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Mn,de,bn,ce,Nt,xe,At,U,Re,yn,dt,io=`The Visual Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Tn,ct,lo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vn,pt,co=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wn,q,Xe,Jn,ut,po='The <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForMultipleChoice">VisualBertForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',kn,pe,Bn,ue,Gt,Ne,Ht,j,Ae,Un,mt,uo=`VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled
output) for Visual Reasoning e.g. for NLVR task.`,jn,ht,mo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Vn,gt,ho=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wn,E,Ge,Cn,ft,go='The <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForVisualReasoning">VisualBertForVisualReasoning</a> forward method, overrides the <code>__call__</code> special method.',In,me,zn,he,qt,He,Et,V,qe,Fn,_t,fo=`VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment
e.g. for Flickr30 Entities task.`,Zn,Mt,_o=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$n,bt,Mo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xn,L,Ee,Rn,yt,bo='The <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForRegionToPhraseAlignment">VisualBertForRegionToPhraseAlignment</a> forward method, overrides the <code>__call__</code> special method.',Xn,ge,Nn,fe,Lt,Le,Qt,Tt,Yt;return Me=new ee({props:{title:"VisualBERT",local:"visualbert",headingTag:"h1"}}),ne=new Qe({props:{warning:!1,$$slots:{default:[Co]},$$scope:{ctx:v}}}),oe=new Vo({props:{id:"usage",options:["AutoModel"],$$slots:{default:[zo]},$$scope:{ctx:v}}}),ve=new ee({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Je=new ee({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Be=new ee({props:{title:"VisualBertConfig",local:"transformers.VisualBertConfig",headingTag:"h2"}}),Ue=new N({props:{name:"class transformers.VisualBertConfig",anchor:"transformers.VisualBertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"visual_embedding_dim",val:" = 512"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"bypass_transformer",val:" = False"},{name:"special_visual_initialize",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.VisualBertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the VisualBERT model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel">VisualBertModel</a>. Vocabulary size of the model. Defines the
different tokens that can be represented by the <code>inputs_ids</code> passed to the forward method of
<a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel">VisualBertModel</a>.`,name:"vocab_size"},{anchor:"transformers.VisualBertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.VisualBertConfig.visual_embedding_dim",description:`<strong>visual_embedding_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the visual embeddings to be passed to the model.`,name:"visual_embedding_dim"},{anchor:"transformers.VisualBertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.VisualBertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.VisualBertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.VisualBertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.VisualBertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.VisualBertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.VisualBertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.VisualBertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel">VisualBertModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.VisualBertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.VisualBertConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.VisualBertConfig.bypass_transformer",description:`<strong>bypass_transformer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the model should bypass the transformer for the visual embeddings. If set to <code>True</code>, the
model directly concatenates the visual embeddings from <code>VisualBertEmbeddings</code> with text output from
transformers, and then pass it to a self-attention layer.`,name:"bypass_transformer"},{anchor:"transformers.VisualBertConfig.special_visual_initialize",description:`<strong>special_visual_initialize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the visual token type and position type embedding weights should be initialized the same as
the textual token type and positive type embeddings. When set to <code>True</code>, the weights of the textual token
type and position type embeddings are copied to the respective visual embedding layers.`,name:"special_visual_initialize"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/configuration_visual_bert.py#L24"}}),se=new Ye({props:{anchor:"transformers.VisualBertConfig.example",$$slots:{default:[Fo]},$$scope:{ctx:v}}}),je=new ee({props:{title:"VisualBertModel",local:"transformers.VisualBertModel",headingTag:"h2"}}),Ve=new N({props:{name:"class transformers.VisualBertModel",anchor:"transformers.VisualBertModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.VisualBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel">VisualBertModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.VisualBertModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L551"}}),We=new N({props:{name:"forward",anchor:"transformers.VisualBertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"visual_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"image_text_alignment",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.VisualBertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.VisualBertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.VisualBertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.VisualBertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.VisualBertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VisualBertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.VisualBertModel.forward.visual_embeds",description:`<strong>visual_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length, visual_embedding_dim)</code>, <em>optional</em>) &#x2014;
The embedded representation of the visual inputs, generally derived using using an object detector.`,name:"visual_embeds"},{anchor:"transformers.VisualBertModel.forward.visual_attention_mask",description:`<strong>visual_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on visual embeddings. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"visual_attention_mask"},{anchor:"transformers.VisualBertModel.forward.visual_token_type_ids",description:`<strong>visual_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate different portions of the visual embeds.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a> The authors of VisualBERT set the
<em>visual_token_type_ids</em> to <em>1</em> for all tokens.`,name:"visual_token_type_ids"},{anchor:"transformers.VisualBertModel.forward.image_text_alignment",description:`<strong>image_text_alignment</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length, alignment_number)</code>, <em>optional</em>) &#x2014;
Image-Text alignment uses to decide the position IDs of the visual embeddings.`,name:"image_text_alignment"},{anchor:"transformers.VisualBertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VisualBertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VisualBertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L587",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig"
>VisualBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, hidden_size)</code>) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
the classification token after processing through a linear layer and a tanh activation function. The linear
layer weights are trained from the next sentence prediction (classification) objective during pretraining.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ae=new Qe({props:{$$slots:{default:[Zo]},$$scope:{ctx:v}}}),re=new Ye({props:{anchor:"transformers.VisualBertModel.forward.example",$$slots:{default:[$o]},$$scope:{ctx:v}}}),Ce=new ee({props:{title:"VisualBertForPreTraining",local:"transformers.VisualBertForPreTraining",headingTag:"h2"}}),Ie=new N({props:{name:"class transformers.VisualBertForPreTraining",anchor:"transformers.VisualBertForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.VisualBertForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForPreTraining">VisualBertForPreTraining</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L757"}}),ze=new N({props:{name:"forward",anchor:"transformers.VisualBertForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"visual_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"image_text_alignment",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"sentence_image_labels",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.VisualBertForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.VisualBertForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.VisualBertForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.VisualBertForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.VisualBertForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VisualBertForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.VisualBertForPreTraining.forward.visual_embeds",description:`<strong>visual_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length, visual_embedding_dim)</code>, <em>optional</em>) &#x2014;
The embedded representation of the visual inputs, generally derived using using an object detector.`,name:"visual_embeds"},{anchor:"transformers.VisualBertForPreTraining.forward.visual_attention_mask",description:`<strong>visual_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on visual embeddings. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"visual_attention_mask"},{anchor:"transformers.VisualBertForPreTraining.forward.visual_token_type_ids",description:`<strong>visual_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate different portions of the visual embeds.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a> The authors of VisualBERT set the
<em>visual_token_type_ids</em> to <em>1</em> for all tokens.`,name:"visual_token_type_ids"},{anchor:"transformers.VisualBertForPreTraining.forward.image_text_alignment",description:`<strong>image_text_alignment</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length, alignment_number)</code>, <em>optional</em>) &#x2014;
Image-Text alignment uses to decide the position IDs of the visual embeddings.`,name:"image_text_alignment"},{anchor:"transformers.VisualBertForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VisualBertForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VisualBertForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.VisualBertForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, total_sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.VisualBertForPreTraining.forward.sentence_image_labels",description:`<strong>sentence_image_labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sentence-image prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a matching pair of sequence A for the given image,</li>
<li>1 indicates sequence B is a random sequence w.r.t A for the given image.</li>
</ul>`,name:"sentence_image_labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L776",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.visual_bert.modeling_visual_bert.VisualBertForPreTrainingOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig"
>VisualBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) — Total loss as the sum of the masked language modeling loss and the sentence-image prediction
(classification) loss.</p>
</li>
<li>
<p><strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) — Prediction scores of the sentence-image prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
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


<p><code>transformers.models.visual_bert.modeling_visual_bert.VisualBertForPreTrainingOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ie=new Qe({props:{$$slots:{default:[xo]},$$scope:{ctx:v}}}),le=new Ye({props:{anchor:"transformers.VisualBertForPreTraining.forward.example",$$slots:{default:[Ro]},$$scope:{ctx:v}}}),Fe=new ee({props:{title:"VisualBertForQuestionAnswering",local:"transformers.VisualBertForQuestionAnswering",headingTag:"h2"}}),Ze=new N({props:{name:"class transformers.VisualBertForQuestionAnswering",anchor:"transformers.VisualBertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.VisualBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForQuestionAnswering">VisualBertForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1097"}}),$e=new N({props:{name:"forward",anchor:"transformers.VisualBertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"visual_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"image_text_alignment",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.VisualBertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.visual_embeds",description:`<strong>visual_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length, visual_embedding_dim)</code>, <em>optional</em>) &#x2014;
The embedded representation of the visual inputs, generally derived using using an object detector.`,name:"visual_embeds"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.visual_attention_mask",description:`<strong>visual_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on visual embeddings. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"visual_attention_mask"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.visual_token_type_ids",description:`<strong>visual_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate different portions of the visual embeds.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a> The authors of VisualBERT set the
<em>visual_token_type_ids</em> to <em>1</em> for all tokens.`,name:"visual_token_type_ids"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.image_text_alignment",description:`<strong>image_text_alignment</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length, alignment_number)</code>, <em>optional</em>) &#x2014;
Image-Text alignment uses to decide the position IDs of the visual embeddings.`,name:"image_text_alignment"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.VisualBertForQuestionAnswering.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, total_sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. A KLDivLoss is computed between the labels and the returned logits.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1109",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig"
>VisualBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),de=new Qe({props:{$$slots:{default:[Xo]},$$scope:{ctx:v}}}),ce=new Ye({props:{anchor:"transformers.VisualBertForQuestionAnswering.forward.example",$$slots:{default:[No]},$$scope:{ctx:v}}}),xe=new ee({props:{title:"VisualBertForMultipleChoice",local:"transformers.VisualBertForMultipleChoice",headingTag:"h2"}}),Re=new N({props:{name:"class transformers.VisualBertForMultipleChoice",anchor:"transformers.VisualBertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.VisualBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForMultipleChoice">VisualBertForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L910"}}),Xe=new N({props:{name:"forward",anchor:"transformers.VisualBertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"visual_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"image_text_alignment",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.VisualBertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.VisualBertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.VisualBertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.VisualBertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.VisualBertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VisualBertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.VisualBertForMultipleChoice.forward.visual_embeds",description:`<strong>visual_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length, visual_embedding_dim)</code>, <em>optional</em>) &#x2014;
The embedded representation of the visual inputs, generally derived using using an object detector.`,name:"visual_embeds"},{anchor:"transformers.VisualBertForMultipleChoice.forward.visual_attention_mask",description:`<strong>visual_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on visual embeddings. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"visual_attention_mask"},{anchor:"transformers.VisualBertForMultipleChoice.forward.visual_token_type_ids",description:`<strong>visual_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate different portions of the visual embeds.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a> The authors of VisualBERT set the
<em>visual_token_type_ids</em> to <em>1</em> for all tokens.`,name:"visual_token_type_ids"},{anchor:"transformers.VisualBertForMultipleChoice.forward.image_text_alignment",description:`<strong>image_text_alignment</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length, alignment_number)</code>, <em>optional</em>) &#x2014;
Image-Text alignment uses to decide the position IDs of the visual embeddings.`,name:"image_text_alignment"},{anchor:"transformers.VisualBertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VisualBertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VisualBertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.VisualBertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L921",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig"
>VisualBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) — <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),pe=new Qe({props:{$$slots:{default:[Ao]},$$scope:{ctx:v}}}),ue=new Ye({props:{anchor:"transformers.VisualBertForMultipleChoice.forward.example",$$slots:{default:[Go]},$$scope:{ctx:v}}}),Ne=new ee({props:{title:"VisualBertForVisualReasoning",local:"transformers.VisualBertForVisualReasoning",headingTag:"h2"}}),Ae=new N({props:{name:"class transformers.VisualBertForVisualReasoning",anchor:"transformers.VisualBertForVisualReasoning",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.VisualBertForVisualReasoning.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForVisualReasoning">VisualBertForVisualReasoning</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1235"}}),Ge=new N({props:{name:"forward",anchor:"transformers.VisualBertForVisualReasoning.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"visual_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"image_text_alignment",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.VisualBertForVisualReasoning.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.VisualBertForVisualReasoning.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.VisualBertForVisualReasoning.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.VisualBertForVisualReasoning.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.VisualBertForVisualReasoning.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VisualBertForVisualReasoning.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.VisualBertForVisualReasoning.forward.visual_embeds",description:`<strong>visual_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length, visual_embedding_dim)</code>, <em>optional</em>) &#x2014;
The embedded representation of the visual inputs, generally derived using using an object detector.`,name:"visual_embeds"},{anchor:"transformers.VisualBertForVisualReasoning.forward.visual_attention_mask",description:`<strong>visual_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on visual embeddings. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"visual_attention_mask"},{anchor:"transformers.VisualBertForVisualReasoning.forward.visual_token_type_ids",description:`<strong>visual_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate different portions of the visual embeds.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a> The authors of VisualBERT set the
<em>visual_token_type_ids</em> to <em>1</em> for all tokens.`,name:"visual_token_type_ids"},{anchor:"transformers.VisualBertForVisualReasoning.forward.image_text_alignment",description:`<strong>image_text_alignment</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length, alignment_number)</code>, <em>optional</em>) &#x2014;
Image-Text alignment uses to decide the position IDs of the visual embeddings.`,name:"image_text_alignment"},{anchor:"transformers.VisualBertForVisualReasoning.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VisualBertForVisualReasoning.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VisualBertForVisualReasoning.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.VisualBertForVisualReasoning.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. A classification loss is computed (Cross-Entropy) against these labels.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1247",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig"
>VisualBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),me=new Qe({props:{$$slots:{default:[Ho]},$$scope:{ctx:v}}}),he=new Ye({props:{anchor:"transformers.VisualBertForVisualReasoning.forward.example",$$slots:{default:[qo]},$$scope:{ctx:v}}}),He=new ee({props:{title:"VisualBertForRegionToPhraseAlignment",local:"transformers.VisualBertForRegionToPhraseAlignment",headingTag:"h2"}}),qe=new N({props:{name:"class transformers.VisualBertForRegionToPhraseAlignment",anchor:"transformers.VisualBertForRegionToPhraseAlignment",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.VisualBertForRegionToPhraseAlignment.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForRegionToPhraseAlignment">VisualBertForRegionToPhraseAlignment</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1404"}}),Ee=new N({props:{name:"forward",anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"visual_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"visual_token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"image_text_alignment",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"region_to_phrase_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.visual_embeds",description:`<strong>visual_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length, visual_embedding_dim)</code>, <em>optional</em>) &#x2014;
The embedded representation of the visual inputs, generally derived using using an object detector.`,name:"visual_embeds"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.visual_attention_mask",description:`<strong>visual_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on visual embeddings. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"visual_attention_mask"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.visual_token_type_ids",description:`<strong>visual_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate different portions of the visual embeds.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a> The authors of VisualBERT set the
<em>visual_token_type_ids</em> to <em>1</em> for all tokens.`,name:"visual_token_type_ids"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.image_text_alignment",description:`<strong>image_text_alignment</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, visual_seq_length, alignment_number)</code>, <em>optional</em>) &#x2014;
Image-Text alignment uses to decide the position IDs of the visual embeddings.`,name:"image_text_alignment"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.region_to_phrase_position",description:`<strong>region_to_phrase_position</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, total_sequence_length)</code>, <em>optional</em>) &#x2014;
The positions depicting the position of the image embedding corresponding to the textual tokens.`,name:"region_to_phrase_position"},{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, total_sequence_length, visual_sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. KLDivLoss is computed against these labels and the
outputs from the attention layer.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1418",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig"
>VisualBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ge=new Qe({props:{$$slots:{default:[Eo]},$$scope:{ctx:v}}}),fe=new Ye({props:{anchor:"transformers.VisualBertForRegionToPhraseAlignment.forward.example",$$slots:{default:[Lo]},$$scope:{ctx:v}}}),Le=new jo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/visual_bert.md"}}),{c(){t=p("meta"),m=i(),o=p("p"),r=i(),T=p("p"),T.innerHTML=n,w=i(),te=p("div"),te.innerHTML=An,vt=i(),h(Me.$$.fragment),wt=i(),be=p("p"),be.innerHTML=Gn,Jt=i(),ye=p("p"),ye.innerHTML=Hn,kt=i(),h(ne.$$.fragment),Bt=i(),Te=p("p"),Te.innerHTML=qn,Ut=i(),h(oe.$$.fragment),jt=i(),h(ve.$$.fragment),Vt=i(),we=p("ul"),we.innerHTML=En,Wt=i(),h(Je.$$.fragment),Ct=i(),ke=p("ul"),ke.innerHTML=Ln,It=i(),h(Be.$$.fragment),zt=i(),z=p("div"),h(Ue.$$.fragment),St=i(),Se=p("p"),Se.innerHTML=Qn,Pt=i(),Pe=p("p"),Pe.innerHTML=Yn,Ot=i(),h(se.$$.fragment),Ft=i(),h(je.$$.fragment),Zt=i(),J=p("div"),h(Ve.$$.fragment),Dt=i(),Oe=p("p"),Oe.innerHTML=Sn,Kt=i(),De=p("p"),De.innerHTML=Pn,en=i(),Ke=p("p"),Ke.innerHTML=On,tn=i(),A=p("div"),h(We.$$.fragment),nn=i(),et=p("p"),et.innerHTML=Dn,on=i(),h(ae.$$.fragment),sn=i(),h(re.$$.fragment),$t=i(),h(Ce.$$.fragment),xt=i(),k=p("div"),h(Ie.$$.fragment),an=i(),tt=p("p"),tt.innerHTML=Kn,rn=i(),nt=p("p"),nt.innerHTML=eo,ln=i(),ot=p("p"),ot.innerHTML=to,dn=i(),G=p("div"),h(ze.$$.fragment),cn=i(),st=p("p"),st.innerHTML=no,pn=i(),h(ie.$$.fragment),un=i(),h(le.$$.fragment),Rt=i(),h(Fe.$$.fragment),Xt=i(),B=p("div"),h(Ze.$$.fragment),mn=i(),at=p("p"),at.textContent=oo,hn=i(),rt=p("p"),rt.innerHTML=so,gn=i(),it=p("p"),it.innerHTML=ao,fn=i(),H=p("div"),h($e.$$.fragment),_n=i(),lt=p("p"),lt.innerHTML=ro,Mn=i(),h(de.$$.fragment),bn=i(),h(ce.$$.fragment),Nt=i(),h(xe.$$.fragment),At=i(),U=p("div"),h(Re.$$.fragment),yn=i(),dt=p("p"),dt.textContent=io,Tn=i(),ct=p("p"),ct.innerHTML=lo,vn=i(),pt=p("p"),pt.innerHTML=co,wn=i(),q=p("div"),h(Xe.$$.fragment),Jn=i(),ut=p("p"),ut.innerHTML=po,kn=i(),h(pe.$$.fragment),Bn=i(),h(ue.$$.fragment),Gt=i(),h(Ne.$$.fragment),Ht=i(),j=p("div"),h(Ae.$$.fragment),Un=i(),mt=p("p"),mt.textContent=uo,jn=i(),ht=p("p"),ht.innerHTML=mo,Vn=i(),gt=p("p"),gt.innerHTML=ho,Wn=i(),E=p("div"),h(Ge.$$.fragment),Cn=i(),ft=p("p"),ft.innerHTML=go,In=i(),h(me.$$.fragment),zn=i(),h(he.$$.fragment),qt=i(),h(He.$$.fragment),Et=i(),V=p("div"),h(qe.$$.fragment),Fn=i(),_t=p("p"),_t.textContent=fo,Zn=i(),Mt=p("p"),Mt.innerHTML=_o,$n=i(),bt=p("p"),bt.innerHTML=Mo,xn=i(),L=p("div"),h(Ee.$$.fragment),Rn=i(),yt=p("p"),yt.innerHTML=bo,Xn=i(),h(ge.$$.fragment),Nn=i(),h(fe.$$.fragment),Lt=i(),h(Le.$$.fragment),Qt=i(),Tt=p("p"),this.h()},l(e){const s=Bo("svelte-u9bgzb",document.head);t=u(s,"META",{name:!0,content:!0}),s.forEach(a),m=l(e),o=u(e,"P",{}),W(o).forEach(a),r=l(e),T=u(e,"P",{"data-svelte-h":!0}),y(T)!=="svelte-1kszy90"&&(T.innerHTML=n),w=l(e),te=u(e,"DIV",{style:!0,"data-svelte-h":!0}),y(te)!=="svelte-wa5t4p"&&(te.innerHTML=An),vt=l(e),g(Me.$$.fragment,e),wt=l(e),be=u(e,"P",{"data-svelte-h":!0}),y(be)!=="svelte-158hptw"&&(be.innerHTML=Gn),Jt=l(e),ye=u(e,"P",{"data-svelte-h":!0}),y(ye)!=="svelte-hnghl0"&&(ye.innerHTML=Hn),kt=l(e),g(ne.$$.fragment,e),Bt=l(e),Te=u(e,"P",{"data-svelte-h":!0}),y(Te)!=="svelte-wuzlvy"&&(Te.innerHTML=qn),Ut=l(e),g(oe.$$.fragment,e),jt=l(e),g(ve.$$.fragment,e),Vt=l(e),we=u(e,"UL",{"data-svelte-h":!0}),y(we)!=="svelte-7zdo04"&&(we.innerHTML=En),Wt=l(e),g(Je.$$.fragment,e),Ct=l(e),ke=u(e,"UL",{"data-svelte-h":!0}),y(ke)!=="svelte-kgc2ic"&&(ke.innerHTML=Ln),It=l(e),g(Be.$$.fragment,e),zt=l(e),z=u(e,"DIV",{class:!0});var Q=W(z);g(Ue.$$.fragment,Q),St=l(Q),Se=u(Q,"P",{"data-svelte-h":!0}),y(Se)!=="svelte-1pibdjj"&&(Se.innerHTML=Qn),Pt=l(Q),Pe=u(Q,"P",{"data-svelte-h":!0}),y(Pe)!=="svelte-1ek1ss9"&&(Pe.innerHTML=Yn),Ot=l(Q),g(se.$$.fragment,Q),Q.forEach(a),Ft=l(e),g(je.$$.fragment,e),Zt=l(e),J=u(e,"DIV",{class:!0});var F=W(J);g(Ve.$$.fragment,F),Dt=l(F),Oe=u(F,"P",{"data-svelte-h":!0}),y(Oe)!=="svelte-t7tkwd"&&(Oe.innerHTML=Sn),Kt=l(F),De=u(F,"P",{"data-svelte-h":!0}),y(De)!=="svelte-q52n56"&&(De.innerHTML=Pn),en=l(F),Ke=u(F,"P",{"data-svelte-h":!0}),y(Ke)!=="svelte-hswkmf"&&(Ke.innerHTML=On),tn=l(F),A=u(F,"DIV",{class:!0});var Y=W(A);g(We.$$.fragment,Y),nn=l(Y),et=u(Y,"P",{"data-svelte-h":!0}),y(et)!=="svelte-ksdrg1"&&(et.innerHTML=Dn),on=l(Y),g(ae.$$.fragment,Y),sn=l(Y),g(re.$$.fragment,Y),Y.forEach(a),F.forEach(a),$t=l(e),g(Ce.$$.fragment,e),xt=l(e),k=u(e,"DIV",{class:!0});var Z=W(k);g(Ie.$$.fragment,Z),an=l(Z),tt=u(Z,"P",{"data-svelte-h":!0}),y(tt)!=="svelte-izdyyc"&&(tt.innerHTML=Kn),rn=l(Z),nt=u(Z,"P",{"data-svelte-h":!0}),y(nt)!=="svelte-q52n56"&&(nt.innerHTML=eo),ln=l(Z),ot=u(Z,"P",{"data-svelte-h":!0}),y(ot)!=="svelte-hswkmf"&&(ot.innerHTML=to),dn=l(Z),G=u(Z,"DIV",{class:!0});var S=W(G);g(ze.$$.fragment,S),cn=l(S),st=u(S,"P",{"data-svelte-h":!0}),y(st)!=="svelte-oprz9n"&&(st.innerHTML=no),pn=l(S),g(ie.$$.fragment,S),un=l(S),g(le.$$.fragment,S),S.forEach(a),Z.forEach(a),Rt=l(e),g(Fe.$$.fragment,e),Xt=l(e),B=u(e,"DIV",{class:!0});var $=W(B);g(Ze.$$.fragment,$),mn=l($),at=u($,"P",{"data-svelte-h":!0}),y(at)!=="svelte-1kebopj"&&(at.textContent=oo),hn=l($),rt=u($,"P",{"data-svelte-h":!0}),y(rt)!=="svelte-q52n56"&&(rt.innerHTML=so),gn=l($),it=u($,"P",{"data-svelte-h":!0}),y(it)!=="svelte-hswkmf"&&(it.innerHTML=ao),fn=l($),H=u($,"DIV",{class:!0});var P=W(H);g($e.$$.fragment,P),_n=l(P),lt=u(P,"P",{"data-svelte-h":!0}),y(lt)!=="svelte-ik2j65"&&(lt.innerHTML=ro),Mn=l(P),g(de.$$.fragment,P),bn=l(P),g(ce.$$.fragment,P),P.forEach(a),$.forEach(a),Nt=l(e),g(xe.$$.fragment,e),At=l(e),U=u(e,"DIV",{class:!0});var x=W(U);g(Re.$$.fragment,x),yn=l(x),dt=u(x,"P",{"data-svelte-h":!0}),y(dt)!=="svelte-umjqji"&&(dt.textContent=io),Tn=l(x),ct=u(x,"P",{"data-svelte-h":!0}),y(ct)!=="svelte-q52n56"&&(ct.innerHTML=lo),vn=l(x),pt=u(x,"P",{"data-svelte-h":!0}),y(pt)!=="svelte-hswkmf"&&(pt.innerHTML=co),wn=l(x),q=u(x,"DIV",{class:!0});var O=W(q);g(Xe.$$.fragment,O),Jn=l(O),ut=u(O,"P",{"data-svelte-h":!0}),y(ut)!=="svelte-1pciiht"&&(ut.innerHTML=po),kn=l(O),g(pe.$$.fragment,O),Bn=l(O),g(ue.$$.fragment,O),O.forEach(a),x.forEach(a),Gt=l(e),g(Ne.$$.fragment,e),Ht=l(e),j=u(e,"DIV",{class:!0});var R=W(j);g(Ae.$$.fragment,R),Un=l(R),mt=u(R,"P",{"data-svelte-h":!0}),y(mt)!=="svelte-15dl5em"&&(mt.textContent=uo),jn=l(R),ht=u(R,"P",{"data-svelte-h":!0}),y(ht)!=="svelte-q52n56"&&(ht.innerHTML=mo),Vn=l(R),gt=u(R,"P",{"data-svelte-h":!0}),y(gt)!=="svelte-hswkmf"&&(gt.innerHTML=ho),Wn=l(R),E=u(R,"DIV",{class:!0});var D=W(E);g(Ge.$$.fragment,D),Cn=l(D),ft=u(D,"P",{"data-svelte-h":!0}),y(ft)!=="svelte-19x7jil"&&(ft.innerHTML=go),In=l(D),g(me.$$.fragment,D),zn=l(D),g(he.$$.fragment,D),D.forEach(a),R.forEach(a),qt=l(e),g(He.$$.fragment,e),Et=l(e),V=u(e,"DIV",{class:!0});var X=W(V);g(qe.$$.fragment,X),Fn=l(X),_t=u(X,"P",{"data-svelte-h":!0}),y(_t)!=="svelte-5zhgq1"&&(_t.textContent=fo),Zn=l(X),Mt=u(X,"P",{"data-svelte-h":!0}),y(Mt)!=="svelte-q52n56"&&(Mt.innerHTML=_o),$n=l(X),bt=u(X,"P",{"data-svelte-h":!0}),y(bt)!=="svelte-hswkmf"&&(bt.innerHTML=Mo),xn=l(X),L=u(X,"DIV",{class:!0});var K=W(L);g(Ee.$$.fragment,K),Rn=l(K),yt=u(K,"P",{"data-svelte-h":!0}),y(yt)!=="svelte-18keww7"&&(yt.innerHTML=bo),Xn=l(K),g(ge.$$.fragment,K),Nn=l(K),g(fe.$$.fragment,K),K.forEach(a),X.forEach(a),Lt=l(e),g(Le.$$.fragment,e),Qt=l(e),Tt=u(e,"P",{}),W(Tt).forEach(a),this.h()},h(){C(t,"name","hf:doc:metadata"),C(t,"content",Yo),Uo(te,"float","right"),C(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){d(document.head,t),c(e,m,s),c(e,o,s),c(e,r,s),c(e,T,s),c(e,w,s),c(e,te,s),c(e,vt,s),f(Me,e,s),c(e,wt,s),c(e,be,s),c(e,Jt,s),c(e,ye,s),c(e,kt,s),f(ne,e,s),c(e,Bt,s),c(e,Te,s),c(e,Ut,s),f(oe,e,s),c(e,jt,s),f(ve,e,s),c(e,Vt,s),c(e,we,s),c(e,Wt,s),f(Je,e,s),c(e,Ct,s),c(e,ke,s),c(e,It,s),f(Be,e,s),c(e,zt,s),c(e,z,s),f(Ue,z,null),d(z,St),d(z,Se),d(z,Pt),d(z,Pe),d(z,Ot),f(se,z,null),c(e,Ft,s),f(je,e,s),c(e,Zt,s),c(e,J,s),f(Ve,J,null),d(J,Dt),d(J,Oe),d(J,Kt),d(J,De),d(J,en),d(J,Ke),d(J,tn),d(J,A),f(We,A,null),d(A,nn),d(A,et),d(A,on),f(ae,A,null),d(A,sn),f(re,A,null),c(e,$t,s),f(Ce,e,s),c(e,xt,s),c(e,k,s),f(Ie,k,null),d(k,an),d(k,tt),d(k,rn),d(k,nt),d(k,ln),d(k,ot),d(k,dn),d(k,G),f(ze,G,null),d(G,cn),d(G,st),d(G,pn),f(ie,G,null),d(G,un),f(le,G,null),c(e,Rt,s),f(Fe,e,s),c(e,Xt,s),c(e,B,s),f(Ze,B,null),d(B,mn),d(B,at),d(B,hn),d(B,rt),d(B,gn),d(B,it),d(B,fn),d(B,H),f($e,H,null),d(H,_n),d(H,lt),d(H,Mn),f(de,H,null),d(H,bn),f(ce,H,null),c(e,Nt,s),f(xe,e,s),c(e,At,s),c(e,U,s),f(Re,U,null),d(U,yn),d(U,dt),d(U,Tn),d(U,ct),d(U,vn),d(U,pt),d(U,wn),d(U,q),f(Xe,q,null),d(q,Jn),d(q,ut),d(q,kn),f(pe,q,null),d(q,Bn),f(ue,q,null),c(e,Gt,s),f(Ne,e,s),c(e,Ht,s),c(e,j,s),f(Ae,j,null),d(j,Un),d(j,mt),d(j,jn),d(j,ht),d(j,Vn),d(j,gt),d(j,Wn),d(j,E),f(Ge,E,null),d(E,Cn),d(E,ft),d(E,In),f(me,E,null),d(E,zn),f(he,E,null),c(e,qt,s),f(He,e,s),c(e,Et,s),c(e,V,s),f(qe,V,null),d(V,Fn),d(V,_t),d(V,Zn),d(V,Mt),d(V,$n),d(V,bt),d(V,xn),d(V,L),f(Ee,L,null),d(L,Rn),d(L,yt),d(L,Xn),f(ge,L,null),d(L,Nn),f(fe,L,null),c(e,Lt,s),f(Le,e,s),c(e,Qt,s),c(e,Tt,s),Yt=!0},p(e,[s]){const Q={};s&2&&(Q.$$scope={dirty:s,ctx:e}),ne.$set(Q);const F={};s&2&&(F.$$scope={dirty:s,ctx:e}),oe.$set(F);const Y={};s&2&&(Y.$$scope={dirty:s,ctx:e}),se.$set(Y);const Z={};s&2&&(Z.$$scope={dirty:s,ctx:e}),ae.$set(Z);const S={};s&2&&(S.$$scope={dirty:s,ctx:e}),re.$set(S);const $={};s&2&&($.$$scope={dirty:s,ctx:e}),ie.$set($);const P={};s&2&&(P.$$scope={dirty:s,ctx:e}),le.$set(P);const x={};s&2&&(x.$$scope={dirty:s,ctx:e}),de.$set(x);const O={};s&2&&(O.$$scope={dirty:s,ctx:e}),ce.$set(O);const R={};s&2&&(R.$$scope={dirty:s,ctx:e}),pe.$set(R);const D={};s&2&&(D.$$scope={dirty:s,ctx:e}),ue.$set(D);const X={};s&2&&(X.$$scope={dirty:s,ctx:e}),me.$set(X);const K={};s&2&&(K.$$scope={dirty:s,ctx:e}),he.$set(K);const yo={};s&2&&(yo.$$scope={dirty:s,ctx:e}),ge.$set(yo);const To={};s&2&&(To.$$scope={dirty:s,ctx:e}),fe.$set(To)},i(e){Yt||(_(Me.$$.fragment,e),_(ne.$$.fragment,e),_(oe.$$.fragment,e),_(ve.$$.fragment,e),_(Je.$$.fragment,e),_(Be.$$.fragment,e),_(Ue.$$.fragment,e),_(se.$$.fragment,e),_(je.$$.fragment,e),_(Ve.$$.fragment,e),_(We.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(Ce.$$.fragment,e),_(Ie.$$.fragment,e),_(ze.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),_(Fe.$$.fragment,e),_(Ze.$$.fragment,e),_($e.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(xe.$$.fragment,e),_(Re.$$.fragment,e),_(Xe.$$.fragment,e),_(pe.$$.fragment,e),_(ue.$$.fragment,e),_(Ne.$$.fragment,e),_(Ae.$$.fragment,e),_(Ge.$$.fragment,e),_(me.$$.fragment,e),_(he.$$.fragment,e),_(He.$$.fragment,e),_(qe.$$.fragment,e),_(Ee.$$.fragment,e),_(ge.$$.fragment,e),_(fe.$$.fragment,e),_(Le.$$.fragment,e),Yt=!0)},o(e){M(Me.$$.fragment,e),M(ne.$$.fragment,e),M(oe.$$.fragment,e),M(ve.$$.fragment,e),M(Je.$$.fragment,e),M(Be.$$.fragment,e),M(Ue.$$.fragment,e),M(se.$$.fragment,e),M(je.$$.fragment,e),M(Ve.$$.fragment,e),M(We.$$.fragment,e),M(ae.$$.fragment,e),M(re.$$.fragment,e),M(Ce.$$.fragment,e),M(Ie.$$.fragment,e),M(ze.$$.fragment,e),M(ie.$$.fragment,e),M(le.$$.fragment,e),M(Fe.$$.fragment,e),M(Ze.$$.fragment,e),M($e.$$.fragment,e),M(de.$$.fragment,e),M(ce.$$.fragment,e),M(xe.$$.fragment,e),M(Re.$$.fragment,e),M(Xe.$$.fragment,e),M(pe.$$.fragment,e),M(ue.$$.fragment,e),M(Ne.$$.fragment,e),M(Ae.$$.fragment,e),M(Ge.$$.fragment,e),M(me.$$.fragment,e),M(he.$$.fragment,e),M(He.$$.fragment,e),M(qe.$$.fragment,e),M(Ee.$$.fragment,e),M(ge.$$.fragment,e),M(fe.$$.fragment,e),M(Le.$$.fragment,e),Yt=!1},d(e){e&&(a(m),a(o),a(r),a(T),a(w),a(te),a(vt),a(wt),a(be),a(Jt),a(ye),a(kt),a(Bt),a(Te),a(Ut),a(jt),a(Vt),a(we),a(Wt),a(Ct),a(ke),a(It),a(zt),a(z),a(Ft),a(Zt),a(J),a($t),a(xt),a(k),a(Rt),a(Xt),a(B),a(Nt),a(At),a(U),a(Gt),a(Ht),a(j),a(qt),a(Et),a(V),a(Lt),a(Qt),a(Tt)),a(t),b(Me,e),b(ne,e),b(oe,e),b(ve,e),b(Je,e),b(Be,e),b(Ue),b(se),b(je,e),b(Ve),b(We),b(ae),b(re),b(Ce,e),b(Ie),b(ze),b(ie),b(le),b(Fe,e),b(Ze),b($e),b(de),b(ce),b(xe,e),b(Re),b(Xe),b(pe),b(ue),b(Ne,e),b(Ae),b(Ge),b(me),b(he),b(He,e),b(qe),b(Ee),b(ge),b(fe),b(Le,e)}}}const Yo='{"title":"VisualBERT","local":"visualbert","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"VisualBertConfig","local":"transformers.VisualBertConfig","sections":[],"depth":2},{"title":"VisualBertModel","local":"transformers.VisualBertModel","sections":[],"depth":2},{"title":"VisualBertForPreTraining","local":"transformers.VisualBertForPreTraining","sections":[],"depth":2},{"title":"VisualBertForQuestionAnswering","local":"transformers.VisualBertForQuestionAnswering","sections":[],"depth":2},{"title":"VisualBertForMultipleChoice","local":"transformers.VisualBertForMultipleChoice","sections":[],"depth":2},{"title":"VisualBertForVisualReasoning","local":"transformers.VisualBertForVisualReasoning","sections":[],"depth":2},{"title":"VisualBertForRegionToPhraseAlignment","local":"transformers.VisualBertForRegionToPhraseAlignment","sections":[],"depth":2}],"depth":1}';function So(v){return wo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ss extends Jo{constructor(t){super(),ko(this,t,So,Qo,vo,{})}}export{ss as component};
