import{s as Dt,o as At,n as Be}from"../chunks/scheduler.18a86fab.js";import{S as Kt,i as Ot,g as d,s,r as w,A as eo,h as p,f as n,c as r,j as V,x as h,u as y,k as F,l as to,y as l,a as i,v as b,d as _,t as M,w as v}from"../chunks/index.98837b22.js";import{T as Et}from"../chunks/Tip.77304350.js";import{D as ue}from"../chunks/Docstring.a1ef7999.js";import{C as Ie}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Yt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as xe,E as oo}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as no,a as so}from"../chunks/HfOption.6641485e.js";function ro(Q){let t,f="Click on the ColQwen2 models in the right sidebar for more examples of how to use ColQwen2 for image retrieval.";return{c(){t=d("p"),t.textContent=f},l(a){t=p(a,"P",{"data-svelte-h":!0}),h(t)!=="svelte-2qyslm"&&(t.textContent=f)},m(a,m){i(a,t,m)},p:Be,d(a){a&&n(t)}}}function ao(Q){let t,f,a,m="If you have issue with loading the images with PIL, you can use the following code to create dummy images:",g,c,J;return t=new Ie({props:{code:"aW1wb3J0JTIwcmVxdWVzdHMlMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQSUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBDb2xRd2VuMkZvclJldHJpZXZhbCUyQyUyMENvbFF3ZW4yUHJvY2Vzc29yJTBBZnJvbSUyMHRyYW5zZm9ybWVycy51dGlscy5pbXBvcnRfdXRpbHMlMjBpbXBvcnQlMjBpc19mbGFzaF9hdHRuXzJfYXZhaWxhYmxlJTBBJTBBJTBBJTIzJTIwTG9hZCUyMHRoZSUyMG1vZGVsJTIwYW5kJTIwdGhlJTIwcHJvY2Vzc29yJTBBbW9kZWxfbmFtZSUyMCUzRCUyMCUyMnZpZG9yZSUyRmNvbHF3ZW4yLXYxLjAtaGYlMjIlMEElMEFtb2RlbCUyMCUzRCUyMENvbFF3ZW4yRm9yUmV0cmlldmFsLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjBtb2RlbF9uYW1lJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTIwJTIwJTIzJTIwJTIyY3B1JTIyJTJDJTIwJTIyY3VkYSUyMiUyQyUyMCUyMnhwdSUyMiUyMG9yJTIwJTIybXBzJTIyJTIwZm9yJTIwQXBwbGUlMjBTaWxpY29uJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMmZsYXNoX2F0dGVudGlvbl8yJTIyJTIwaWYlMjBpc19mbGFzaF9hdHRuXzJfYXZhaWxhYmxlKCklMjBlbHNlJTIwJTIyc2RwYSUyMiUyQyUwQSklMEFwcm9jZXNzb3IlMjAlM0QlMjBDb2xRd2VuMlByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQobW9kZWxfbmFtZSklMEElMEElMjMlMjBUaGUlMjBkb2N1bWVudCUyMHBhZ2UlMjBzY3JlZW5zaG90cyUyMGZyb20lMjB5b3VyJTIwY29ycHVzJTBBdXJsMSUyMCUzRCUyMCUyMmh0dHBzJTNBJTJGJTJGdXBsb2FkLndpa2ltZWRpYS5vcmclMkZ3aWtpcGVkaWElMkZjb21tb25zJTJGOCUyRjg5JTJGVVMtb3JpZ2luYWwtRGVjbGFyYXRpb24tMTc3Ni5qcGclMjIlMEF1cmwyJTIwJTNEJTIwJTIyaHR0cHMlM0ElMkYlMkZ1cGxvYWQud2lraW1lZGlhLm9yZyUyRndpa2lwZWRpYSUyRmNvbW1vbnMlMkZ0aHVtYiUyRjQlMkY0YyUyRlJvbWVvYW5kanVsaWV0MTU5Ny5qcGclMkY1MDBweC1Sb21lb2FuZGp1bGlldDE1OTcuanBnJTIyJTBBJTBBaW1hZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwSW1hZ2Uub3BlbihyZXF1ZXN0cy5nZXQodXJsMSUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMkMlMEElMjAlMjAlMjAlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwyJTJDJTIwc3RyZWFtJTNEVHJ1ZSkucmF3KSUyQyUwQSU1RCUwQSUwQSUyMyUyMFRoZSUyMHF1ZXJpZXMlMjB5b3UlMjB3YW50JTIwdG8lMjByZXRyaWV2ZSUyMGRvY3VtZW50cyUyMGZvciUwQXF1ZXJpZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlMjJXaGVuJTIwd2FzJTIwdGhlJTIwVW5pdGVkJTIwU3RhdGVzJTIwRGVjbGFyYXRpb24lMjBvZiUyMEluZGVwZW5kZW5jZSUyMHByb2NsYWltZWQlM0YlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjJXaG8lMjBwcmludGVkJTIwdGhlJTIwZWRpdGlvbiUyMG9mJTIwUm9tZW8lMjBhbmQlMjBKdWxpZXQlM0YlMjIlMkMlMEElNUQlMEElMEElMjMlMjBQcm9jZXNzJTIwdGhlJTIwaW5wdXRzJTBBaW5wdXRzX2ltYWdlcyUyMCUzRCUyMHByb2Nlc3NvcihpbWFnZXMlM0RpbWFnZXMpLnRvKG1vZGVsLmRldmljZSklMEFpbnB1dHNfdGV4dCUyMCUzRCUyMHByb2Nlc3Nvcih0ZXh0JTNEcXVlcmllcykudG8obW9kZWwuZGV2aWNlKSUwQSUwQSUyMyUyMEZvcndhcmQlMjBwYXNzJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGltYWdlX2VtYmVkZGluZ3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0c19pbWFnZXMpLmVtYmVkZGluZ3MlMEElMjAlMjAlMjAlMjBxdWVyeV9lbWJlZGRpbmdzJTIwJTNEJTIwbW9kZWwoKippbnB1dHNfdGV4dCkuZW1iZWRkaW5ncyUwQSUwQSUyMyUyMFNjb3JlJTIwdGhlJTIwcXVlcmllcyUyMGFnYWluc3QlMjB0aGUlMjBpbWFnZXMlMEFzY29yZXMlMjAlM0QlMjBwcm9jZXNzb3Iuc2NvcmVfcmV0cmlldmFsKHF1ZXJ5X2VtYmVkZGluZ3MlMkMlMjBpbWFnZV9lbWJlZGRpbmdzKSUwQSUwQXByaW50KCUyMlJldHJpZXZhbCUyMHNjb3JlcyUyMChxdWVyeSUyMHglMjBpbWFnZSklM0ElMjIpJTBBcHJpbnQoc2NvcmVzKQ==",highlighted:`<span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ColQwen2ForRetrieval, ColQwen2Processor
<span class="hljs-keyword">from</span> transformers.utils.import_utils <span class="hljs-keyword">import</span> is_flash_attn_2_available


<span class="hljs-comment"># Load the model and the processor</span>
model_name = <span class="hljs-string">&quot;vidore/colqwen2-v1.0-hf&quot;</span>

model = ColQwen2ForRetrieval.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,  <span class="hljs-comment"># &quot;cpu&quot;, &quot;cuda&quot;, &quot;xpu&quot; or &quot;mps&quot; for Apple Silicon</span>
    attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span> <span class="hljs-keyword">if</span> is_flash_attn_2_available() <span class="hljs-keyword">else</span> <span class="hljs-string">&quot;sdpa&quot;</span>,
)
processor = ColQwen2Processor.from_pretrained(model_name)

<span class="hljs-comment"># The document page screenshots from your corpus</span>
url1 = <span class="hljs-string">&quot;https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg&quot;</span>
url2 = <span class="hljs-string">&quot;https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg&quot;</span>

images = [
    Image.<span class="hljs-built_in">open</span>(requests.get(url1, stream=<span class="hljs-literal">True</span>).raw),
    Image.<span class="hljs-built_in">open</span>(requests.get(url2, stream=<span class="hljs-literal">True</span>).raw),
]

<span class="hljs-comment"># The queries you want to retrieve documents for</span>
queries = [
    <span class="hljs-string">&quot;When was the United States Declaration of Independence proclaimed?&quot;</span>,
    <span class="hljs-string">&quot;Who printed the edition of Romeo and Juliet?&quot;</span>,
]

<span class="hljs-comment"># Process the inputs</span>
inputs_images = processor(images=images).to(model.device)
inputs_text = processor(text=queries).to(model.device)

<span class="hljs-comment"># Forward pass</span>
<span class="hljs-keyword">with</span> torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

<span class="hljs-comment"># Score the queries against the images</span>
scores = processor.score_retrieval(query_embeddings, image_embeddings)

<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Retrieval scores (query x image):&quot;</span>)
<span class="hljs-built_in">print</span>(scores)`,wrap:!1}}),c=new Ie({props:{code:"aW1hZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwSW1hZ2UubmV3KCUyMlJHQiUyMiUyQyUyMCgxMjglMkMlMjAxMjgpJTJDJTIwY29sb3IlM0QlMjJ3aGl0ZSUyMiklMkMlMEElMjAlMjAlMjAlMjBJbWFnZS5uZXcoJTIyUkdCJTIyJTJDJTIwKDY0JTJDJTIwMzIpJTJDJTIwY29sb3IlM0QlMjJibGFjayUyMiklMkMlMEElNUQ=",highlighted:`images = [
    Image.new(<span class="hljs-string">&quot;RGB&quot;</span>, (<span class="hljs-number">128</span>, <span class="hljs-number">128</span>), color=<span class="hljs-string">&quot;white&quot;</span>),
    Image.new(<span class="hljs-string">&quot;RGB&quot;</span>, (<span class="hljs-number">64</span>, <span class="hljs-number">32</span>), color=<span class="hljs-string">&quot;black&quot;</span>),
]`,wrap:!1}}),{c(){w(t.$$.fragment),f=s(),a=d("p"),a.textContent=m,g=s(),w(c.$$.fragment)},l(u){y(t.$$.fragment,u),f=r(u),a=p(u,"P",{"data-svelte-h":!0}),h(a)!=="svelte-19pwz5p"&&(a.textContent=m),g=r(u),y(c.$$.fragment,u)},m(u,z){b(t,u,z),i(u,f,z),i(u,a,z),i(u,g,z),b(c,u,z),J=!0},p:Be,i(u){J||(_(t.$$.fragment,u),_(c.$$.fragment,u),J=!0)},o(u){M(t.$$.fragment,u),M(c.$$.fragment,u),J=!1},d(u){u&&(n(f),n(a),n(g)),v(t,u),v(c,u)}}}function lo(Q){let t,f;return t=new so({props:{id:"usage",option:"image retrieval",$$slots:{default:[ao]},$$scope:{ctx:Q}}}),{c(){w(t.$$.fragment)},l(a){y(t.$$.fragment,a)},m(a,m){b(t,a,m),f=!0},p(a,m){const g={};m&2&&(g.$$scope={dirty:m,ctx:a}),t.$set(g)},i(a){f||(_(t.$$.fragment,a),f=!0)},o(a){M(t.$$.fragment,a),f=!1},d(a){v(t,a)}}}function io(Q){let t,f="Example:",a,m,g;return m=new Ie({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycy5tb2RlbHMuY29scXdlbjIlMjBpbXBvcnQlMjBDb2xRd2VuMkNvbmZpZyUyQyUyMENvbFF3ZW4yRm9yUmV0cmlldmFsJTBBJTBBY29uZmlnJTIwJTNEJTIwQ29sUXdlbjJDb25maWcoKSUwQW1vZGVsJTIwJTNEJTIwQ29sUXdlbjJGb3JSZXRyaWV2YWwoY29uZmlnKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers.models.colqwen2 <span class="hljs-keyword">import</span> ColQwen2Config, ColQwen2ForRetrieval

config = ColQwen2Config()
model = ColQwen2ForRetrieval(config)`,wrap:!1}}),{c(){t=d("p"),t.textContent=f,a=s(),w(m.$$.fragment)},l(c){t=p(c,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=f),a=r(c),y(m.$$.fragment,c)},m(c,J){i(c,t,J),i(c,a,J),b(m,c,J),g=!0},p:Be,i(c){g||(_(m.$$.fragment,c),g=!0)},o(c){M(m.$$.fragment,c),g=!1},d(c){c&&(n(t),n(a)),v(m,c)}}}function co(Q){let t,f=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=f},l(a){t=p(a,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=f)},m(a,m){i(a,t,m)},p:Be,d(a){a&&n(t)}}}function po(Q){let t,f="Example:",a,m,g;return m=new Ie({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=d("p"),t.textContent=f,a=s(),w(m.$$.fragment)},l(c){t=p(c,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=f),a=r(c),y(m.$$.fragment,c)},m(c,J){i(c,t,J),i(c,a,J),b(m,c,J),g=!0},p:Be,i(c){g||(_(m.$$.fragment,c),g=!0)},o(c){M(m.$$.fragment,c),g=!1},d(c){c&&(n(t),n(a)),v(m,c)}}}function mo(Q){let t,f,a,m,g,c="<em>This model was released on 2024-06-27 and added to Hugging Face Transformers on 2025-06-02.</em>",J,u,z='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',We,S,Re,E,Ct='<a href="https://huggingface.co/papers/2407.01449" rel="nofollow">ColQwen2</a> is a variant of the <a href="./colpali">ColPali</a> model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColQwen2 treats each page as an image. It uses the <a href="./qwen2_vl">Qwen2-VL</a> backbone to capture not only text, but also the layout, tables, charts, and other visual elements to create detailed multi-vector embeddings that can be used for retrieval by computing pairwise late interaction similarity scores. This offers a more comprehensive understanding of documents and enables more efficient and accurate retrieval.',qe,Y,Ut='This model was contributed by <a href="https://huggingface.co/tonywu71" rel="nofollow">@tonywu71</a> (ILLUIN Technology) and <a href="https://huggingface.co/yonigozlan" rel="nofollow">@yonigozlan</a> (HuggingFace).',Ve,D,jt='You can find all the original ColPali checkpoints under Vidore’s <a href="https://huggingface.co/collections/vidore/hf-native-colvision-models-6755d68fc60a8553acaa96f7" rel="nofollow">Hf-native ColVision Models</a> collection.',Fe,H,ze,X,He,A,Qt='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Xe,K,$t='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to quantize the weights to int4.',Pe,O,Ge,ee,Ne,te,kt='<li><a href="/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2Processor.score_retrieval">score_retrieval()</a> returns a 2D tensor where the first dimension is the number of queries and the second dimension is the number of images. A higher score indicates more similarity between the query and image.</li> <li>Unlike ColPali, ColQwen2 supports arbitrary image resolutions and aspect ratios, which means images are not resized into fixed-size squares. This preserves more of the original input signal.</li> <li>Larger input images generate longer multi-vector embeddings, allowing users to adjust image resolution to balance performance and memory usage.</li>',Le,oe,Se,U,ne,tt,he,Zt=`Configuration class to store the configuration of a <code>ColQ2en2ForRetrieval</code>. It is used to instantiate an instance
of <code>ColQwen2ForRetrieval</code> according to the specified arguments, defining the model architecture following the methodology
from the “ColPali: Efficient Document Retrieval with Vision Language Models” paper.`,ot,fe,xt=`Instantiating a configuration with the defaults will yield a similar configuration to the vision encoder used by the pre-trained
ColQwen2-v1.0 model, e.g. <a href="https://huggingface.co/vidore/colqwen2-v1.0-hf" rel="nofollow">vidore/colqwen2-v1.0-hf</a>.`,nt,ge,It=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,st,P,Ee,se,Ye,C,re,rt,we,Bt=`Constructs a ColQwen2 processor which wraps a Qwen2VLProcessor and special methods to process images and queries, as
well as to compute the late-interaction retrieval score.`,at,ye,Wt=`<a href="/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2Processor">ColQwen2Processor</a> offers all the functionalities of <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLProcessor">Qwen2VLProcessor</a>. See the <code>__call__()</code>
for more information.`,lt,x,ae,it,be,Rt=`Prepare for the model one or several image(s). This method is a wrapper around the <code>__call__</code> method of the ColQwen2Processor’s
<code>ColQwen2Processor.__call__()</code>.`,ct,_e,qt="This method forwards the <code>images</code> and <code>kwargs</code> arguments to the image processor.",dt,I,le,pt,Me,Vt=`Prepare for the model one or several texts. This method is a wrapper around the <code>__call__</code> method of the ColQwen2Processor’s
<code>ColQwen2Processor.__call__()</code>.`,mt,ve,Ft="This method forwards the <code>text</code> and <code>kwargs</code> arguments to the tokenizer.",ut,B,ie,ht,Te,zt=`Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
query embeddings (<code>qs</code>) and passage embeddings (<code>ps</code>). For ColQwen2, a passage is the
image of a document page.`,ft,Je,Ht=`Because the embedding tensors are multi-vector and can thus have different shapes, they
should be fed as:
(1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
(2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
obtained by padding the list of tensors.`,De,ce,Ae,T,de,gt,Ce,Xt=`Following the ColPali approach, ColQwen2 leverages VLMs to construct efficient multi-vector embeddings directly
from document images (“screenshots”) for document retrieval. The model is trained to maximize the similarity
between these document embeddings and the corresponding query embeddings, using the late interaction method
introduced in ColBERT.`,wt,Ue,Pt=`Using ColQwen2 removes the need for potentially complex and brittle layout recognition and OCR pipelines with
a single model that can take into account both the textual and visual content (layout, charts, …) of a document.`,yt,je,Gt=`ColQwen2 is part of the ColVision model family, which was introduced with ColPali in the following paper:
<a href="https://huggingface.co/papers/2407.01449" rel="nofollow"><em>ColPali: Efficient Document Retrieval with Vision Language Models</em></a>.`,bt,Qe,Nt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_t,$e,Lt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Mt,Z,pe,vt,ke,St='The <a href="/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2ForRetrieval">ColQwen2ForRetrieval</a> forward method, overrides the <code>__call__</code> special method.',Tt,G,Jt,N,Ke,me,Oe,Ze,et;return S=new xe({props:{title:"ColQwen2",local:"colqwen2",headingTag:"h1"}}),H=new Et({props:{warning:!1,$$slots:{default:[ro]},$$scope:{ctx:Q}}}),X=new no({props:{id:"usage",options:["image retrieval"],$$slots:{default:[lo]},$$scope:{ctx:Q}}}),O=new Ie({props:{code:"aW1wb3J0JTIwcmVxdWVzdHMlMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQSUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBCaXRzQW5kQnl0ZXNDb25maWclMkMlMjBDb2xRd2VuMkZvclJldHJpZXZhbCUyQyUyMENvbFF3ZW4yUHJvY2Vzc29yJTJDJTIwaW5mZXJfZGV2aWNlJTBBJTBBbW9kZWxfbmFtZSUyMCUzRCUyMCUyMnZpZG9yZSUyRmNvbHF3ZW4yLXYxLjAtaGYlMjIlMEFkZXZpY2UlMjAlM0QlMjBpbmZlcl9kZXZpY2UoKSUwQSUwQSUyMyUyMDQtYml0JTIwcXVhbnRpemF0aW9uJTIwY29uZmlndXJhdGlvbiUwQWJuYl9jb25maWclMjAlM0QlMjBCaXRzQW5kQnl0ZXNDb25maWcoJTBBJTIwJTIwJTIwJTIwbG9hZF9pbl80Yml0JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBKSUwQSUwQW1vZGVsJTIwJTNEJTIwQ29sUXdlbjJGb3JSZXRyaWV2YWwuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMG1vZGVsX25hbWUlMkMlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEYm5iX2NvbmZpZyUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0RkZXZpY2UlMkMlMEEpLmV2YWwoKSUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMENvbFF3ZW4yUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQSUwQXVybDElMjAlM0QlMjAlMjJodHRwcyUzQSUyRiUyRnVwbG9hZC53aWtpbWVkaWEub3JnJTJGd2lraXBlZGlhJTJGY29tbW9ucyUyRjglMkY4OSUyRlVTLW9yaWdpbmFsLURlY2xhcmF0aW9uLTE3NzYuanBnJTIyJTBBdXJsMiUyMCUzRCUyMCUyMmh0dHBzJTNBJTJGJTJGdXBsb2FkLndpa2ltZWRpYS5vcmclMkZ3aWtpcGVkaWElMkZjb21tb25zJTJGdGh1bWIlMkY0JTJGNGMlMkZSb21lb2FuZGp1bGlldDE1OTcuanBnJTJGNTAwcHgtUm9tZW9hbmRqdWxpZXQxNTk3LmpwZyUyMiUwQSUwQWltYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMEltYWdlLm9wZW4ocmVxdWVzdHMuZ2V0KHVybDElMkMlMjBzdHJlYW0lM0RUcnVlKS5yYXcpJTJDJTBBJTIwJTIwJTIwJTIwSW1hZ2Uub3BlbihyZXF1ZXN0cy5nZXQodXJsMiUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMkMlMEElNUQlMEElMEFxdWVyaWVzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIyV2hlbiUyMHdhcyUyMHRoZSUyMFVuaXRlZCUyMFN0YXRlcyUyMERlY2xhcmF0aW9uJTIwb2YlMjBJbmRlcGVuZGVuY2UlMjBwcm9jbGFpbWVkJTNGJTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIyV2hvJTIwcHJpbnRlZCUyMHRoZSUyMGVkaXRpb24lMjBvZiUyMFJvbWVvJTIwYW5kJTIwSnVsaWV0JTNGJTIyJTJDJTBBJTVEJTBBJTBBJTIzJTIwUHJvY2VzcyUyMHRoZSUyMGlucHV0cyUwQWlucHV0c19pbWFnZXMlMjAlM0QlMjBwcm9jZXNzb3IoaW1hZ2VzJTNEaW1hZ2VzJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQWlucHV0c190ZXh0JTIwJTNEJTIwcHJvY2Vzc29yKHRleHQlM0RxdWVyaWVzJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQSUyMyUyMEZvcndhcmQlMjBwYXNzJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGltYWdlX2VtYmVkZGluZ3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0c19pbWFnZXMpLmVtYmVkZGluZ3MlMEElMjAlMjAlMjAlMjBxdWVyeV9lbWJlZGRpbmdzJTIwJTNEJTIwbW9kZWwoKippbnB1dHNfdGV4dCkuZW1iZWRkaW5ncyUwQSUwQSUyMyUyMFNjb3JlJTIwdGhlJTIwcXVlcmllcyUyMGFnYWluc3QlMjB0aGUlMjBpbWFnZXMlMEFzY29yZXMlMjAlM0QlMjBwcm9jZXNzb3Iuc2NvcmVfcmV0cmlldmFsKHF1ZXJ5X2VtYmVkZGluZ3MlMkMlMjBpbWFnZV9lbWJlZGRpbmdzKSUwQSUwQXByaW50KCUyMlJldHJpZXZhbCUyMHNjb3JlcyUyMChxdWVyeSUyMHglMjBpbWFnZSklM0ElMjIpJTBBcHJpbnQoc2NvcmVzKQ==",highlighted:`<span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, ColQwen2ForRetrieval, ColQwen2Processor, infer_device

model_name = <span class="hljs-string">&quot;vidore/colqwen2-v1.0-hf&quot;</span>
device = infer_device()

<span class="hljs-comment"># 4-bit quantization configuration</span>
bnb_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_compute_dtype=torch.float16,
)

model = ColQwen2ForRetrieval.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device,
).<span class="hljs-built_in">eval</span>()

processor = ColQwen2Processor.from_pretrained(model_name)

url1 = <span class="hljs-string">&quot;https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg&quot;</span>
url2 = <span class="hljs-string">&quot;https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg&quot;</span>

images = [
    Image.<span class="hljs-built_in">open</span>(requests.get(url1, stream=<span class="hljs-literal">True</span>).raw),
    Image.<span class="hljs-built_in">open</span>(requests.get(url2, stream=<span class="hljs-literal">True</span>).raw),
]

queries = [
    <span class="hljs-string">&quot;When was the United States Declaration of Independence proclaimed?&quot;</span>,
    <span class="hljs-string">&quot;Who printed the edition of Romeo and Juliet?&quot;</span>,
]

<span class="hljs-comment"># Process the inputs</span>
inputs_images = processor(images=images, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
inputs_text = processor(text=queries, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-comment"># Forward pass</span>
<span class="hljs-keyword">with</span> torch.no_grad():
    image_embeddings = model(**inputs_images).embeddings
    query_embeddings = model(**inputs_text).embeddings

<span class="hljs-comment"># Score the queries against the images</span>
scores = processor.score_retrieval(query_embeddings, image_embeddings)

<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;Retrieval scores (query x image):&quot;</span>)
<span class="hljs-built_in">print</span>(scores)`,wrap:!1}}),ee=new xe({props:{title:"Notes",local:"notes",headingTag:"h2"}}),oe=new xe({props:{title:"ColQwen2Config",local:"transformers.ColQwen2Config",headingTag:"h2"}}),ne=new ue({props:{name:"class transformers.ColQwen2Config",anchor:"transformers.ColQwen2Config",parameters:[{name:"vlm_config",val:" = None"},{name:"embedding_dim",val:": int = 128"},{name:"initializer_range",val:": float = 0.02"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ColQwen2Config.vlm_config",description:`<strong>vlm_config</strong> (<code>PretrainedConfig</code>, <em>optional</em>) &#x2014;
Configuration of the VLM backbone model.`,name:"vlm_config"},{anchor:"transformers.ColQwen2Config.embedding_dim",description:`<strong>embedding_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Dimension of the multi-vector embeddings produced by the model.`,name:"embedding_dim"},{anchor:"transformers.ColQwen2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colqwen2/configuration_colqwen2.py#L27"}}),P=new Yt({props:{anchor:"transformers.ColQwen2Config.example",$$slots:{default:[io]},$$scope:{ctx:Q}}}),se=new xe({props:{title:"ColQwen2Processor",local:"transformers.ColQwen2Processor",headingTag:"h2"}}),re=new ue({props:{name:"class transformers.ColQwen2Processor",anchor:"transformers.ColQwen2Processor",parameters:[{name:"image_processor",val:" = None"},{name:"tokenizer",val:" = None"},{name:"chat_template",val:" = None"},{name:"visual_prompt_prefix",val:": typing.Optional[str] = None"},{name:"query_prefix",val:": typing.Optional[str] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ColQwen2Processor.image_processor",description:`<strong>image_processor</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor">Qwen2VLImageProcessor</a>, <em>optional</em>) &#x2014;
The image processor is a required input.`,name:"image_processor"},{anchor:"transformers.ColQwen2Processor.tokenizer",description:`<strong>tokenizer</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2TokenizerFast">Qwen2TokenizerFast</a>, <em>optional</em>) &#x2014;
The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.ColQwen2Processor.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014; A Jinja template which will be used to convert lists of messages
in a chat into a tokenizable string.`,name:"chat_template"},{anchor:"transformers.ColQwen2Processor.visual_prompt_prefix",description:"<strong>visual_prompt_prefix</strong> (<code>str</code>, <em>optional</em>) &#x2014; A string that gets tokenized and prepended to the image tokens.",name:"visual_prompt_prefix"},{anchor:"transformers.ColQwen2Processor.query_prefix",description:"<strong>query_prefix</strong> (<code>str</code>, <em>optional</em>) &#x2014; A prefix to be used for the query.",name:"query_prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colqwen2/processing_colqwen2.py#L48"}}),ae=new ue({props:{name:"process_images",anchor:"transformers.ColQwen2Processor.process_images",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.colqwen2.processing_colqwen2.ColQwen2ProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.ColQwen2Processor.process_images.images",description:`<strong>images</strong> (<code>PIL.Image.Image</code>, <code>np.ndarray</code>, <code>torch.Tensor</code>, <code>list[PIL.Image.Image]</code>, <code>list[np.ndarray]</code>, <code>list[torch.Tensor]</code>) &#x2014;
The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
number of channels, H and W are image height and width.`,name:"images"},{anchor:"transformers.ColQwen2Processor.process_images.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colqwen2/processing_colqwen2.py#L261",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>input_ids</strong> — List of token ids to be fed to a model.</li>
<li><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code> and if <code>text</code> is not
<code>None</code>).</li>
<li><strong>pixel_values</strong> — Pixel values to be fed to a model. Returned when <code>images</code> is not <code>None</code>.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),le=new ue({props:{name:"process_queries",anchor:"transformers.ColQwen2Processor.process_queries",parameters:[{name:"text",val:": typing.Union[str, list[str]]"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.colqwen2.processing_colqwen2.ColQwen2ProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.ColQwen2Processor.process_queries.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.ColQwen2Processor.process_queries.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colqwen2/processing_colqwen2.py#L296",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>input_ids</strong> — List of token ids to be fed to a model.</li>
<li><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code> and if <code>text</code> is not
<code>None</code>).</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),ie=new ue({props:{name:"score_retrieval",anchor:"transformers.ColQwen2Processor.score_retrieval",parameters:[{name:"query_embeddings",val:": typing.Union[ForwardRef('torch.Tensor'), list['torch.Tensor']]"},{name:"passage_embeddings",val:": typing.Union[ForwardRef('torch.Tensor'), list['torch.Tensor']]"},{name:"batch_size",val:": int = 128"},{name:"output_dtype",val:": typing.Optional[ForwardRef('torch.dtype')] = None"},{name:"output_device",val:": typing.Union[ForwardRef('torch.device'), str] = 'cpu'"}],parametersDescription:[{anchor:"transformers.ColQwen2Processor.score_retrieval.query_embeddings",description:"<strong>query_embeddings</strong> (<code>Union[torch.Tensor, list[torch.Tensor]</code>) &#x2014; Query embeddings.",name:"query_embeddings"},{anchor:"transformers.ColQwen2Processor.score_retrieval.passage_embeddings",description:"<strong>passage_embeddings</strong> (<code>Union[torch.Tensor, list[torch.Tensor]</code>) &#x2014; Passage embeddings.",name:"passage_embeddings"},{anchor:"transformers.ColQwen2Processor.score_retrieval.batch_size",description:"<strong>batch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014; Batch size for computing scores.",name:"batch_size"},{anchor:"transformers.ColQwen2Processor.score_retrieval.output_dtype",description:`<strong>output_dtype</strong> (<code>torch.dtype</code>, <em>optional</em>, defaults to <code>torch.float32</code>) &#x2014; The dtype of the output tensor.
If <code>None</code>, the dtype of the input embeddings is used.`,name:"output_dtype"},{anchor:"transformers.ColQwen2Processor.score_retrieval.output_device",description:"<strong>output_device</strong> (<code>torch.device</code> or <code>str</code>, <em>optional</em>, defaults to &#x201C;cpu&#x201D;) &#x2014; The device of the output tensor.",name:"output_device"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colqwen2/processing_colqwen2.py#L330",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A tensor of shape <code>(n_queries, n_passages)</code> containing the scores. The score
tensor is saved on the “cpu” device.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>torch.Tensor</code></p>
`}}),ce=new xe({props:{title:"ColQwen2ForRetrieval",local:"transformers.ColQwen2ForRetrieval",headingTag:"h2"}}),de=new ue({props:{name:"class transformers.ColQwen2ForRetrieval",anchor:"transformers.ColQwen2ForRetrieval",parameters:[{name:"config",val:": ColQwen2Config"}],parametersDescription:[{anchor:"transformers.ColQwen2ForRetrieval.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2Config">ColQwen2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colqwen2/modeling_colqwen2.py#L106"}}),pe=new ue({props:{name:"forward",anchor:"transformers.ColQwen2ForRetrieval.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"pixel_values",val:": typing.Optional[torch.Tensor] = None"},{name:"image_grid_thw",val:": typing.Optional[torch.LongTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.ColQwen2ForRetrieval.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ColQwen2ForRetrieval.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ColQwen2ForRetrieval.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ColQwen2ForRetrieval.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ColQwen2ForRetrieval.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.ColQwen2ForRetrieval.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ColQwen2ForRetrieval.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ColQwen2ForRetrieval.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ColQwen2ForRetrieval.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ColQwen2ForRetrieval.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ColQwen2ForRetrieval.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<code>image_processor_class</code>. See <code>image_processor_class.__call__</code> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2Processor">ColQwen2Processor</a> uses
<code>image_processor_class</code> for processing images).`,name:"pixel_values"},{anchor:"transformers.ColQwen2ForRetrieval.forward.image_grid_thw",description:`<strong>image_grid_thw</strong> (<code>torch.LongTensor</code> of shape <code>(num_images, 3)</code>, <em>optional</em>) &#x2014;
The temporal, height and width of feature shape of each image in LLM.`,name:"image_grid_thw"},{anchor:"transformers.ColQwen2ForRetrieval.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colqwen2/modeling_colqwen2.py#L125",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.colqwen2.modeling_colqwen2.ColQwen2ForRetrievalOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/colqwen2#transformers.ColQwen2Config"
>ColQwen2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>embeddings</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — The embeddings of the model.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.colqwen2.modeling_colqwen2.ColQwen2ForRetrievalOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),G=new Et({props:{$$slots:{default:[co]},$$scope:{ctx:Q}}}),N=new Yt({props:{anchor:"transformers.ColQwen2ForRetrieval.forward.example",$$slots:{default:[po]},$$scope:{ctx:Q}}}),me=new oo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/colqwen2.md"}}),{c(){t=d("meta"),f=s(),a=d("p"),m=s(),g=d("p"),g.innerHTML=c,J=s(),u=d("div"),u.innerHTML=z,We=s(),w(S.$$.fragment),Re=s(),E=d("p"),E.innerHTML=Ct,qe=s(),Y=d("p"),Y.innerHTML=Ut,Ve=s(),D=d("p"),D.innerHTML=jt,Fe=s(),w(H.$$.fragment),ze=s(),w(X.$$.fragment),He=s(),A=d("p"),A.innerHTML=Qt,Xe=s(),K=d("p"),K.innerHTML=$t,Pe=s(),w(O.$$.fragment),Ge=s(),w(ee.$$.fragment),Ne=s(),te=d("ul"),te.innerHTML=kt,Le=s(),w(oe.$$.fragment),Se=s(),U=d("div"),w(ne.$$.fragment),tt=s(),he=d("p"),he.innerHTML=Zt,ot=s(),fe=d("p"),fe.innerHTML=xt,nt=s(),ge=d("p"),ge.innerHTML=It,st=s(),w(P.$$.fragment),Ee=s(),w(se.$$.fragment),Ye=s(),C=d("div"),w(re.$$.fragment),rt=s(),we=d("p"),we.textContent=Bt,at=s(),ye=d("p"),ye.innerHTML=Wt,lt=s(),x=d("div"),w(ae.$$.fragment),it=s(),be=d("p"),be.innerHTML=Rt,ct=s(),_e=d("p"),_e.innerHTML=qt,dt=s(),I=d("div"),w(le.$$.fragment),pt=s(),Me=d("p"),Me.innerHTML=Vt,mt=s(),ve=d("p"),ve.innerHTML=Ft,ut=s(),B=d("div"),w(ie.$$.fragment),ht=s(),Te=d("p"),Te.innerHTML=zt,ft=s(),Je=d("p"),Je.textContent=Ht,De=s(),w(ce.$$.fragment),Ae=s(),T=d("div"),w(de.$$.fragment),gt=s(),Ce=d("p"),Ce.textContent=Xt,wt=s(),Ue=d("p"),Ue.textContent=Pt,yt=s(),je=d("p"),je.innerHTML=Gt,bt=s(),Qe=d("p"),Qe.innerHTML=Nt,_t=s(),$e=d("p"),$e.innerHTML=Lt,Mt=s(),Z=d("div"),w(pe.$$.fragment),vt=s(),ke=d("p"),ke.innerHTML=St,Tt=s(),w(G.$$.fragment),Jt=s(),w(N.$$.fragment),Ke=s(),w(me.$$.fragment),Oe=s(),Ze=d("p"),this.h()},l(e){const o=eo("svelte-u9bgzb",document.head);t=p(o,"META",{name:!0,content:!0}),o.forEach(n),f=r(e),a=p(e,"P",{}),V(a).forEach(n),m=r(e),g=p(e,"P",{"data-svelte-h":!0}),h(g)!=="svelte-1qp8gja"&&(g.innerHTML=c),J=r(e),u=p(e,"DIV",{style:!0,"data-svelte-h":!0}),h(u)!=="svelte-wa5t4p"&&(u.innerHTML=z),We=r(e),y(S.$$.fragment,e),Re=r(e),E=p(e,"P",{"data-svelte-h":!0}),h(E)!=="svelte-1dywzpv"&&(E.innerHTML=Ct),qe=r(e),Y=p(e,"P",{"data-svelte-h":!0}),h(Y)!=="svelte-1usq28k"&&(Y.innerHTML=Ut),Ve=r(e),D=p(e,"P",{"data-svelte-h":!0}),h(D)!=="svelte-st4jp7"&&(D.innerHTML=jt),Fe=r(e),y(H.$$.fragment,e),ze=r(e),y(X.$$.fragment,e),He=r(e),A=p(e,"P",{"data-svelte-h":!0}),h(A)!=="svelte-nf5ooi"&&(A.innerHTML=Qt),Xe=r(e),K=p(e,"P",{"data-svelte-h":!0}),h(K)!=="svelte-x2dyjs"&&(K.innerHTML=$t),Pe=r(e),y(O.$$.fragment,e),Ge=r(e),y(ee.$$.fragment,e),Ne=r(e),te=p(e,"UL",{"data-svelte-h":!0}),h(te)!=="svelte-5x9mme"&&(te.innerHTML=kt),Le=r(e),y(oe.$$.fragment,e),Se=r(e),U=p(e,"DIV",{class:!0});var k=V(U);y(ne.$$.fragment,k),tt=r(k),he=p(k,"P",{"data-svelte-h":!0}),h(he)!=="svelte-13u0c3g"&&(he.innerHTML=Zt),ot=r(k),fe=p(k,"P",{"data-svelte-h":!0}),h(fe)!=="svelte-s1jvyl"&&(fe.innerHTML=xt),nt=r(k),ge=p(k,"P",{"data-svelte-h":!0}),h(ge)!=="svelte-1ek1ss9"&&(ge.innerHTML=It),st=r(k),y(P.$$.fragment,k),k.forEach(n),Ee=r(e),y(se.$$.fragment,e),Ye=r(e),C=p(e,"DIV",{class:!0});var j=V(C);y(re.$$.fragment,j),rt=r(j),we=p(j,"P",{"data-svelte-h":!0}),h(we)!=="svelte-1byksx7"&&(we.textContent=Bt),at=r(j),ye=p(j,"P",{"data-svelte-h":!0}),h(ye)!=="svelte-ai5fk4"&&(ye.innerHTML=Wt),lt=r(j),x=p(j,"DIV",{class:!0});var W=V(x);y(ae.$$.fragment,W),it=r(W),be=p(W,"P",{"data-svelte-h":!0}),h(be)!=="svelte-zcwhrr"&&(be.innerHTML=Rt),ct=r(W),_e=p(W,"P",{"data-svelte-h":!0}),h(_e)!=="svelte-1at6zp"&&(_e.innerHTML=qt),W.forEach(n),dt=r(j),I=p(j,"DIV",{class:!0});var R=V(I);y(le.$$.fragment,R),pt=r(R),Me=p(R,"P",{"data-svelte-h":!0}),h(Me)!=="svelte-1dfsyqi"&&(Me.innerHTML=Vt),mt=r(R),ve=p(R,"P",{"data-svelte-h":!0}),h(ve)!=="svelte-1y7x5ok"&&(ve.innerHTML=Ft),R.forEach(n),ut=r(j),B=p(j,"DIV",{class:!0});var q=V(B);y(ie.$$.fragment,q),ht=r(q),Te=p(q,"P",{"data-svelte-h":!0}),h(Te)!=="svelte-12e1v66"&&(Te.innerHTML=zt),ft=r(q),Je=p(q,"P",{"data-svelte-h":!0}),h(Je)!=="svelte-3nd4tx"&&(Je.textContent=Ht),q.forEach(n),j.forEach(n),De=r(e),y(ce.$$.fragment,e),Ae=r(e),T=p(e,"DIV",{class:!0});var $=V(T);y(de.$$.fragment,$),gt=r($),Ce=p($,"P",{"data-svelte-h":!0}),h(Ce)!=="svelte-zb0dff"&&(Ce.textContent=Xt),wt=r($),Ue=p($,"P",{"data-svelte-h":!0}),h(Ue)!=="svelte-1xlbnaa"&&(Ue.textContent=Pt),yt=r($),je=p($,"P",{"data-svelte-h":!0}),h(je)!=="svelte-zq1zk6"&&(je.innerHTML=Gt),bt=r($),Qe=p($,"P",{"data-svelte-h":!0}),h(Qe)!=="svelte-q52n56"&&(Qe.innerHTML=Nt),_t=r($),$e=p($,"P",{"data-svelte-h":!0}),h($e)!=="svelte-hswkmf"&&($e.innerHTML=Lt),Mt=r($),Z=p($,"DIV",{class:!0});var L=V(Z);y(pe.$$.fragment,L),vt=r(L),ke=p(L,"P",{"data-svelte-h":!0}),h(ke)!=="svelte-yp4ogi"&&(ke.innerHTML=St),Tt=r(L),y(G.$$.fragment,L),Jt=r(L),y(N.$$.fragment,L),L.forEach(n),$.forEach(n),Ke=r(e),y(me.$$.fragment,e),Oe=r(e),Ze=p(e,"P",{}),V(Ze).forEach(n),this.h()},h(){F(t,"name","hf:doc:metadata"),F(t,"content",uo),to(u,"float","right"),F(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){l(document.head,t),i(e,f,o),i(e,a,o),i(e,m,o),i(e,g,o),i(e,J,o),i(e,u,o),i(e,We,o),b(S,e,o),i(e,Re,o),i(e,E,o),i(e,qe,o),i(e,Y,o),i(e,Ve,o),i(e,D,o),i(e,Fe,o),b(H,e,o),i(e,ze,o),b(X,e,o),i(e,He,o),i(e,A,o),i(e,Xe,o),i(e,K,o),i(e,Pe,o),b(O,e,o),i(e,Ge,o),b(ee,e,o),i(e,Ne,o),i(e,te,o),i(e,Le,o),b(oe,e,o),i(e,Se,o),i(e,U,o),b(ne,U,null),l(U,tt),l(U,he),l(U,ot),l(U,fe),l(U,nt),l(U,ge),l(U,st),b(P,U,null),i(e,Ee,o),b(se,e,o),i(e,Ye,o),i(e,C,o),b(re,C,null),l(C,rt),l(C,we),l(C,at),l(C,ye),l(C,lt),l(C,x),b(ae,x,null),l(x,it),l(x,be),l(x,ct),l(x,_e),l(C,dt),l(C,I),b(le,I,null),l(I,pt),l(I,Me),l(I,mt),l(I,ve),l(C,ut),l(C,B),b(ie,B,null),l(B,ht),l(B,Te),l(B,ft),l(B,Je),i(e,De,o),b(ce,e,o),i(e,Ae,o),i(e,T,o),b(de,T,null),l(T,gt),l(T,Ce),l(T,wt),l(T,Ue),l(T,yt),l(T,je),l(T,bt),l(T,Qe),l(T,_t),l(T,$e),l(T,Mt),l(T,Z),b(pe,Z,null),l(Z,vt),l(Z,ke),l(Z,Tt),b(G,Z,null),l(Z,Jt),b(N,Z,null),i(e,Ke,o),b(me,e,o),i(e,Oe,o),i(e,Ze,o),et=!0},p(e,[o]){const k={};o&2&&(k.$$scope={dirty:o,ctx:e}),H.$set(k);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),X.$set(j);const W={};o&2&&(W.$$scope={dirty:o,ctx:e}),P.$set(W);const R={};o&2&&(R.$$scope={dirty:o,ctx:e}),G.$set(R);const q={};o&2&&(q.$$scope={dirty:o,ctx:e}),N.$set(q)},i(e){et||(_(S.$$.fragment,e),_(H.$$.fragment,e),_(X.$$.fragment,e),_(O.$$.fragment,e),_(ee.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(P.$$.fragment,e),_(se.$$.fragment,e),_(re.$$.fragment,e),_(ae.$$.fragment,e),_(le.$$.fragment,e),_(ie.$$.fragment,e),_(ce.$$.fragment,e),_(de.$$.fragment,e),_(pe.$$.fragment,e),_(G.$$.fragment,e),_(N.$$.fragment,e),_(me.$$.fragment,e),et=!0)},o(e){M(S.$$.fragment,e),M(H.$$.fragment,e),M(X.$$.fragment,e),M(O.$$.fragment,e),M(ee.$$.fragment,e),M(oe.$$.fragment,e),M(ne.$$.fragment,e),M(P.$$.fragment,e),M(se.$$.fragment,e),M(re.$$.fragment,e),M(ae.$$.fragment,e),M(le.$$.fragment,e),M(ie.$$.fragment,e),M(ce.$$.fragment,e),M(de.$$.fragment,e),M(pe.$$.fragment,e),M(G.$$.fragment,e),M(N.$$.fragment,e),M(me.$$.fragment,e),et=!1},d(e){e&&(n(f),n(a),n(m),n(g),n(J),n(u),n(We),n(Re),n(E),n(qe),n(Y),n(Ve),n(D),n(Fe),n(ze),n(He),n(A),n(Xe),n(K),n(Pe),n(Ge),n(Ne),n(te),n(Le),n(Se),n(U),n(Ee),n(Ye),n(C),n(De),n(Ae),n(T),n(Ke),n(Oe),n(Ze)),n(t),v(S,e),v(H,e),v(X,e),v(O,e),v(ee,e),v(oe,e),v(ne),v(P),v(se,e),v(re),v(ae),v(le),v(ie),v(ce,e),v(de),v(pe),v(G),v(N),v(me,e)}}}const uo='{"title":"ColQwen2","local":"colqwen2","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"ColQwen2Config","local":"transformers.ColQwen2Config","sections":[],"depth":2},{"title":"ColQwen2Processor","local":"transformers.ColQwen2Processor","sections":[],"depth":2},{"title":"ColQwen2ForRetrieval","local":"transformers.ColQwen2ForRetrieval","sections":[],"depth":2}],"depth":1}';function ho(Q){return At(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class To extends Kt{constructor(t){super(),Ot(this,t,ho,mo,Dt,{})}}export{To as component};
