import{s as Ot,o as eo,n as We}from"../chunks/scheduler.18a86fab.js";import{S as to,i as oo,g as c,s,r as M,A as no,h as d,f as n,c as r,j as G,x as u,u as y,k as V,l as so,y as a,a as i,v as b,d as _,t as v,w}from"../chunks/index.98837b22.js";import{T as Dt}from"../chunks/Tip.77304350.js";import{D as ue}from"../chunks/Docstring.a1ef7999.js";import{C as Re}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Kt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Be,E as ro}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as ao,a as lo}from"../chunks/HfOption.6641485e.js";function io(P){let t,h="Click on the ColPali models in the right sidebar for more examples of how to use ColPali for image retrieval.";return{c(){t=c("p"),t.textContent=h},l(l){t=d(l,"P",{"data-svelte-h":!0}),u(t)!=="svelte-19qg7yk"&&(t.textContent=h)},m(l,p){i(l,t,p)},p:We,d(l){l&&n(t)}}}function co(P){let t,h,l,p="If you have issue with loading the images with PIL, you can use the following code to create dummy images:",g,m,C;return t=new Re({props:{code:"aW1wb3J0JTIwcmVxdWVzdHMlMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQSUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBDb2xQYWxpRm9yUmV0cmlldmFsJTJDJTIwQ29sUGFsaVByb2Nlc3NvciUwQSUwQSUwQSUyMyUyMExvYWQlMjB0aGUlMjBtb2RlbCUyMGFuZCUyMHRoZSUyMHByb2Nlc3NvciUwQW1vZGVsX25hbWUlMjAlM0QlMjAlMjJ2aWRvcmUlMkZjb2xwYWxpLXYxLjMtaGYlMjIlMEElMEFtb2RlbCUyMCUzRCUyMENvbFBhbGlGb3JSZXRyaWV2YWwuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMG1vZGVsX25hbWUlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMjAlMjAlMjMlMjAlMjJjcHUlMjIlMkMlMjAlMjJjdWRhJTIyJTJDJTIwJTIyeHB1JTIyJTJDJTIwb3IlMjAlMjJtcHMlMjIlMjBmb3IlMjBBcHBsZSUyMFNpbGljb24lMEEpJTBBcHJvY2Vzc29yJTIwJTNEJTIwQ29sUGFsaVByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQobW9kZWxfbmFtZSklMEElMEElMjMlMjBUaGUlMjBkb2N1bWVudCUyMHBhZ2UlMjBzY3JlZW5zaG90cyUyMGZyb20lMjB5b3VyJTIwY29ycHVzJTBBdXJsMSUyMCUzRCUyMCUyMmh0dHBzJTNBJTJGJTJGdXBsb2FkLndpa2ltZWRpYS5vcmclMkZ3aWtpcGVkaWElMkZjb21tb25zJTJGOCUyRjg5JTJGVVMtb3JpZ2luYWwtRGVjbGFyYXRpb24tMTc3Ni5qcGclMjIlMEF1cmwyJTIwJTNEJTIwJTIyaHR0cHMlM0ElMkYlMkZ1cGxvYWQud2lraW1lZGlhLm9yZyUyRndpa2lwZWRpYSUyRmNvbW1vbnMlMkZ0aHVtYiUyRjQlMkY0YyUyRlJvbWVvYW5kanVsaWV0MTU5Ny5qcGclMkY1MDBweC1Sb21lb2FuZGp1bGlldDE1OTcuanBnJTIyJTBBJTBBaW1hZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwSW1hZ2Uub3BlbihyZXF1ZXN0cy5nZXQodXJsMSUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMkMlMEElMjAlMjAlMjAlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwyJTJDJTIwc3RyZWFtJTNEVHJ1ZSkucmF3KSUyQyUwQSU1RCUwQSUwQSUyMyUyMFRoZSUyMHF1ZXJpZXMlMjB5b3UlMjB3YW50JTIwdG8lMjByZXRyaWV2ZSUyMGRvY3VtZW50cyUyMGZvciUwQXF1ZXJpZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlMjJXaGVuJTIwd2FzJTIwdGhlJTIwVW5pdGVkJTIwU3RhdGVzJTIwRGVjbGFyYXRpb24lMjBvZiUyMEluZGVwZW5kZW5jZSUyMHByb2NsYWltZWQlM0YlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjJXaG8lMjBwcmludGVkJTIwdGhlJTIwZWRpdGlvbiUyMG9mJTIwUm9tZW8lMjBhbmQlMjBKdWxpZXQlM0YlMjIlMkMlMEElNUQlMEElMEElMjMlMjBQcm9jZXNzJTIwdGhlJTIwaW5wdXRzJTBBaW5wdXRzX2ltYWdlcyUyMCUzRCUyMHByb2Nlc3NvcihpbWFnZXMlM0RpbWFnZXMpLnRvKG1vZGVsLmRldmljZSklMEFpbnB1dHNfdGV4dCUyMCUzRCUyMHByb2Nlc3Nvcih0ZXh0JTNEcXVlcmllcykudG8obW9kZWwuZGV2aWNlKSUwQSUwQSUyMyUyMEZvcndhcmQlMjBwYXNzJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGltYWdlX2VtYmVkZGluZ3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0c19pbWFnZXMpLmVtYmVkZGluZ3MlMEElMjAlMjAlMjAlMjBxdWVyeV9lbWJlZGRpbmdzJTIwJTNEJTIwbW9kZWwoKippbnB1dHNfdGV4dCkuZW1iZWRkaW5ncyUwQSUwQSUyMyUyMFNjb3JlJTIwdGhlJTIwcXVlcmllcyUyMGFnYWluc3QlMjB0aGUlMjBpbWFnZXMlMEFzY29yZXMlMjAlM0QlMjBwcm9jZXNzb3Iuc2NvcmVfcmV0cmlldmFsKHF1ZXJ5X2VtYmVkZGluZ3MlMkMlMjBpbWFnZV9lbWJlZGRpbmdzKSUwQSUwQXByaW50KCUyMlJldHJpZXZhbCUyMHNjb3JlcyUyMChxdWVyeSUyMHglMjBpbWFnZSklM0ElMjIpJTBBcHJpbnQoc2NvcmVzKQ==",highlighted:`<span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ColPaliForRetrieval, ColPaliProcessor


<span class="hljs-comment"># Load the model and the processor</span>
model_name = <span class="hljs-string">&quot;vidore/colpali-v1.3-hf&quot;</span>

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,  <span class="hljs-comment"># &quot;cpu&quot;, &quot;cuda&quot;, &quot;xpu&quot;, or &quot;mps&quot; for Apple Silicon</span>
)
processor = ColPaliProcessor.from_pretrained(model_name)

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
<span class="hljs-built_in">print</span>(scores)`,wrap:!1}}),m=new Re({props:{code:"aW1hZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwSW1hZ2UubmV3KCUyMlJHQiUyMiUyQyUyMCgxMjglMkMlMjAxMjgpJTJDJTIwY29sb3IlM0QlMjJ3aGl0ZSUyMiklMkMlMEElMjAlMjAlMjAlMjBJbWFnZS5uZXcoJTIyUkdCJTIyJTJDJTIwKDY0JTJDJTIwMzIpJTJDJTIwY29sb3IlM0QlMjJibGFjayUyMiklMkMlMEElNUQ=",highlighted:`images = [
    Image.new(<span class="hljs-string">&quot;RGB&quot;</span>, (<span class="hljs-number">128</span>, <span class="hljs-number">128</span>), color=<span class="hljs-string">&quot;white&quot;</span>),
    Image.new(<span class="hljs-string">&quot;RGB&quot;</span>, (<span class="hljs-number">64</span>, <span class="hljs-number">32</span>), color=<span class="hljs-string">&quot;black&quot;</span>),
]`,wrap:!1}}),{c(){M(t.$$.fragment),h=s(),l=c("p"),l.textContent=p,g=s(),M(m.$$.fragment)},l(f){y(t.$$.fragment,f),h=r(f),l=d(f,"P",{"data-svelte-h":!0}),u(l)!=="svelte-19pwz5p"&&(l.textContent=p),g=r(f),y(m.$$.fragment,f)},m(f,q){b(t,f,q),i(f,h,q),i(f,l,q),i(f,g,q),b(m,f,q),C=!0},p:We,i(f){C||(_(t.$$.fragment,f),_(m.$$.fragment,f),C=!0)},o(f){v(t.$$.fragment,f),v(m.$$.fragment,f),C=!1},d(f){f&&(n(h),n(l),n(g)),w(t,f),w(m,f)}}}function mo(P){let t,h;return t=new lo({props:{id:"usage",option:"image retrieval",$$slots:{default:[co]},$$scope:{ctx:P}}}),{c(){M(t.$$.fragment)},l(l){y(t.$$.fragment,l)},m(l,p){b(t,l,p),h=!0},p(l,p){const g={};p&2&&(g.$$scope={dirty:p,ctx:l}),t.$set(g)},i(l){h||(_(t.$$.fragment,l),h=!0)},o(l){v(t.$$.fragment,l),h=!1},d(l){w(t,l)}}}function po(P){let t,h="Example:",l,p,g;return p=new Re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycy5tb2RlbHMuY29scGFsaSUyMGltcG9ydCUyMENvbFBhbGlDb25maWclMkMlMjBDb2xQYWxpRm9yUmV0cmlldmFsJTBBJTBBY29uZmlnJTIwJTNEJTIwQ29sUGFsaUNvbmZpZygpJTBBbW9kZWwlMjAlM0QlMjBDb2xQYWxpRm9yUmV0cmlldmFsKGNvbmZpZyk=",highlighted:`<span class="hljs-keyword">from</span> transformers.models.colpali <span class="hljs-keyword">import</span> ColPaliConfig, ColPaliForRetrieval

config = ColPaliConfig()
model = ColPaliForRetrieval(config)`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,l=s(),M(p.$$.fragment)},l(m){t=d(m,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),l=r(m),y(p.$$.fragment,m)},m(m,C){i(m,t,C),i(m,l,C),b(p,m,C),g=!0},p:We,i(m){g||(_(p.$$.fragment,m),g=!0)},o(m){v(p.$$.fragment,m),g=!1},d(m){m&&(n(t),n(l)),w(p,m)}}}function uo(P){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(l){t=d(l,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(l,p){i(l,t,p)},p:We,d(l){l&&n(t)}}}function fo(P){let t,h="Example:",l,p,g;return p=new Re({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=c("p"),t.textContent=h,l=s(),M(p.$$.fragment)},l(m){t=d(m,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),l=r(m),y(p.$$.fragment,m)},m(m,C){i(m,t,C),i(m,l,C),b(p,m,C),g=!0},p:We,i(m){g||(_(p.$$.fragment,m),g=!0)},o(m){v(p.$$.fragment,m),g=!1},d(m){m&&(n(t),n(l)),w(p,m)}}}function ho(P){let t,h,l,p,g,m="<em>This model was released on 2024-06-27 and added to Hugging Face Transformers on 2024-12-17.</em>",C,f,q='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ie,L,Fe,S,Ut='<a href="https://huggingface.co/papers/2407.01449" rel="nofollow">ColPali</a> is a model designed to retrieve documents by analyzing their visual features. Unlike traditional systems that rely heavily on text extraction and OCR, ColPali treats each page as an image. It uses <a href="./paligemma">Paligemma-3B</a> to capture not only text, but also the layout, tables, charts, and other visual elements to create detailed multi-vector embeddings that can be used for retrieval by computing pairwise late interaction similarity scores. This offers a more comprehensive understanding of documents and enables more efficient and accurate retrieval.',Ge,Y,$t='This model was contributed by <a href="https://huggingface.co/tonywu71" rel="nofollow">@tonywu71</a> (ILLUIN Technology) and <a href="https://huggingface.co/yonigozlan" rel="nofollow">@yonigozlan</a> (HuggingFace).',Ve,A,Pt='You can find all the original ColPali checkpoints under Vidore’s <a href="https://huggingface.co/collections/vidore/hf-native-colvision-models-6755d68fc60a8553acaa96f7" rel="nofollow">Hf-native ColVision Models</a> collection.',qe,z,ze,E,Ee,D,kt='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Xe,K,xt='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to quantize the weights to int4.',Qe,O,He,ee,Ne,te,Zt='<li><a href="/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliProcessor.score_retrieval">score_retrieval()</a> returns a 2D tensor where the first dimension is the number of queries and the second dimension is the number of images. A higher score indicates more similarity between the query and image.</li>',Le,oe,Se,j,ne,ot,fe,Bt=`Configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliForRetrieval">ColPaliForRetrieval</a>. It is used to instantiate an instance
of <code>ColPaliForRetrieval</code> according to the specified arguments, defining the model architecture following the methodology
from the “ColPali: Efficient Document Retrieval with Vision Language Models” paper.`,nt,he,Rt=`Creating a configuration with the default settings will result in a configuration where the VLM backbone is set to the
default PaliGemma configuration, i.e the one from <a href="https://huggingface.co/vidore/colpali-v1.2" rel="nofollow">vidore/colpali-v1.2</a>.`,st,ge,Wt=`Note that contrarily to what the class name suggests (actually the name refers to the ColPali <strong>methodology</strong>), you can
use a different VLM backbone model than PaliGemma by passing the corresponding VLM configuration to the class constructor.`,rt,Me,It=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,at,X,Ye,se,Ae,J,re,lt,ye,Ft=`Constructs a ColPali processor which wraps a PaliGemmaProcessor and special methods to process images and queries, as
well as to compute the late-interaction retrieval score.`,it,be,Gt=`<a href="/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliProcessor">ColPaliProcessor</a> offers all the functionalities of <a href="/docs/transformers/v4.56.2/en/model_doc/paligemma#transformers.PaliGemmaProcessor">PaliGemmaProcessor</a>. See the <code>__call__()</code>
for more information.`,ct,Z,ae,dt,_e,Vt=`Prepare for the model one or several image(s). This method is a wrapper around the <code>__call__</code> method of the ColPaliProcessor’s
<code>ColPaliProcessor.__call__()</code>.`,mt,ve,qt="This method forwards the <code>images</code> and <code>kwargs</code> arguments to the image processor.",pt,B,le,ut,we,zt=`Prepare for the model one or several texts. This method is a wrapper around the <code>__call__</code> method of the ColPaliProcessor’s
<code>ColPaliProcessor.__call__()</code>.`,ft,Te,Et="This method forwards the <code>text</code> and <code>kwargs</code> arguments to the tokenizer.",ht,R,ie,gt,Ce,Xt=`Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
query embeddings (<code>qs</code>) and passage embeddings (<code>ps</code>). For ColPali, a passage is the
image of a document page.`,Mt,je,Qt=`Because the embedding tensors are multi-vector and can thus have different shapes, they
should be fed as:
(1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
(2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
obtained by padding the list of tensors.`,De,ce,Ke,T,de,yt,Je,Ht=`The ColPali architecture leverages VLMs to construct efficient multi-vector embeddings directly
from document images (“screenshots”) for document retrieval. The model is trained to maximize the similarity
between these document embeddings and the corresponding query embeddings, using the late interaction method
introduced in ColBERT.`,bt,Ue,Nt=`Using ColPali removes the need for potentially complex and brittle layout recognition and OCR pipelines with a
single model that can take into account both the textual and visual content (layout, charts, etc.) of a document.`,_t,$e,Lt=`ColPali is part of the ColVision model family, which was first introduced in the following paper:
<a href="https://huggingface.co/papers/2407.01449" rel="nofollow"><em>ColPali: Efficient Document Retrieval with Vision Language Models</em></a>.`,vt,Pe,St=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,wt,ke,Yt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Tt,x,me,Ct,xe,At='The <a href="/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliForRetrieval">ColPaliForRetrieval</a> forward method, overrides the <code>__call__</code> special method.',jt,Q,Jt,H,Oe,pe,et,Ze,tt;return L=new Be({props:{title:"ColPali",local:"colpali",headingTag:"h1"}}),z=new Dt({props:{warning:!1,$$slots:{default:[io]},$$scope:{ctx:P}}}),E=new ao({props:{id:"usage",options:["image retrieval"],$$slots:{default:[mo]},$$scope:{ctx:P}}}),O=new Re({props:{code:"aW1wb3J0JTIwcmVxdWVzdHMlMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQSUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBCaXRzQW5kQnl0ZXNDb25maWclMkMlMjBDb2xQYWxpRm9yUmV0cmlldmFsJTJDJTIwQ29sUGFsaVByb2Nlc3NvciUwQSUwQSUwQW1vZGVsX25hbWUlMjAlM0QlMjAlMjJ2aWRvcmUlMkZjb2xwYWxpLXYxLjMtaGYlMjIlMEElMEElMjMlMjA0LWJpdCUyMHF1YW50aXphdGlvbiUyMGNvbmZpZ3VyYXRpb24lMEFibmJfY29uZmlnJTIwJTNEJTIwQml0c0FuZEJ5dGVzQ29uZmlnKCUwQSUyMCUyMCUyMCUyMGxvYWRfaW5fNGJpdCUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjBibmJfNGJpdF91c2VfZG91YmxlX3F1YW50JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3F1YW50X3R5cGUlM0QlMjJuZjQlMjIlMkMlMEElMjAlMjAlMjAlMjBibmJfNGJpdF9jb21wdXRlX2R0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSklMEElMEFtb2RlbCUyMCUzRCUyMENvbFBhbGlGb3JSZXRyaWV2YWwuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMG1vZGVsX25hbWUlMkMlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEYm5iX2NvbmZpZyUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBKSUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMENvbFBhbGlQcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX25hbWUpJTBBJTBBdXJsMSUyMCUzRCUyMCUyMmh0dHBzJTNBJTJGJTJGdXBsb2FkLndpa2ltZWRpYS5vcmclMkZ3aWtpcGVkaWElMkZjb21tb25zJTJGOCUyRjg5JTJGVVMtb3JpZ2luYWwtRGVjbGFyYXRpb24tMTc3Ni5qcGclMjIlMEF1cmwyJTIwJTNEJTIwJTIyaHR0cHMlM0ElMkYlMkZ1cGxvYWQud2lraW1lZGlhLm9yZyUyRndpa2lwZWRpYSUyRmNvbW1vbnMlMkZ0aHVtYiUyRjQlMkY0YyUyRlJvbWVvYW5kanVsaWV0MTU5Ny5qcGclMkY1MDBweC1Sb21lb2FuZGp1bGlldDE1OTcuanBnJTIyJTBBJTBBaW1hZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwSW1hZ2Uub3BlbihyZXF1ZXN0cy5nZXQodXJsMSUyQyUyMHN0cmVhbSUzRFRydWUpLnJhdyklMkMlMEElMjAlMjAlMjAlMjBJbWFnZS5vcGVuKHJlcXVlc3RzLmdldCh1cmwyJTJDJTIwc3RyZWFtJTNEVHJ1ZSkucmF3KSUyQyUwQSU1RCUwQSUwQXF1ZXJpZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlMjJXaGVuJTIwd2FzJTIwdGhlJTIwVW5pdGVkJTIwU3RhdGVzJTIwRGVjbGFyYXRpb24lMjBvZiUyMEluZGVwZW5kZW5jZSUyMHByb2NsYWltZWQlM0YlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjJXaG8lMjBwcmludGVkJTIwdGhlJTIwZWRpdGlvbiUyMG9mJTIwUm9tZW8lMjBhbmQlMjBKdWxpZXQlM0YlMjIlMkMlMEElNUQlMEElMEElMjMlMjBQcm9jZXNzJTIwdGhlJTIwaW5wdXRzJTBBaW5wdXRzX2ltYWdlcyUyMCUzRCUyMHByb2Nlc3NvcihpbWFnZXMlM0RpbWFnZXMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBaW5wdXRzX3RleHQlMjAlM0QlMjBwcm9jZXNzb3IodGV4dCUzRHF1ZXJpZXMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBJTIzJTIwRm9yd2FyZCUyMHBhc3MlMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwaW1hZ2VfZW1iZWRkaW5ncyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzX2ltYWdlcykuZW1iZWRkaW5ncyUwQSUyMCUyMCUyMCUyMHF1ZXJ5X2VtYmVkZGluZ3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0c190ZXh0KS5lbWJlZGRpbmdzJTBBJTBBJTIzJTIwU2NvcmUlMjB0aGUlMjBxdWVyaWVzJTIwYWdhaW5zdCUyMHRoZSUyMGltYWdlcyUwQXNjb3JlcyUyMCUzRCUyMHByb2Nlc3Nvci5zY29yZV9yZXRyaWV2YWwocXVlcnlfZW1iZWRkaW5ncyUyQyUyMGltYWdlX2VtYmVkZGluZ3MpJTBBJTBBcHJpbnQoJTIyUmV0cmlldmFsJTIwc2NvcmVzJTIwKHF1ZXJ5JTIweCUyMGltYWdlKSUzQSUyMiklMEFwcmludChzY29yZXMp",highlighted:`<span class="hljs-keyword">import</span> requests
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image

<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, ColPaliForRetrieval, ColPaliProcessor


model_name = <span class="hljs-string">&quot;vidore/colpali-v1.3-hf&quot;</span>

<span class="hljs-comment"># 4-bit quantization configuration</span>
bnb_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_compute_dtype=torch.float16,
)

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)

processor = ColPaliProcessor.from_pretrained(model_name)

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
<span class="hljs-built_in">print</span>(scores)`,wrap:!1}}),ee=new Be({props:{title:"Notes",local:"notes",headingTag:"h2"}}),oe=new Be({props:{title:"ColPaliConfig",local:"transformers.ColPaliConfig",headingTag:"h2"}}),ne=new ue({props:{name:"class transformers.ColPaliConfig",anchor:"transformers.ColPaliConfig",parameters:[{name:"vlm_config",val:" = None"},{name:"text_config",val:" = None"},{name:"embedding_dim",val:": int = 128"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ColPaliConfig.vlm_config",description:`<strong>vlm_config</strong> (<code>PretrainedConfig</code>, <em>optional</em>) &#x2014;
Configuration of the VLM backbone model.`,name:"vlm_config"},{anchor:"transformers.ColPaliConfig.text_config",description:`<strong>text_config</strong> (<code>PretrainedConfig</code>, <em>optional</em>) &#x2014;
Configuration of the text backbone model. Overrides the <code>text_config</code> attribute of the <code>vlm_config</code> if provided.`,name:"text_config"},{anchor:"transformers.ColPaliConfig.embedding_dim",description:`<strong>embedding_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Dimension of the multi-vector embeddings produced by the model.`,name:"embedding_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/configuration_colpali.py#L27"}}),X=new Kt({props:{anchor:"transformers.ColPaliConfig.example",$$slots:{default:[po]},$$scope:{ctx:P}}}),se=new Be({props:{title:"ColPaliProcessor",local:"transformers.ColPaliProcessor",headingTag:"h2"}}),re=new ue({props:{name:"class transformers.ColPaliProcessor",anchor:"transformers.ColPaliProcessor",parameters:[{name:"image_processor",val:" = None"},{name:"tokenizer",val:" = None"},{name:"chat_template",val:" = None"},{name:"visual_prompt_prefix",val:": str = 'Describe the image.'"},{name:"query_prefix",val:": str = 'Question: '"}],parametersDescription:[{anchor:"transformers.ColPaliProcessor.image_processor",description:`<strong>image_processor</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor">SiglipImageProcessor</a>, <em>optional</em>) &#x2014;
The image processor is a required input.`,name:"image_processor"},{anchor:"transformers.ColPaliProcessor.tokenizer",description:`<strong>tokenizer</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast">LlamaTokenizerFast</a>, <em>optional</em>) &#x2014;
The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.ColPaliProcessor.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014; A Jinja template which will be used to convert lists of messages
in a chat into a tokenizable string.`,name:"chat_template"},{anchor:"transformers.ColPaliProcessor.visual_prompt_prefix",description:`<strong>visual_prompt_prefix</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;Describe the image.&quot;</code>) &#x2014;
A string that gets tokenized and prepended to the image tokens.`,name:"visual_prompt_prefix"},{anchor:"transformers.ColPaliProcessor.query_prefix",description:`<strong>query_prefix</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;Question -- &quot;</code>):
A prefix to be used for the query.`,name:"query_prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/processing_colpali.py#L75"}}),ae=new ue({props:{name:"process_images",anchor:"transformers.ColPaliProcessor.process_images",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.colpali.processing_colpali.ColPaliProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.ColPaliProcessor.process_images.images",description:`<strong>images</strong> (<code>PIL.Image.Image</code>, <code>np.ndarray</code>, <code>torch.Tensor</code>, <code>list[PIL.Image.Image]</code>, <code>list[np.ndarray]</code>, <code>list[torch.Tensor]</code>) &#x2014;
The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
number of channels, H and W are image height and width.`,name:"images"},{anchor:"transformers.ColPaliProcessor.process_images.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/processing_colpali.py#L280",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
`}}),le=new ue({props:{name:"process_queries",anchor:"transformers.ColPaliProcessor.process_queries",parameters:[{name:"text",val:": typing.Union[str, list[str]]"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.colpali.processing_colpali.ColPaliProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.ColPaliProcessor.process_queries.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.ColPaliProcessor.process_queries.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/processing_colpali.py#L315",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


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
`}}),ie=new ue({props:{name:"score_retrieval",anchor:"transformers.ColPaliProcessor.score_retrieval",parameters:[{name:"query_embeddings",val:": typing.Union[ForwardRef('torch.Tensor'), list['torch.Tensor']]"},{name:"passage_embeddings",val:": typing.Union[ForwardRef('torch.Tensor'), list['torch.Tensor']]"},{name:"batch_size",val:": int = 128"},{name:"output_dtype",val:": typing.Optional[ForwardRef('torch.dtype')] = None"},{name:"output_device",val:": typing.Union[ForwardRef('torch.device'), str] = 'cpu'"}],parametersDescription:[{anchor:"transformers.ColPaliProcessor.score_retrieval.query_embeddings",description:"<strong>query_embeddings</strong> (<code>Union[torch.Tensor, list[torch.Tensor]</code>) &#x2014; Query embeddings.",name:"query_embeddings"},{anchor:"transformers.ColPaliProcessor.score_retrieval.passage_embeddings",description:"<strong>passage_embeddings</strong> (<code>Union[torch.Tensor, list[torch.Tensor]</code>) &#x2014; Passage embeddings.",name:"passage_embeddings"},{anchor:"transformers.ColPaliProcessor.score_retrieval.batch_size",description:"<strong>batch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014; Batch size for computing scores.",name:"batch_size"},{anchor:"transformers.ColPaliProcessor.score_retrieval.output_dtype",description:`<strong>output_dtype</strong> (<code>torch.dtype</code>, <em>optional</em>, defaults to <code>torch.float32</code>) &#x2014; The dtype of the output tensor.
If <code>None</code>, the dtype of the input embeddings is used.`,name:"output_dtype"},{anchor:"transformers.ColPaliProcessor.score_retrieval.output_device",description:"<strong>output_device</strong> (<code>torch.device</code> or <code>str</code>, <em>optional</em>, defaults to &#x201C;cpu&#x201D;) &#x2014; The device of the output tensor.",name:"output_device"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/processing_colpali.py#L349",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A tensor of shape <code>(n_queries, n_passages)</code> containing the scores. The score
tensor is saved on the “cpu” device.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>torch.Tensor</code></p>
`}}),ce=new Be({props:{title:"ColPaliForRetrieval",local:"transformers.ColPaliForRetrieval",headingTag:"h2"}}),de=new ue({props:{name:"class transformers.ColPaliForRetrieval",anchor:"transformers.ColPaliForRetrieval",parameters:[{name:"config",val:": ColPaliConfig"}],parametersDescription:[{anchor:"transformers.ColPaliForRetrieval.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliConfig">ColPaliConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/modeling_colpali.py#L102"}}),me=new ue({props:{name:"forward",anchor:"transformers.ColPaliForRetrieval.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"pixel_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ColPaliForRetrieval.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ColPaliForRetrieval.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<code>image_processor_class</code>. See <code>image_processor_class.__call__</code> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliProcessor">ColPaliProcessor</a> uses
<code>image_processor_class</code> for processing images).`,name:"pixel_values"},{anchor:"transformers.ColPaliForRetrieval.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ColPaliForRetrieval.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ColPaliForRetrieval.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ColPaliForRetrieval.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/colpali/modeling_colpali.py#L126",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.colpali.modeling_colpali.ColPaliForRetrievalOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliConfig"
>ColPaliConfig</a>) and inputs.</p>
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
<li>
<p><strong>image_hidden_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) — A <code>torch.FloatTensor</code> of size <code>(batch_size, num_images, sequence_length, hidden_size)</code>.
image_hidden_states of the model produced by the vision encoder after projecting last hidden state.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.colpali.modeling_colpali.ColPaliForRetrievalOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Q=new Dt({props:{$$slots:{default:[uo]},$$scope:{ctx:P}}}),H=new Kt({props:{anchor:"transformers.ColPaliForRetrieval.forward.example",$$slots:{default:[fo]},$$scope:{ctx:P}}}),pe=new ro({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/colpali.md"}}),{c(){t=c("meta"),h=s(),l=c("p"),p=s(),g=c("p"),g.innerHTML=m,C=s(),f=c("div"),f.innerHTML=q,Ie=s(),M(L.$$.fragment),Fe=s(),S=c("p"),S.innerHTML=Ut,Ge=s(),Y=c("p"),Y.innerHTML=$t,Ve=s(),A=c("p"),A.innerHTML=Pt,qe=s(),M(z.$$.fragment),ze=s(),M(E.$$.fragment),Ee=s(),D=c("p"),D.innerHTML=kt,Xe=s(),K=c("p"),K.innerHTML=xt,Qe=s(),M(O.$$.fragment),He=s(),M(ee.$$.fragment),Ne=s(),te=c("ul"),te.innerHTML=Zt,Le=s(),M(oe.$$.fragment),Se=s(),j=c("div"),M(ne.$$.fragment),ot=s(),fe=c("p"),fe.innerHTML=Bt,nt=s(),he=c("p"),he.innerHTML=Rt,st=s(),ge=c("p"),ge.innerHTML=Wt,rt=s(),Me=c("p"),Me.innerHTML=It,at=s(),M(X.$$.fragment),Ye=s(),M(se.$$.fragment),Ae=s(),J=c("div"),M(re.$$.fragment),lt=s(),ye=c("p"),ye.textContent=Ft,it=s(),be=c("p"),be.innerHTML=Gt,ct=s(),Z=c("div"),M(ae.$$.fragment),dt=s(),_e=c("p"),_e.innerHTML=Vt,mt=s(),ve=c("p"),ve.innerHTML=qt,pt=s(),B=c("div"),M(le.$$.fragment),ut=s(),we=c("p"),we.innerHTML=zt,ft=s(),Te=c("p"),Te.innerHTML=Et,ht=s(),R=c("div"),M(ie.$$.fragment),gt=s(),Ce=c("p"),Ce.innerHTML=Xt,Mt=s(),je=c("p"),je.textContent=Qt,De=s(),M(ce.$$.fragment),Ke=s(),T=c("div"),M(de.$$.fragment),yt=s(),Je=c("p"),Je.textContent=Ht,bt=s(),Ue=c("p"),Ue.textContent=Nt,_t=s(),$e=c("p"),$e.innerHTML=Lt,vt=s(),Pe=c("p"),Pe.innerHTML=St,wt=s(),ke=c("p"),ke.innerHTML=Yt,Tt=s(),x=c("div"),M(me.$$.fragment),Ct=s(),xe=c("p"),xe.innerHTML=At,jt=s(),M(Q.$$.fragment),Jt=s(),M(H.$$.fragment),Oe=s(),M(pe.$$.fragment),et=s(),Ze=c("p"),this.h()},l(e){const o=no("svelte-u9bgzb",document.head);t=d(o,"META",{name:!0,content:!0}),o.forEach(n),h=r(e),l=d(e,"P",{}),G(l).forEach(n),p=r(e),g=d(e,"P",{"data-svelte-h":!0}),u(g)!=="svelte-13k78rk"&&(g.innerHTML=m),C=r(e),f=d(e,"DIV",{style:!0,"data-svelte-h":!0}),u(f)!=="svelte-wa5t4p"&&(f.innerHTML=q),Ie=r(e),y(L.$$.fragment,e),Fe=r(e),S=d(e,"P",{"data-svelte-h":!0}),u(S)!=="svelte-zfv536"&&(S.innerHTML=Ut),Ge=r(e),Y=d(e,"P",{"data-svelte-h":!0}),u(Y)!=="svelte-1usq28k"&&(Y.innerHTML=$t),Ve=r(e),A=d(e,"P",{"data-svelte-h":!0}),u(A)!=="svelte-st4jp7"&&(A.innerHTML=Pt),qe=r(e),y(z.$$.fragment,e),ze=r(e),y(E.$$.fragment,e),Ee=r(e),D=d(e,"P",{"data-svelte-h":!0}),u(D)!=="svelte-nf5ooi"&&(D.innerHTML=kt),Xe=r(e),K=d(e,"P",{"data-svelte-h":!0}),u(K)!=="svelte-x2dyjs"&&(K.innerHTML=xt),Qe=r(e),y(O.$$.fragment,e),He=r(e),y(ee.$$.fragment,e),Ne=r(e),te=d(e,"UL",{"data-svelte-h":!0}),u(te)!=="svelte-czjfxr"&&(te.innerHTML=Zt),Le=r(e),y(oe.$$.fragment,e),Se=r(e),j=d(e,"DIV",{class:!0});var U=G(j);y(ne.$$.fragment,U),ot=r(U),fe=d(U,"P",{"data-svelte-h":!0}),u(fe)!=="svelte-1g3tceo"&&(fe.innerHTML=Bt),nt=r(U),he=d(U,"P",{"data-svelte-h":!0}),u(he)!=="svelte-1fyeyvm"&&(he.innerHTML=Rt),st=r(U),ge=d(U,"P",{"data-svelte-h":!0}),u(ge)!=="svelte-bnl7k3"&&(ge.innerHTML=Wt),rt=r(U),Me=d(U,"P",{"data-svelte-h":!0}),u(Me)!=="svelte-1ek1ss9"&&(Me.innerHTML=It),at=r(U),y(X.$$.fragment,U),U.forEach(n),Ye=r(e),y(se.$$.fragment,e),Ae=r(e),J=d(e,"DIV",{class:!0});var $=G(J);y(re.$$.fragment,$),lt=r($),ye=d($,"P",{"data-svelte-h":!0}),u(ye)!=="svelte-1xjue1y"&&(ye.textContent=Ft),it=r($),be=d($,"P",{"data-svelte-h":!0}),u(be)!=="svelte-1g718z6"&&(be.innerHTML=Gt),ct=r($),Z=d($,"DIV",{class:!0});var W=G(Z);y(ae.$$.fragment,W),dt=r(W),_e=d(W,"P",{"data-svelte-h":!0}),u(_e)!=="svelte-1gls1hp"&&(_e.innerHTML=Vt),mt=r(W),ve=d(W,"P",{"data-svelte-h":!0}),u(ve)!=="svelte-1at6zp"&&(ve.innerHTML=qt),W.forEach(n),pt=r($),B=d($,"DIV",{class:!0});var I=G(B);y(le.$$.fragment,I),ut=r(I),we=d(I,"P",{"data-svelte-h":!0}),u(we)!=="svelte-d7fz8o"&&(we.innerHTML=zt),ft=r(I),Te=d(I,"P",{"data-svelte-h":!0}),u(Te)!=="svelte-1y7x5ok"&&(Te.innerHTML=Et),I.forEach(n),ht=r($),R=d($,"DIV",{class:!0});var F=G(R);y(ie.$$.fragment,F),gt=r(F),Ce=d(F,"P",{"data-svelte-h":!0}),u(Ce)!=="svelte-1728slr"&&(Ce.innerHTML=Xt),Mt=r(F),je=d(F,"P",{"data-svelte-h":!0}),u(je)!=="svelte-3nd4tx"&&(je.textContent=Qt),F.forEach(n),$.forEach(n),De=r(e),y(ce.$$.fragment,e),Ke=r(e),T=d(e,"DIV",{class:!0});var k=G(T);y(de.$$.fragment,k),yt=r(k),Je=d(k,"P",{"data-svelte-h":!0}),u(Je)!=="svelte-2l0au6"&&(Je.textContent=Ht),bt=r(k),Ue=d(k,"P",{"data-svelte-h":!0}),u(Ue)!=="svelte-rxcp7f"&&(Ue.textContent=Nt),_t=r(k),$e=d(k,"P",{"data-svelte-h":!0}),u($e)!=="svelte-13suc5t"&&($e.innerHTML=Lt),vt=r(k),Pe=d(k,"P",{"data-svelte-h":!0}),u(Pe)!=="svelte-q52n56"&&(Pe.innerHTML=St),wt=r(k),ke=d(k,"P",{"data-svelte-h":!0}),u(ke)!=="svelte-hswkmf"&&(ke.innerHTML=Yt),Tt=r(k),x=d(k,"DIV",{class:!0});var N=G(x);y(me.$$.fragment,N),Ct=r(N),xe=d(N,"P",{"data-svelte-h":!0}),u(xe)!=="svelte-jdiy4l"&&(xe.innerHTML=At),jt=r(N),y(Q.$$.fragment,N),Jt=r(N),y(H.$$.fragment,N),N.forEach(n),k.forEach(n),Oe=r(e),y(pe.$$.fragment,e),et=r(e),Ze=d(e,"P",{}),G(Ze).forEach(n),this.h()},h(){V(t,"name","hf:doc:metadata"),V(t,"content",go),so(f,"float","right"),V(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){a(document.head,t),i(e,h,o),i(e,l,o),i(e,p,o),i(e,g,o),i(e,C,o),i(e,f,o),i(e,Ie,o),b(L,e,o),i(e,Fe,o),i(e,S,o),i(e,Ge,o),i(e,Y,o),i(e,Ve,o),i(e,A,o),i(e,qe,o),b(z,e,o),i(e,ze,o),b(E,e,o),i(e,Ee,o),i(e,D,o),i(e,Xe,o),i(e,K,o),i(e,Qe,o),b(O,e,o),i(e,He,o),b(ee,e,o),i(e,Ne,o),i(e,te,o),i(e,Le,o),b(oe,e,o),i(e,Se,o),i(e,j,o),b(ne,j,null),a(j,ot),a(j,fe),a(j,nt),a(j,he),a(j,st),a(j,ge),a(j,rt),a(j,Me),a(j,at),b(X,j,null),i(e,Ye,o),b(se,e,o),i(e,Ae,o),i(e,J,o),b(re,J,null),a(J,lt),a(J,ye),a(J,it),a(J,be),a(J,ct),a(J,Z),b(ae,Z,null),a(Z,dt),a(Z,_e),a(Z,mt),a(Z,ve),a(J,pt),a(J,B),b(le,B,null),a(B,ut),a(B,we),a(B,ft),a(B,Te),a(J,ht),a(J,R),b(ie,R,null),a(R,gt),a(R,Ce),a(R,Mt),a(R,je),i(e,De,o),b(ce,e,o),i(e,Ke,o),i(e,T,o),b(de,T,null),a(T,yt),a(T,Je),a(T,bt),a(T,Ue),a(T,_t),a(T,$e),a(T,vt),a(T,Pe),a(T,wt),a(T,ke),a(T,Tt),a(T,x),b(me,x,null),a(x,Ct),a(x,xe),a(x,jt),b(Q,x,null),a(x,Jt),b(H,x,null),i(e,Oe,o),b(pe,e,o),i(e,et,o),i(e,Ze,o),tt=!0},p(e,[o]){const U={};o&2&&(U.$$scope={dirty:o,ctx:e}),z.$set(U);const $={};o&2&&($.$$scope={dirty:o,ctx:e}),E.$set($);const W={};o&2&&(W.$$scope={dirty:o,ctx:e}),X.$set(W);const I={};o&2&&(I.$$scope={dirty:o,ctx:e}),Q.$set(I);const F={};o&2&&(F.$$scope={dirty:o,ctx:e}),H.$set(F)},i(e){tt||(_(L.$$.fragment,e),_(z.$$.fragment,e),_(E.$$.fragment,e),_(O.$$.fragment,e),_(ee.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(X.$$.fragment,e),_(se.$$.fragment,e),_(re.$$.fragment,e),_(ae.$$.fragment,e),_(le.$$.fragment,e),_(ie.$$.fragment,e),_(ce.$$.fragment,e),_(de.$$.fragment,e),_(me.$$.fragment,e),_(Q.$$.fragment,e),_(H.$$.fragment,e),_(pe.$$.fragment,e),tt=!0)},o(e){v(L.$$.fragment,e),v(z.$$.fragment,e),v(E.$$.fragment,e),v(O.$$.fragment,e),v(ee.$$.fragment,e),v(oe.$$.fragment,e),v(ne.$$.fragment,e),v(X.$$.fragment,e),v(se.$$.fragment,e),v(re.$$.fragment,e),v(ae.$$.fragment,e),v(le.$$.fragment,e),v(ie.$$.fragment,e),v(ce.$$.fragment,e),v(de.$$.fragment,e),v(me.$$.fragment,e),v(Q.$$.fragment,e),v(H.$$.fragment,e),v(pe.$$.fragment,e),tt=!1},d(e){e&&(n(h),n(l),n(p),n(g),n(C),n(f),n(Ie),n(Fe),n(S),n(Ge),n(Y),n(Ve),n(A),n(qe),n(ze),n(Ee),n(D),n(Xe),n(K),n(Qe),n(He),n(Ne),n(te),n(Le),n(Se),n(j),n(Ye),n(Ae),n(J),n(De),n(Ke),n(T),n(Oe),n(et),n(Ze)),n(t),w(L,e),w(z,e),w(E,e),w(O,e),w(ee,e),w(oe,e),w(ne),w(X),w(se,e),w(re),w(ae),w(le),w(ie),w(ce,e),w(de),w(me),w(Q),w(H),w(pe,e)}}}const go='{"title":"ColPali","local":"colpali","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"ColPaliConfig","local":"transformers.ColPaliConfig","sections":[],"depth":2},{"title":"ColPaliProcessor","local":"transformers.ColPaliProcessor","sections":[],"depth":2},{"title":"ColPaliForRetrieval","local":"transformers.ColPaliForRetrieval","sections":[],"depth":2}],"depth":1}';function Mo(P){return eo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Jo extends to{constructor(t){super(),oo(this,t,Mo,ho,Ot,{})}}export{Jo as component};
