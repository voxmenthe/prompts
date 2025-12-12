import{s as Yt,o as Qt,n as Je}from"../chunks/scheduler.18a86fab.js";import{S as At,i as Dt,g as c,s,r as m,A as Kt,h as p,f as o,c as a,j as W,x as u,u as g,k as I,y as d,a as i,v as f,d as _,t as T,w as b}from"../chunks/index.98837b22.js";import{T as Ct}from"../chunks/Tip.77304350.js";import{D as _e}from"../chunks/Docstring.a1ef7999.js";import{C as ot}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as $t}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as O,E as eo}from"../chunks/getInferenceSnippets.06c2775f.js";function to(C){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=y},l(l){n=p(l,"P",{"data-svelte-h":!0}),u(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(l,h){i(l,n,h)},p:Je,d(l){l&&o(n)}}}function oo(C){let n,y="Example:",l,h,M;return h=new ot({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMb25nVDVNb2RlbCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmxvbmctdDUtbG9jYWwtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMExvbmdUNU1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZsb25nLXQ1LWxvY2FsLWJhc2UlMjIpJTBBJTBBJTIzJTIwTGV0J3MlMjB0cnklMjBhJTIwdmVyeSUyMGxvbmclMjBlbmNvZGVyJTIwaW5wdXQuJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMDEwMCUyMColMjAlMjJTdHVkaWVzJTIwaGF2ZSUyMGJlZW4lMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSkuaW5wdXRfaWRzJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQSUwQWRlY29kZXJfaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlN0dWRpZXMlMjBzaG93JTIwdGhhdCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEElMEElMjMlMjBmb3J3YXJkJTIwcGFzcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dF9pZHMlMkMlMjBkZWNvZGVyX2lucHV0X2lkcyUzRGRlY29kZXJfaW5wdXRfaWRzKSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LongT5Model

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/long-t5-local-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LongT5Model.from_pretrained(<span class="hljs-string">&quot;google/long-t5-local-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Let&#x27;s try a very long encoder input.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-number">100</span> * <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># forward pass</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){n=c("p"),n.textContent=y,l=s(),m(h.$$.fragment)},l(r){n=p(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-11lpom8"&&(n.textContent=y),l=a(r),g(h.$$.fragment,r)},m(r,k){i(r,n,k),i(r,l,k),f(h,r,k),M=!0},p:Je,i(r){M||(_(h.$$.fragment,r),M=!0)},o(r){T(h.$$.fragment,r),M=!1},d(r){r&&(o(n),o(l)),b(h,r)}}}function no(C){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=y},l(l){n=p(l,"P",{"data-svelte-h":!0}),u(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(l,h){i(l,n,h)},p:Je,d(l){l&&o(n)}}}function so(C){let n,y="Examples:",l,h,M;return h=new ot({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMb25nVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJTdGFuY2xkJTJGbG9uZ3Q1LXRnbG9iYWwtbGFyZ2UtMTYzODQtcHVibWVkLTNrX3N0ZXBzJTIyKSUwQW1vZGVsJTIwJTNEJTIwTG9uZ1Q1Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJTdGFuY2xkJTJGbG9uZ3Q1LXRnbG9iYWwtbGFyZ2UtMTYzODQtcHVibWVkLTNrX3N0ZXBzJTIyJTBBKSUwQSUwQSUyMyUyMExldCdzJTIwdHJ5JTIwYSUyMHZlcnklMjBsb25nJTIwaW5wdXQuJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKDEwMCUyMColMjAlMjJzdHVkaWVzJTIwaGF2ZSUyMHNob3duJTIwdGhhdCUyMG93bmluZyUyMGElMjBkb2clMjBpcyUyMGdvb2QlMjBmb3IlMjB5b3UlMjAlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWlucHV0X2lkcyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRfaWRzKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LongT5ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Stancld/longt5-tglobal-large-16384-pubmed-3k_steps&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LongT5ForConditionalGeneration.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Stancld/longt5-tglobal-large-16384-pubmed-3k_steps&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Let&#x27;s try a very long input.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-number">100</span> * <span class="hljs-string">&quot;studies have shown that owning a dog is good for you &quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = inputs.input_ids

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.generate(input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))
abstractthe aim of this article <span class="hljs-keyword">is</span> to provide an overview of the literature on the role of dog`,wrap:!1}}),{c(){n=c("p"),n.textContent=y,l=s(),m(h.$$.fragment)},l(r){n=p(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-kvfsh7"&&(n.textContent=y),l=a(r),g(h.$$.fragment,r)},m(r,k){i(r,n,k),i(r,l,k),f(h,r,k),M=!0},p:Je,i(r){M||(_(h.$$.fragment,r),M=!0)},o(r){T(h.$$.fragment,r),M=!1},d(r){r&&(o(n),o(l)),b(h,r)}}}function ao(C){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=y},l(l){n=p(l,"P",{"data-svelte-h":!0}),u(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(l,h){i(l,n,h)},p:Je,d(l){l&&o(n)}}}function ro(C){let n,y="Example:",l,h,M;return h=new ot({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMb25nVDVGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZsb25nLXQ1LWxvY2FsLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBMb25nVDVFbmNvZGVyTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmxvbmctdDUtbG9jYWwtYmFzZSUyMiklMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwMTAwJTIwKiUyMCUyMlN0dWRpZXMlMjBoYXZlJTIwYmVlbiUyMHNob3duJTIwdGhhdCUyMG93bmluZyUyMGElMjBkb2clMjBpcyUyMGdvb2QlMjBmb3IlMjB5b3UlMjAlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUzRGlucHV0X2lkcyklMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LongT5ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/long-t5-local-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LongT5EncoderModel.from_pretrained(<span class="hljs-string">&quot;google/long-t5-local-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-number">100</span> * <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you &quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){n=c("p"),n.textContent=y,l=s(),m(h.$$.fragment)},l(r){n=p(r,"P",{"data-svelte-h":!0}),u(n)!=="svelte-11lpom8"&&(n.textContent=y),l=a(r),g(h.$$.fragment,r)},m(r,k){i(r,n,k),i(r,l,k),f(h,r,k),M=!0},p:Je,i(r){M||(_(h.$$.fragment,r),M=!0)},o(r){T(h.$$.fragment,r),M=!1},d(r){r&&(o(n),o(l)),b(h,r)}}}function io(C){let n,y,l,h,M,r="<em>This model was released on 2021-12-15 and added to Hugging Face Transformers on 2022-06-13.</em>",k,P,Fe,B,jt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',qe,Y,Ie,Q,zt=`The LongT5 model was proposed in <a href="https://huggingface.co/papers/2112.07916" rel="nofollow">LongT5: Efficient Text-To-Text Transformer for Long Sequences</a>
by Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung and Yinfei Yang. It’s an
encoder-decoder transformer pre-trained in a text-to-text denoising generative setting. LongT5 model is an extension of
T5 model, and it enables using one of the two different efficient attention mechanisms - (1) Local attention, or (2)
Transient-Global attention.`,Ze,A,Ut="The abstract from the paper is the following:",We,D,xt=`<em>Recent work has shown that either (1) increasing the input length or (2) increasing model size can improve the
performance of Transformer-based neural models. In this paper, we present a new model, called LongT5, with which we
explore the effects of scaling both the input length and model size at the same time. Specifically, we integrated
attention ideas from long-input transformers (ETC), and adopted pre-training strategies from summarization pre-training
(PEGASUS) into the scalable T5 architecture. The result is a new attention mechanism we call {\\em Transient Global}
(TGlobal), which mimics ETC’s local/global attention mechanism, but without requiring additional side-inputs. We are
able to achieve state-of-the-art results on several summarization tasks and outperform the original T5 models on
question answering tasks.</em>`,Be,K,Jt=`This model was contributed by <a href="https://huggingface.co/stancld" rel="nofollow">stancld</a>.
The original code can be found <a href="https://github.com/google-research/longt5" rel="nofollow">here</a>.`,He,ee,Ne,te,Gt=`<li><a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5ForConditionalGeneration">LongT5ForConditionalGeneration</a> is an extension of <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5ForConditionalGeneration">T5ForConditionalGeneration</a> exchanging the traditional
encoder <em>self-attention</em> layer with efficient either <em>local</em> attention or <em>transient-global</em> (<em>tglobal</em>) attention.</li> <li>Unlike the T5 model, LongT5 does not use a task prefix. Furthermore, it uses a different pre-training objective
inspired by the pre-training of <a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusForConditionalGeneration">PegasusForConditionalGeneration</a>.</li> <li>LongT5 model is designed to work efficiently and very well on long-range <em>sequence-to-sequence</em> tasks where the
input sequence exceeds commonly used 512 tokens. It is capable of handling input sequences of a length up to 16,384 tokens.</li> <li>For <em>Local Attention</em>, the sparse sliding-window local attention operation allows a given token to attend only <code>r</code>
tokens to the left and right of it (with <code>r=127</code> by default). <em>Local Attention</em> does not introduce any new parameters
to the model. The complexity of the mechanism is linear in input sequence length <code>l</code>: <code>O(l*r)</code>.</li> <li><em>Transient Global Attention</em> is an extension of the <em>Local Attention</em>. It, furthermore, allows each input token to
interact with all other tokens in the layer. This is achieved via splitting an input sequence into blocks of a fixed
length <code>k</code> (with a default <code>k=16</code>). Then, a global token for such a block is obtained via summing and normalizing the embeddings of every token
in the block. Thanks to this, the attention allows each token to attend to both nearby tokens like in Local attention, and
also every global token like in the case of standard global attention (<em>transient</em> represents the fact the global tokens
are constructed dynamically within each attention operation).  As a consequence, <em>TGlobal</em> attention introduces
a few new parameters — global relative position biases and a layer normalization for global token’s embedding.
The complexity of this mechanism is <code>O(l(r + l/k))</code>.</li> <li>An example showing how to evaluate a fine-tuned LongT5 model on the <a href="https://huggingface.co/datasets/scientific_papers" rel="nofollow">pubmed dataset</a> is below.</li>`,Xe,oe,Re,ne,Ve,se,Ft='<li><a href="../tasks/translation">Translation task guide</a></li> <li><a href="../tasks/summarization">Summarization task guide</a></li>',Ee,ae,Se,G,re,nt,Te,qt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Model">LongT5Model</a> or a <code>FlaxLongT5Model</code>. It is
used to instantiate a LongT5 model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the LongT5
<a href="https://huggingface.co/google/long-t5-local-base" rel="nofollow">google/long-t5-local-base</a> architecture.`,st,be,It=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Oe,ie,Pe,v,de,at,ye,Zt="The bare Longt5 Model outputting raw hidden-states without any specific head on top.",rt,Me,Wt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,it,ke,Bt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,dt,U,le,lt,ve,Ht='The <a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Model">LongT5Model</a> forward method, overrides the <code>__call__</code> special method.',ct,H,pt,N,Ye,ce,Qe,w,pe,ht,we,Nt="LONGT5 Model with a <code>language modeling</code> head on top.",ut,Le,Xt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,mt,Ce,Rt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,gt,x,he,ft,$e,Vt='The <a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5ForConditionalGeneration">LongT5ForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',_t,X,Tt,R,Ae,ue,De,L,me,bt,je,Et="The bare Longt5 Model outputting raw hidden-states without any specific head on top.",yt,ze,St=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Mt,Ue,Ot=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,kt,J,ge,vt,xe,Pt='The <a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5EncoderModel">LongT5EncoderModel</a> forward method, overrides the <code>__call__</code> special method.',wt,V,Lt,E,Ke,fe,et,Ge,tt;return P=new O({props:{title:"LongT5",local:"longt5",headingTag:"h1"}}),Y=new O({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ee=new O({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),oe=new ot({props:{code:"aW1wb3J0JTIwZXZhbHVhdGUlMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMExvbmdUNUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQWRhdGFzZXQlMjAlM0QlMjBsb2FkX2RhdGFzZXQoJTIyc2NpZW50aWZpY19wYXBlcnMlMjIlMkMlMjAlMjJwdWJtZWQlMjIlMkMlMjBzcGxpdCUzRCUyMnZhbGlkYXRpb24lMjIpJTBBbW9kZWwlMjAlM0QlMjAoJTBBJTIwJTIwJTIwJTIwTG9uZ1Q1Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJTdGFuY2xkJTJGbG9uZ3Q1LXRnbG9iYWwtbGFyZ2UtMTYzODQtcHVibWVkLTNrX3N0ZXBzJTIyKSUwQSUyMCUyMCUyMCUyMC50byglMjJhdXRvJTIyKSUwQSUyMCUyMCUyMCUyMC5oYWxmKCklMEEpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyU3RhbmNsZCUyRmxvbmd0NS10Z2xvYmFsLWxhcmdlLTE2Mzg0LXB1Ym1lZC0za19zdGVwcyUyMiklMEElMEElMEFkZWYlMjBnZW5lcmF0ZV9hbnN3ZXJzKGJhdGNoKSUzQSUwQSUyMCUyMCUyMCUyMGlucHV0c19kaWN0JTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGJhdGNoJTVCJTIyYXJ0aWNsZSUyMiU1RCUyQyUyMG1heF9sZW5ndGglM0QxNjM4NCUyQyUyMHBhZGRpbmclM0QlMjJtYXhfbGVuZ3RoJTIyJTJDJTIwdHJ1bmNhdGlvbiUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBJTIwJTIwJTIwJTIwKSUwQSUyMCUyMCUyMCUyMGlucHV0X2lkcyUyMCUzRCUyMGlucHV0c19kaWN0LmlucHV0X2lkcy50byhtb2RlbC5kZXZpY2UpJTBBJTIwJTIwJTIwJTIwYXR0ZW50aW9uX21hc2slMjAlM0QlMjBpbnB1dHNfZGljdC5hdHRlbnRpb25fbWFzay50byhtb2RlbC5kZXZpY2UpJTBBJTIwJTIwJTIwJTIwb3V0cHV0X2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0X2lkcyUyQyUyMGF0dGVudGlvbl9tYXNrJTNEYXR0ZW50aW9uX21hc2slMkMlMjBtYXhfbGVuZ3RoJTNENTEyJTJDJTIwbnVtX2JlYW1zJTNEMiklMEElMjAlMjAlMjAlMjBiYXRjaCU1QiUyMnByZWRpY3RlZF9hYnN0cmFjdCUyMiU1RCUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUob3V0cHV0X2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUyMCUyMCUyMCUyMHJldHVybiUyMGJhdGNoJTBBJTBBJTBBcmVzdWx0JTIwJTNEJTIwZGF0YXNldC5tYXAoZ2VuZXJhdGVfYW5zd2VyJTJDJTIwYmF0Y2hlZCUzRFRydWUlMkMlMjBiYXRjaF9zaXplJTNEMiklMEFyb3VnZSUyMCUzRCUyMGV2YWx1YXRlLmxvYWQoJTIycm91Z2UlMjIpJTBBcm91Z2UuY29tcHV0ZShwcmVkaWN0aW9ucyUzRHJlc3VsdCU1QiUyMnByZWRpY3RlZF9hYnN0cmFjdCUyMiU1RCUyQyUyMHJlZmVyZW5jZXMlM0RyZXN1bHQlNUIlMjJhYnN0cmFjdCUyMiU1RCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> evaluate
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LongT5ForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;scientific_papers&quot;</span>, <span class="hljs-string">&quot;pubmed&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = (
<span class="hljs-meta">... </span>    LongT5ForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;Stancld/longt5-tglobal-large-16384-pubmed-3k_steps&quot;</span>)
<span class="hljs-meta">... </span>    .to(<span class="hljs-string">&quot;auto&quot;</span>)
<span class="hljs-meta">... </span>    .half()
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Stancld/longt5-tglobal-large-16384-pubmed-3k_steps&quot;</span>)


<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">def</span> <span class="hljs-title function_">generate_answers</span>(<span class="hljs-params">batch</span>):
<span class="hljs-meta">... </span>    inputs_dict = tokenizer(
<span class="hljs-meta">... </span>        batch[<span class="hljs-string">&quot;article&quot;</span>], max_length=<span class="hljs-number">16384</span>, padding=<span class="hljs-string">&quot;max_length&quot;</span>, truncation=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>    )
<span class="hljs-meta">... </span>    input_ids = inputs_dict.input_ids.to(model.device)
<span class="hljs-meta">... </span>    attention_mask = inputs_dict.attention_mask.to(model.device)
<span class="hljs-meta">... </span>    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=<span class="hljs-number">512</span>, num_beams=<span class="hljs-number">2</span>)
<span class="hljs-meta">... </span>    batch[<span class="hljs-string">&quot;predicted_abstract&quot;</span>] = tokenizer.batch_decode(output_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-meta">... </span>    <span class="hljs-keyword">return</span> batch


<span class="hljs-meta">&gt;&gt;&gt; </span>result = dataset.<span class="hljs-built_in">map</span>(generate_answer, batched=<span class="hljs-literal">True</span>, batch_size=<span class="hljs-number">2</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>rouge = evaluate.load(<span class="hljs-string">&quot;rouge&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>rouge.compute(predictions=result[<span class="hljs-string">&quot;predicted_abstract&quot;</span>], references=result[<span class="hljs-string">&quot;abstract&quot;</span>])`,wrap:!1}}),ne=new O({props:{title:"Resources",local:"resources",headingTag:"h2"}}),ae=new O({props:{title:"LongT5Config",local:"transformers.LongT5Config",headingTag:"h2"}}),re=new _e({props:{name:"class transformers.LongT5Config",anchor:"transformers.LongT5Config",parameters:[{name:"vocab_size",val:" = 32128"},{name:"d_model",val:" = 512"},{name:"d_kv",val:" = 64"},{name:"d_ff",val:" = 2048"},{name:"num_layers",val:" = 6"},{name:"num_decoder_layers",val:" = None"},{name:"num_heads",val:" = 8"},{name:"local_radius",val:" = 127"},{name:"global_block_size",val:" = 16"},{name:"relative_attention_num_buckets",val:" = 32"},{name:"relative_attention_max_distance",val:" = 128"},{name:"dropout_rate",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-06"},{name:"initializer_factor",val:" = 1.0"},{name:"feed_forward_proj",val:" = 'relu'"},{name:"is_encoder_decoder",val:" = True"},{name:"encoder_attention_type",val:" = 'local'"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LongT5Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32128) &#x2014;
Vocabulary size of the LongT5 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Model">LongT5Model</a>.`,name:"vocab_size"},{anchor:"transformers.LongT5Config.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Size of the encoder layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.LongT5Config.d_kv",description:`<strong>d_kv</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Size of the key, query, value projections per attention head. <code>d_kv</code> has to be equal to <code>d_model // num_heads</code>.`,name:"d_kv"},{anchor:"transformers.LongT5Config.d_ff",description:`<strong>d_ff</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Size of the intermediate feed forward layer in each <code>LongT5Block</code>.`,name:"d_ff"},{anchor:"transformers.LongT5Config.num_layers",description:`<strong>num_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_layers"},{anchor:"transformers.LongT5Config.num_decoder_layers",description:`<strong>num_decoder_layers</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Number of hidden layers in the Transformer decoder. Will use the same value as <code>num_layers</code> if not set.`,name:"num_decoder_layers"},{anchor:"transformers.LongT5Config.num_heads",description:`<strong>num_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_heads"},{anchor:"transformers.LongT5Config.local_radius",description:`<strong>local_radius</strong> (<code>int</code>, <em>optional</em>, defaults to 127) &#x2014;
Number of tokens to the left/right for each token to locally self-attend in a local attention mechanism.`,name:"local_radius"},{anchor:"transformers.LongT5Config.global_block_size",description:`<strong>global_block_size</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Length of blocks an input sequence is divided into for a global token representation. Used only for
<code>encoder_attention_type = &quot;transient-global&quot;</code>.`,name:"global_block_size"},{anchor:"transformers.LongT5Config.relative_attention_num_buckets",description:`<strong>relative_attention_num_buckets</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of buckets to use for each attention layer.`,name:"relative_attention_num_buckets"},{anchor:"transformers.LongT5Config.relative_attention_max_distance",description:`<strong>relative_attention_max_distance</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The maximum distance of the longer sequences for the bucket separation.`,name:"relative_attention_max_distance"},{anchor:"transformers.LongT5Config.dropout_rate",description:`<strong>dropout_rate</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The ratio for all dropout layers.`,name:"dropout_rate"},{anchor:"transformers.LongT5Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-6) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.LongT5Config.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
testing).`,name:"initializer_factor"},{anchor:"transformers.LongT5Config.feed_forward_proj",description:`<strong>feed_forward_proj</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
Type of feed forward layer to be used. Should be one of <code>&quot;relu&quot;</code> or <code>&quot;gated-gelu&quot;</code>. LongT5v1.1 uses the
<code>&quot;gated-gelu&quot;</code> feed forward projection. Original LongT5 implementation uses <code>&quot;gated-gelu&quot;</code>.`,name:"feed_forward_proj"},{anchor:"transformers.LongT5Config.encoder_attention_type",description:`<strong>encoder_attention_type</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;local&quot;</code>) &#x2014;
Type of encoder attention to be used. Should be one of <code>&quot;local&quot;</code> or <code>&quot;transient-global&quot;</code>, which are
supported by LongT5 implementation.`,name:"encoder_attention_type"},{anchor:"transformers.LongT5Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longt5/configuration_longt5.py#L27"}}),ie=new O({props:{title:"LongT5Model",local:"transformers.LongT5Model",headingTag:"h2"}}),de=new _e({props:{name:"class transformers.LongT5Model",anchor:"transformers.LongT5Model",parameters:[{name:"config",val:": LongT5Config"}],parametersDescription:[{anchor:"transformers.LongT5Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config">LongT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longt5/modeling_longt5.py#L1694"}}),le=new _e({props:{name:"forward",anchor:"transformers.LongT5Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.LongT5Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. LongT5 is a model with relative position embeddings so
you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./longt5#training">LONGT5
Training</a>.`,name:"input_ids"},{anchor:"transformers.LongT5Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LongT5Model.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>LONGT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./longt5#training">LONGT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.LongT5Model.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.LongT5Model.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LongT5Model.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.LongT5Model.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.LongT5Model.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.LongT5Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LongT5Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LongT5Model.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.LongT5Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.LongT5Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LongT5Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LongT5Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.LongT5Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longt5/modeling_longt5.py#L1743",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config"
>LongT5Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),H=new Ct({props:{$$slots:{default:[to]},$$scope:{ctx:C}}}),N=new $t({props:{anchor:"transformers.LongT5Model.forward.example",$$slots:{default:[oo]},$$scope:{ctx:C}}}),ce=new O({props:{title:"LongT5ForConditionalGeneration",local:"transformers.LongT5ForConditionalGeneration",headingTag:"h2"}}),pe=new _e({props:{name:"class transformers.LongT5ForConditionalGeneration",anchor:"transformers.LongT5ForConditionalGeneration",parameters:[{name:"config",val:": LongT5Config"}],parametersDescription:[{anchor:"transformers.LongT5ForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config">LongT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longt5/modeling_longt5.py#L1890"}}),he=new _e({props:{name:"forward",anchor:"transformers.LongT5ForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.LongT5ForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. LongT5 is a model with relative position embeddings so
you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./longt5#training">LONGT5
Training</a>.`,name:"input_ids"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>LONGT5 uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>To know more on how to prepare <code>decoder_input_ids</code> for pretraining take a look at <a href="./longt5#training">LONGT5
Training</a>.`,name:"decoder_input_ids"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[-100, 0, ..., config.vocab_size - 1]</code>. All labels set to <code>-100</code> are ignored (masked), the loss is only computed for
labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.LongT5ForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longt5/modeling_longt5.py#L1935",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config"
>LongT5Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),X=new Ct({props:{$$slots:{default:[no]},$$scope:{ctx:C}}}),R=new $t({props:{anchor:"transformers.LongT5ForConditionalGeneration.forward.example",$$slots:{default:[so]},$$scope:{ctx:C}}}),ue=new O({props:{title:"LongT5EncoderModel",local:"transformers.LongT5EncoderModel",headingTag:"h2"}}),me=new _e({props:{name:"class transformers.LongT5EncoderModel",anchor:"transformers.LongT5EncoderModel",parameters:[{name:"config",val:": LongT5Config"}],parametersDescription:[{anchor:"transformers.LongT5EncoderModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config">LongT5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longt5/modeling_longt5.py#L2109"}}),ge=new _e({props:{name:"forward",anchor:"transformers.LongT5EncoderModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.LongT5EncoderModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. LongT5 is a model with relative position embeddings so
you should be able to pad the inputs on both the right and the left.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for detail.</p>
<p>To know more on how to prepare <code>input_ids</code> for pretraining take a look a <a href="./longt5#training">LONGT5
Training</a>.`,name:"input_ids"},{anchor:"transformers.LongT5EncoderModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LongT5EncoderModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.LongT5EncoderModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LongT5EncoderModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.LongT5EncoderModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.LongT5EncoderModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/longt5/modeling_longt5.py#L2147",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/longt5#transformers.LongT5Config"
>LongT5Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),V=new Ct({props:{$$slots:{default:[ao]},$$scope:{ctx:C}}}),E=new $t({props:{anchor:"transformers.LongT5EncoderModel.forward.example",$$slots:{default:[ro]},$$scope:{ctx:C}}}),fe=new eo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/longt5.md"}}),{c(){n=c("meta"),y=s(),l=c("p"),h=s(),M=c("p"),M.innerHTML=r,k=s(),m(P.$$.fragment),Fe=s(),B=c("div"),B.innerHTML=jt,qe=s(),m(Y.$$.fragment),Ie=s(),Q=c("p"),Q.innerHTML=zt,Ze=s(),A=c("p"),A.textContent=Ut,We=s(),D=c("p"),D.innerHTML=xt,Be=s(),K=c("p"),K.innerHTML=Jt,He=s(),m(ee.$$.fragment),Ne=s(),te=c("ul"),te.innerHTML=Gt,Xe=s(),m(oe.$$.fragment),Re=s(),m(ne.$$.fragment),Ve=s(),se=c("ul"),se.innerHTML=Ft,Ee=s(),m(ae.$$.fragment),Se=s(),G=c("div"),m(re.$$.fragment),nt=s(),Te=c("p"),Te.innerHTML=qt,st=s(),be=c("p"),be.innerHTML=It,Oe=s(),m(ie.$$.fragment),Pe=s(),v=c("div"),m(de.$$.fragment),at=s(),ye=c("p"),ye.textContent=Zt,rt=s(),Me=c("p"),Me.innerHTML=Wt,it=s(),ke=c("p"),ke.innerHTML=Bt,dt=s(),U=c("div"),m(le.$$.fragment),lt=s(),ve=c("p"),ve.innerHTML=Ht,ct=s(),m(H.$$.fragment),pt=s(),m(N.$$.fragment),Ye=s(),m(ce.$$.fragment),Qe=s(),w=c("div"),m(pe.$$.fragment),ht=s(),we=c("p"),we.innerHTML=Nt,ut=s(),Le=c("p"),Le.innerHTML=Xt,mt=s(),Ce=c("p"),Ce.innerHTML=Rt,gt=s(),x=c("div"),m(he.$$.fragment),ft=s(),$e=c("p"),$e.innerHTML=Vt,_t=s(),m(X.$$.fragment),Tt=s(),m(R.$$.fragment),Ae=s(),m(ue.$$.fragment),De=s(),L=c("div"),m(me.$$.fragment),bt=s(),je=c("p"),je.textContent=Et,yt=s(),ze=c("p"),ze.innerHTML=St,Mt=s(),Ue=c("p"),Ue.innerHTML=Ot,kt=s(),J=c("div"),m(ge.$$.fragment),vt=s(),xe=c("p"),xe.innerHTML=Pt,wt=s(),m(V.$$.fragment),Lt=s(),m(E.$$.fragment),Ke=s(),m(fe.$$.fragment),et=s(),Ge=c("p"),this.h()},l(e){const t=Kt("svelte-u9bgzb",document.head);n=p(t,"META",{name:!0,content:!0}),t.forEach(o),y=a(e),l=p(e,"P",{}),W(l).forEach(o),h=a(e),M=p(e,"P",{"data-svelte-h":!0}),u(M)!=="svelte-1fpnvle"&&(M.innerHTML=r),k=a(e),g(P.$$.fragment,e),Fe=a(e),B=p(e,"DIV",{class:!0,"data-svelte-h":!0}),u(B)!=="svelte-13t8s2t"&&(B.innerHTML=jt),qe=a(e),g(Y.$$.fragment,e),Ie=a(e),Q=p(e,"P",{"data-svelte-h":!0}),u(Q)!=="svelte-18nscf9"&&(Q.innerHTML=zt),Ze=a(e),A=p(e,"P",{"data-svelte-h":!0}),u(A)!=="svelte-vfdo9a"&&(A.textContent=Ut),We=a(e),D=p(e,"P",{"data-svelte-h":!0}),u(D)!=="svelte-8iibnd"&&(D.innerHTML=xt),Be=a(e),K=p(e,"P",{"data-svelte-h":!0}),u(K)!=="svelte-tbsf6f"&&(K.innerHTML=Jt),He=a(e),g(ee.$$.fragment,e),Ne=a(e),te=p(e,"UL",{"data-svelte-h":!0}),u(te)!=="svelte-zlgcrw"&&(te.innerHTML=Gt),Xe=a(e),g(oe.$$.fragment,e),Re=a(e),g(ne.$$.fragment,e),Ve=a(e),se=p(e,"UL",{"data-svelte-h":!0}),u(se)!=="svelte-6ej6p2"&&(se.innerHTML=Ft),Ee=a(e),g(ae.$$.fragment,e),Se=a(e),G=p(e,"DIV",{class:!0});var Z=W(G);g(re.$$.fragment,Z),nt=a(Z),Te=p(Z,"P",{"data-svelte-h":!0}),u(Te)!=="svelte-15jljwy"&&(Te.innerHTML=qt),st=a(Z),be=p(Z,"P",{"data-svelte-h":!0}),u(be)!=="svelte-1ek1ss9"&&(be.innerHTML=It),Z.forEach(o),Oe=a(e),g(ie.$$.fragment,e),Pe=a(e),v=p(e,"DIV",{class:!0});var $=W(v);g(de.$$.fragment,$),at=a($),ye=p($,"P",{"data-svelte-h":!0}),u(ye)!=="svelte-10n0zy1"&&(ye.textContent=Zt),rt=a($),Me=p($,"P",{"data-svelte-h":!0}),u(Me)!=="svelte-q52n56"&&(Me.innerHTML=Wt),it=a($),ke=p($,"P",{"data-svelte-h":!0}),u(ke)!=="svelte-hswkmf"&&(ke.innerHTML=Bt),dt=a($),U=p($,"DIV",{class:!0});var F=W(U);g(le.$$.fragment,F),lt=a(F),ve=p(F,"P",{"data-svelte-h":!0}),u(ve)!=="svelte-1n53b4q"&&(ve.innerHTML=Ht),ct=a(F),g(H.$$.fragment,F),pt=a(F),g(N.$$.fragment,F),F.forEach(o),$.forEach(o),Ye=a(e),g(ce.$$.fragment,e),Qe=a(e),w=p(e,"DIV",{class:!0});var j=W(w);g(pe.$$.fragment,j),ht=a(j),we=p(j,"P",{"data-svelte-h":!0}),u(we)!=="svelte-1h7i1zh"&&(we.innerHTML=Nt),ut=a(j),Le=p(j,"P",{"data-svelte-h":!0}),u(Le)!=="svelte-q52n56"&&(Le.innerHTML=Xt),mt=a(j),Ce=p(j,"P",{"data-svelte-h":!0}),u(Ce)!=="svelte-hswkmf"&&(Ce.innerHTML=Rt),gt=a(j),x=p(j,"DIV",{class:!0});var q=W(x);g(he.$$.fragment,q),ft=a(q),$e=p(q,"P",{"data-svelte-h":!0}),u($e)!=="svelte-rhng1y"&&($e.innerHTML=Vt),_t=a(q),g(X.$$.fragment,q),Tt=a(q),g(R.$$.fragment,q),q.forEach(o),j.forEach(o),Ae=a(e),g(ue.$$.fragment,e),De=a(e),L=p(e,"DIV",{class:!0});var z=W(L);g(me.$$.fragment,z),bt=a(z),je=p(z,"P",{"data-svelte-h":!0}),u(je)!=="svelte-10n0zy1"&&(je.textContent=Et),yt=a(z),ze=p(z,"P",{"data-svelte-h":!0}),u(ze)!=="svelte-q52n56"&&(ze.innerHTML=St),Mt=a(z),Ue=p(z,"P",{"data-svelte-h":!0}),u(Ue)!=="svelte-hswkmf"&&(Ue.innerHTML=Ot),kt=a(z),J=p(z,"DIV",{class:!0});var S=W(J);g(ge.$$.fragment,S),vt=a(S),xe=p(S,"P",{"data-svelte-h":!0}),u(xe)!=="svelte-1wa095u"&&(xe.innerHTML=Pt),wt=a(S),g(V.$$.fragment,S),Lt=a(S),g(E.$$.fragment,S),S.forEach(o),z.forEach(o),Ke=a(e),g(fe.$$.fragment,e),et=a(e),Ge=p(e,"P",{}),W(Ge).forEach(o),this.h()},h(){I(n,"name","hf:doc:metadata"),I(n,"content",lo),I(B,"class","flex flex-wrap space-x-1"),I(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){d(document.head,n),i(e,y,t),i(e,l,t),i(e,h,t),i(e,M,t),i(e,k,t),f(P,e,t),i(e,Fe,t),i(e,B,t),i(e,qe,t),f(Y,e,t),i(e,Ie,t),i(e,Q,t),i(e,Ze,t),i(e,A,t),i(e,We,t),i(e,D,t),i(e,Be,t),i(e,K,t),i(e,He,t),f(ee,e,t),i(e,Ne,t),i(e,te,t),i(e,Xe,t),f(oe,e,t),i(e,Re,t),f(ne,e,t),i(e,Ve,t),i(e,se,t),i(e,Ee,t),f(ae,e,t),i(e,Se,t),i(e,G,t),f(re,G,null),d(G,nt),d(G,Te),d(G,st),d(G,be),i(e,Oe,t),f(ie,e,t),i(e,Pe,t),i(e,v,t),f(de,v,null),d(v,at),d(v,ye),d(v,rt),d(v,Me),d(v,it),d(v,ke),d(v,dt),d(v,U),f(le,U,null),d(U,lt),d(U,ve),d(U,ct),f(H,U,null),d(U,pt),f(N,U,null),i(e,Ye,t),f(ce,e,t),i(e,Qe,t),i(e,w,t),f(pe,w,null),d(w,ht),d(w,we),d(w,ut),d(w,Le),d(w,mt),d(w,Ce),d(w,gt),d(w,x),f(he,x,null),d(x,ft),d(x,$e),d(x,_t),f(X,x,null),d(x,Tt),f(R,x,null),i(e,Ae,t),f(ue,e,t),i(e,De,t),i(e,L,t),f(me,L,null),d(L,bt),d(L,je),d(L,yt),d(L,ze),d(L,Mt),d(L,Ue),d(L,kt),d(L,J),f(ge,J,null),d(J,vt),d(J,xe),d(J,wt),f(V,J,null),d(J,Lt),f(E,J,null),i(e,Ke,t),f(fe,e,t),i(e,et,t),i(e,Ge,t),tt=!0},p(e,[t]){const Z={};t&2&&(Z.$$scope={dirty:t,ctx:e}),H.$set(Z);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),N.$set($);const F={};t&2&&(F.$$scope={dirty:t,ctx:e}),X.$set(F);const j={};t&2&&(j.$$scope={dirty:t,ctx:e}),R.$set(j);const q={};t&2&&(q.$$scope={dirty:t,ctx:e}),V.$set(q);const z={};t&2&&(z.$$scope={dirty:t,ctx:e}),E.$set(z)},i(e){tt||(_(P.$$.fragment,e),_(Y.$$.fragment,e),_(ee.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(ie.$$.fragment,e),_(de.$$.fragment,e),_(le.$$.fragment,e),_(H.$$.fragment,e),_(N.$$.fragment,e),_(ce.$$.fragment,e),_(pe.$$.fragment,e),_(he.$$.fragment,e),_(X.$$.fragment,e),_(R.$$.fragment,e),_(ue.$$.fragment,e),_(me.$$.fragment,e),_(ge.$$.fragment,e),_(V.$$.fragment,e),_(E.$$.fragment,e),_(fe.$$.fragment,e),tt=!0)},o(e){T(P.$$.fragment,e),T(Y.$$.fragment,e),T(ee.$$.fragment,e),T(oe.$$.fragment,e),T(ne.$$.fragment,e),T(ae.$$.fragment,e),T(re.$$.fragment,e),T(ie.$$.fragment,e),T(de.$$.fragment,e),T(le.$$.fragment,e),T(H.$$.fragment,e),T(N.$$.fragment,e),T(ce.$$.fragment,e),T(pe.$$.fragment,e),T(he.$$.fragment,e),T(X.$$.fragment,e),T(R.$$.fragment,e),T(ue.$$.fragment,e),T(me.$$.fragment,e),T(ge.$$.fragment,e),T(V.$$.fragment,e),T(E.$$.fragment,e),T(fe.$$.fragment,e),tt=!1},d(e){e&&(o(y),o(l),o(h),o(M),o(k),o(Fe),o(B),o(qe),o(Ie),o(Q),o(Ze),o(A),o(We),o(D),o(Be),o(K),o(He),o(Ne),o(te),o(Xe),o(Re),o(Ve),o(se),o(Ee),o(Se),o(G),o(Oe),o(Pe),o(v),o(Ye),o(Qe),o(w),o(Ae),o(De),o(L),o(Ke),o(et),o(Ge)),o(n),b(P,e),b(Y,e),b(ee,e),b(oe,e),b(ne,e),b(ae,e),b(re),b(ie,e),b(de),b(le),b(H),b(N),b(ce,e),b(pe),b(he),b(X),b(R),b(ue,e),b(me),b(ge),b(V),b(E),b(fe,e)}}}const lo='{"title":"LongT5","local":"longt5","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"LongT5Config","local":"transformers.LongT5Config","sections":[],"depth":2},{"title":"LongT5Model","local":"transformers.LongT5Model","sections":[],"depth":2},{"title":"LongT5ForConditionalGeneration","local":"transformers.LongT5ForConditionalGeneration","sections":[],"depth":2},{"title":"LongT5EncoderModel","local":"transformers.LongT5EncoderModel","sections":[],"depth":2}],"depth":1}';function co(C){return Qt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class To extends At{constructor(n){super(),Dt(this,n,co,io,Yt,{})}}export{To as component};
