import{s as nn,o as sn,n as X}from"../chunks/scheduler.18a86fab.js";import{S as rn,i as an,g as p,s as a,r as h,A as dn,h as m,f as o,c as d,j as k,x as v,u,k as R,l as cn,y as i,a as c,v as g,d as _,t as f,w as y}from"../chunks/index.98837b22.js";import{T as St}from"../chunks/Tip.77304350.js";import{D as U}from"../chunks/Docstring.a1ef7999.js";import{C as ht}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Yt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as de,E as ln}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as pn,a as mn}from"../chunks/HfOption.6641485e.js";function hn(w){let t,b='This model was contributed by <a href="https://huggingface.co/ola13" rel="nofollow">ola13</a>.',r,l,T="Click on the RAG models in the right sidebar for more examples of how to apply RAG to different language tasks.";return{c(){t=p("p"),t.innerHTML=b,r=a(),l=p("p"),l.textContent=T},l(s){t=m(s,"P",{"data-svelte-h":!0}),v(t)!=="svelte-1182z55"&&(t.innerHTML=b),r=d(s),l=m(s,"P",{"data-svelte-h":!0}),v(l)!=="svelte-1vrafy5"&&(l.textContent=T)},m(s,M){c(s,t,M),c(s,r,M),c(s,l,M)},p:X,d(s){s&&(o(t),o(r),o(l))}}}function un(w){let t,b;return t=new ht({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwUmFnVG9rZW5pemVyJTJDJTIwUmFnUmV0cmlldmVyJTJDJTIwUmFnU2VxdWVuY2VGb3JHZW5lcmF0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwUmFnVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnJhZy1zZXF1ZW5jZS1ucSUyMiklMEFyZXRyaWV2ZXIlMjAlM0QlMjBSYWdSZXRyaWV2ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmZhY2Vib29rJTJGZHByLWN0eF9lbmNvZGVyLXNpbmdsZS1ucS1iYXNlJTIyJTJDJTIwZGF0YXNldCUzRCUyMndpa2lfZHByJTIyJTJDJTIwaW5kZXhfbmFtZSUzRCUyMmNvbXByZXNzZWQlMjIlMEEpJTBBJTBBbW9kZWwlMjAlM0QlMjBSYWdTZXF1ZW5jZUZvckdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmZhY2Vib29rJTJGcmFnLXRva2VuLW5xJTIyJTJDJTBBJTIwJTIwJTIwJTIwcmV0cmlldmVyJTNEcmV0cmlldmVyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMmZsYXNoX2F0dGVudGlvbl8yJTIyJTJDJTBBKSUwQWlucHV0X2RpY3QlMjAlM0QlMjB0b2tlbml6ZXIucHJlcGFyZV9zZXEyc2VxX2JhdGNoKCUyMkhvdyUyMG1hbnklMjBwZW9wbGUlMjBsaXZlJTIwaW4lMjBQYXJpcyUzRiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBZ2VuZXJhdGVkJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRfaWRzJTNEaW5wdXRfZGljdCU1QiUyMmlucHV0X2lkcyUyMiU1RCklMEFwcmludCh0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlZCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/rag-sequence-nq&quot;</span>)
retriever = RagRetriever.from_pretrained(
    <span class="hljs-string">&quot;facebook/dpr-ctx_encoder-single-nq-base&quot;</span>, dataset=<span class="hljs-string">&quot;wiki_dpr&quot;</span>, index_name=<span class="hljs-string">&quot;compressed&quot;</span>
)

model = RagSequenceForGeneration.from_pretrained(
    <span class="hljs-string">&quot;facebook/rag-token-nq&quot;</span>,
    retriever=retriever,
    dtype=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span>,
)
input_dict = tokenizer.prepare_seq2seq_batch(<span class="hljs-string">&quot;How many people live in Paris?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
generated = model.generate(input_ids=input_dict[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-built_in">print</span>(tokenizer.batch_decode(generated, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>])`,wrap:!1}}),{c(){h(t.$$.fragment)},l(r){u(t.$$.fragment,r)},m(r,l){g(t,r,l),b=!0},p:X,i(r){b||(_(t.$$.fragment,r),b=!0)},o(r){f(t.$$.fragment,r),b=!1},d(r){y(t,r)}}}function gn(w){let t,b;return t=new mn({props:{id:"usage",option:"AutoModel",$$slots:{default:[un]},$$scope:{ctx:w}}}),{c(){h(t.$$.fragment)},l(r){u(t.$$.fragment,r)},m(r,l){g(t,r,l),b=!0},p(r,l){const T={};l&2&&(T.$$scope={dirty:l,ctx:r}),t.$set(T)},i(r){b||(_(t.$$.fragment,r),b=!0)},o(r){f(t.$$.fragment,r),b=!1},d(r){y(t,r)}}}function _n(w){let t,b="Examples:",r,l,T;return l=new ht({props:{code:"JTIzJTIwVG8lMjBsb2FkJTIwdGhlJTIwZGVmYXVsdCUyMCUyMndpa2lfZHByJTIyJTIwZGF0YXNldCUyMHdpdGglMjAyMU0lMjBwYXNzYWdlcyUyMGZyb20lMjB3aWtpcGVkaWElMjAoaW5kZXglMjBuYW1lJTIwaXMlMjAnY29tcHJlc3NlZCclMjBvciUyMCdleGFjdCcpJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJhZ1JldHJpZXZlciUwQSUwQXJldHJpZXZlciUyMCUzRCUyMFJhZ1JldHJpZXZlci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZmFjZWJvb2slMkZkcHItY3R4X2VuY29kZXItc2luZ2xlLW5xLWJhc2UlMjIlMkMlMjBkYXRhc2V0JTNEJTIyd2lraV9kcHIlMjIlMkMlMjBpbmRleF9uYW1lJTNEJTIyY29tcHJlc3NlZCUyMiUwQSklMEElMEElMjMlMjBUbyUyMGxvYWQlMjB5b3VyJTIwb3duJTIwaW5kZXhlZCUyMGRhdGFzZXQlMjBidWlsdCUyMHdpdGglMjB0aGUlMjBkYXRhc2V0cyUyMGxpYnJhcnkuJTIwTW9yZSUyMGluZm8lMjBvbiUyMGhvdyUyMHRvJTIwYnVpbGQlMjB0aGUlMjBpbmRleGVkJTIwZGF0YXNldCUyMGluJTIwZXhhbXBsZXMlMkZyYWclMkZ1c2Vfb3duX2tub3dsZWRnZV9kYXRhc2V0LnB5JTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJhZ1JldHJpZXZlciUwQSUwQWRhdGFzZXQlMjAlM0QlMjAoJTBBJTIwJTIwJTIwJTIwLi4uJTBBKSUyMCUyMCUyMyUyMGRhdGFzZXQlMjBtdXN0JTIwYmUlMjBhJTIwZGF0YXNldHMuRGF0YXNldHMlMjBvYmplY3QlMjB3aXRoJTIwY29sdW1ucyUyMCUyMnRpdGxlJTIyJTJDJTIwJTIydGV4dCUyMiUyMGFuZCUyMCUyMmVtYmVkZGluZ3MlMjIlMkMlMjBhbmQlMjBpdCUyMG11c3QlMjBoYXZlJTIwYSUyMHN1cHBvcnRlZCUyMGluZGV4JTIwKGUuZy4lMkMlMjBGYWlzcyUyMG9yJTIwb3RoZXIlMjBpbmRleCUyMHR5cGVzJTIwZGVwZW5kaW5nJTIwb24lMjB5b3VyJTIwc2V0dXApJTBBcmV0cmlldmVyJTIwJTNEJTIwUmFnUmV0cmlldmVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRwci1jdHhfZW5jb2Rlci1zaW5nbGUtbnEtYmFzZSUyMiUyQyUyMGluZGV4ZWRfZGF0YXNldCUzRGRhdGFzZXQpJTBBJTBBJTIzJTIwVG8lMjBsb2FkJTIweW91ciUyMG93biUyMGluZGV4ZWQlMjBkYXRhc2V0JTIwYnVpbHQlMjB3aXRoJTIwdGhlJTIwZGF0YXNldHMlMjBsaWJyYXJ5JTIwdGhhdCUyMHdhcyUyMHNhdmVkJTIwb24lMjBkaXNrLiUyME1vcmUlMjBpbmZvJTIwaW4lMjBleGFtcGxlcyUyRnJhZyUyRnVzZV9vd25fa25vd2xlZGdlX2RhdGFzZXQucHklMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwUmFnUmV0cmlldmVyJTBBJTBBZGF0YXNldF9wYXRoJTIwJTNEJTIwJTIycGF0aCUyRnRvJTJGbXklMkZkYXRhc2V0JTIyJTIwJTIwJTIzJTIwZGF0YXNldCUyMHNhdmVkJTIwdmlhJTIwKmRhdGFzZXQuc2F2ZV90b19kaXNrKC4uLikqJTBBaW5kZXhfcGF0aCUyMCUzRCUyMCUyMnBhdGglMkZ0byUyRm15JTJGaW5kZXglMjIlMjAlMjAlMjMlMjBpbmRleCUyMHNhdmVkJTIwdmlhJTIwKmRhdGFzZXQuZ2V0X2luZGV4KCUyMmVtYmVkZGluZ3MlMjIpLnNhdmUoLi4uKSolMEFyZXRyaWV2ZXIlMjAlM0QlMjBSYWdSZXRyaWV2ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmZhY2Vib29rJTJGZHByLWN0eF9lbmNvZGVyLXNpbmdsZS1ucS1iYXNlJTIyJTJDJTBBJTIwJTIwJTIwJTIwaW5kZXhfbmFtZSUzRCUyMmN1c3RvbSUyMiUyQyUwQSUyMCUyMCUyMCUyMHBhc3NhZ2VzX3BhdGglM0RkYXRhc2V0X3BhdGglMkMlMEElMjAlMjAlMjAlMjBpbmRleF9wYXRoJTNEaW5kZXhfcGF0aCUyQyUwQSklMEElMEElMjMlMjBUbyUyMGxvYWQlMjB0aGUlMjBsZWdhY3klMjBpbmRleCUyMGJ1aWx0JTIwb3JpZ2luYWxseSUyMGZvciUyMFJhZydzJTIwcGFwZXIlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwUmFnUmV0cmlldmVyJTBBJTBBcmV0cmlldmVyJTIwJTNEJTIwUmFnUmV0cmlldmVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmRwci1jdHhfZW5jb2Rlci1zaW5nbGUtbnEtYmFzZSUyMiUyQyUyMGluZGV4X25hbWUlM0QlMjJsZWdhY3klMjIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To load the default &quot;wiki_dpr&quot; dataset with 21M passages from wikipedia (index name is &#x27;compressed&#x27; or &#x27;exact&#x27;)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RagRetriever

<span class="hljs-meta">&gt;&gt;&gt; </span>retriever = RagRetriever.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/dpr-ctx_encoder-single-nq-base&quot;</span>, dataset=<span class="hljs-string">&quot;wiki_dpr&quot;</span>, index_name=<span class="hljs-string">&quot;compressed&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RagRetriever

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = (
<span class="hljs-meta">... </span>    ...
<span class="hljs-meta">... </span>)  <span class="hljs-comment"># dataset must be a datasets.Datasets object with columns &quot;title&quot;, &quot;text&quot; and &quot;embeddings&quot;, and it must have a supported index (e.g., Faiss or other index types depending on your setup)</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>retriever = RagRetriever.from_pretrained(<span class="hljs-string">&quot;facebook/dpr-ctx_encoder-single-nq-base&quot;</span>, indexed_dataset=dataset)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RagRetriever

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset_path = <span class="hljs-string">&quot;path/to/my/dataset&quot;</span>  <span class="hljs-comment"># dataset saved via *dataset.save_to_disk(...)*</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>index_path = <span class="hljs-string">&quot;path/to/my/index&quot;</span>  <span class="hljs-comment"># index saved via *dataset.get_index(&quot;embeddings&quot;).save(...)*</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>retriever = RagRetriever.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/dpr-ctx_encoder-single-nq-base&quot;</span>,
<span class="hljs-meta">... </span>    index_name=<span class="hljs-string">&quot;custom&quot;</span>,
<span class="hljs-meta">... </span>    passages_path=dataset_path,
<span class="hljs-meta">... </span>    index_path=index_path,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To load the legacy index built originally for Rag&#x27;s paper</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RagRetriever

<span class="hljs-meta">&gt;&gt;&gt; </span>retriever = RagRetriever.from_pretrained(<span class="hljs-string">&quot;facebook/dpr-ctx_encoder-single-nq-base&quot;</span>, index_name=<span class="hljs-string">&quot;legacy&quot;</span>)`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,r=a(),h(l.$$.fragment)},l(s){t=m(s,"P",{"data-svelte-h":!0}),v(t)!=="svelte-kvfsh7"&&(t.textContent=b),r=d(s),u(l.$$.fragment,s)},m(s,M){c(s,t,M),c(s,r,M),g(l,s,M),T=!0},p:X,i(s){T||(_(l.$$.fragment,s),T=!0)},o(s){f(l.$$.fragment,s),T=!1},d(s){s&&(o(t),o(r)),y(l,s)}}}function fn(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(r){t=m(r,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(r,l){c(r,t,l)},p:X,d(r){r&&o(t)}}}function yn(w){let t,b="Example:",r,l,T;return l=new ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSYWdSZXRyaWV2ZXIlMkMlMjBSYWdNb2RlbCUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZyYWctdG9rZW4tYmFzZSUyMiklMEFyZXRyaWV2ZXIlMjAlM0QlMjBSYWdSZXRyaWV2ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmZhY2Vib29rJTJGcmFnLXRva2VuLWJhc2UlMjIlMkMlMjBpbmRleF9uYW1lJTNEJTIyZXhhY3QlMjIlMkMlMjB1c2VfZHVtbXlfZGF0YXNldCUzRFRydWUlMEEpJTBBJTIzJTIwaW5pdGlhbGl6ZSUyMHdpdGglMjBSYWdSZXRyaWV2ZXIlMjB0byUyMGRvJTIwZXZlcnl0aGluZyUyMGluJTIwb25lJTIwZm9yd2FyZCUyMGNhbGwlMEFtb2RlbCUyMCUzRCUyMFJhZ01vZGVsLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnJhZy10b2tlbi1iYXNlJTIyJTJDJTIwcmV0cmlldmVyJTNEcmV0cmlldmVyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIb3clMjBtYW55JTIwcGVvcGxlJTIwbGl2ZSUyMGluJTIwUGFyaXMlM0YlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RagRetriever, RagModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/rag-token-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>retriever = RagRetriever.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/rag-token-base&quot;</span>, index_name=<span class="hljs-string">&quot;exact&quot;</span>, use_dummy_dataset=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># initialize with RagRetriever to do everything in one forward call</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RagModel.from_pretrained(<span class="hljs-string">&quot;facebook/rag-token-base&quot;</span>, retriever=retriever)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;How many people live in Paris?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,r=a(),h(l.$$.fragment)},l(s){t=m(s,"P",{"data-svelte-h":!0}),v(t)!=="svelte-11lpom8"&&(t.textContent=b),r=d(s),u(l.$$.fragment,s)},m(s,M){c(s,t,M),c(s,r,M),g(l,s,M),T=!0},p:X,i(s){T||(_(l.$$.fragment,s),T=!0)},o(s){f(l.$$.fragment,s),T=!1},d(s){s&&(o(t),o(r)),y(l,s)}}}function bn(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(r){t=m(r,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(r,l){c(r,t,l)},p:X,d(r){r&&o(t)}}}function vn(w){let t,b="Example:",r,l,T;return l=new ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSYWdSZXRyaWV2ZXIlMkMlMjBSYWdTZXF1ZW5jZUZvckdlbmVyYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGcmFnLXNlcXVlbmNlLW5xJTIyKSUwQXJldHJpZXZlciUyMCUzRCUyMFJhZ1JldHJpZXZlci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZmFjZWJvb2slMkZyYWctc2VxdWVuY2UtbnElMjIlMkMlMjBpbmRleF9uYW1lJTNEJTIyZXhhY3QlMjIlMkMlMjB1c2VfZHVtbXlfZGF0YXNldCUzRFRydWUlMEEpJTBBJTIzJTIwaW5pdGlhbGl6ZSUyMHdpdGglMjBSYWdSZXRyaWV2ZXIlMjB0byUyMGRvJTIwZXZlcnl0aGluZyUyMGluJTIwb25lJTIwZm9yd2FyZCUyMGNhbGwlMEFtb2RlbCUyMCUzRCUyMFJhZ1NlcXVlbmNlRm9yR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZyYWctdG9rZW4tbnElMjIlMkMlMjByZXRyaWV2ZXIlM0RyZXRyaWV2ZXIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhvdyUyMG1hbnklMjBwZW9wbGUlMjBsaXZlJTIwaW4lMjBQYXJpcyUzRiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBdGFyZ2V0cyUyMCUzRCUyMHRva2VuaXplcih0ZXh0X3RhcmdldCUzRCUyMkluJTIwUGFyaXMlMkMlMjB0aGVyZSUyMGFyZSUyMDEwJTIwbWlsbGlvbiUyMHBlb3BsZS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWlucHV0X2lkcyUyMCUzRCUyMGlucHV0cyU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQWxhYmVscyUyMCUzRCUyMHRhcmdldHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQSUwQSUyMyUyMG9yJTIwdXNlJTIwcmV0cmlldmVyJTIwc2VwYXJhdGVseSUwQW1vZGVsJTIwJTNEJTIwUmFnU2VxdWVuY2VGb3JHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnJhZy1zZXF1ZW5jZS1ucSUyMiUyQyUyMHVzZV9kdW1teV9kYXRhc2V0JTNEVHJ1ZSklMEElMjMlMjAxLiUyMEVuY29kZSUwQXF1ZXN0aW9uX2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBtb2RlbC5xdWVzdGlvbl9lbmNvZGVyKGlucHV0X2lkcyklNUIwJTVEJTBBJTIzJTIwMi4lMjBSZXRyaWV2ZSUwQWRvY3NfZGljdCUyMCUzRCUyMHJldHJpZXZlcihpbnB1dF9pZHMubnVtcHkoKSUyQyUyMHF1ZXN0aW9uX2hpZGRlbl9zdGF0ZXMuZGV0YWNoKCkubnVtcHkoKSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBZG9jX3Njb3JlcyUyMCUzRCUyMHRvcmNoLmJtbSglMEElMjAlMjAlMjAlMjBxdWVzdGlvbl9oaWRkZW5fc3RhdGVzLnVuc3F1ZWV6ZSgxKSUyQyUyMGRvY3NfZGljdCU1QiUyMnJldHJpZXZlZF9kb2NfZW1iZWRzJTIyJTVELmZsb2F0KCkudHJhbnNwb3NlKDElMkMlMjAyKSUwQSkuc3F1ZWV6ZSgxKSUwQSUyMyUyMDMuJTIwRm9yd2FyZCUyMHRvJTIwZ2VuZXJhdG9yJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMGNvbnRleHRfaW5wdXRfaWRzJTNEZG9jc19kaWN0JTVCJTIyY29udGV4dF9pbnB1dF9pZHMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBjb250ZXh0X2F0dGVudGlvbl9tYXNrJTNEZG9jc19kaWN0JTVCJTIyY29udGV4dF9hdHRlbnRpb25fbWFzayUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGRvY19zY29yZXMlM0Rkb2Nfc2NvcmVzJTJDJTBBJTIwJTIwJTIwJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RsYWJlbHMlMkMlMEEp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RagRetriever, RagSequenceForGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/rag-sequence-nq&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>retriever = RagRetriever.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/rag-sequence-nq&quot;</span>, index_name=<span class="hljs-string">&quot;exact&quot;</span>, use_dummy_dataset=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># initialize with RagRetriever to do everything in one forward call</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RagSequenceForGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/rag-token-nq&quot;</span>, retriever=retriever)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;How many people live in Paris?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>targets = tokenizer(text_target=<span class="hljs-string">&quot;In Paris, there are 10 million people.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = inputs[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = targets[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, labels=labels)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># or use retriever separately</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RagSequenceForGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/rag-sequence-nq&quot;</span>, use_dummy_dataset=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># 1. Encode</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>question_hidden_states = model.question_encoder(input_ids)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># 2. Retrieve</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>doc_scores = torch.bmm(
<span class="hljs-meta">... </span>    question_hidden_states.unsqueeze(<span class="hljs-number">1</span>), docs_dict[<span class="hljs-string">&quot;retrieved_doc_embeds&quot;</span>].<span class="hljs-built_in">float</span>().transpose(<span class="hljs-number">1</span>, <span class="hljs-number">2</span>)
<span class="hljs-meta">... </span>).squeeze(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># 3. Forward to generator</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    context_input_ids=docs_dict[<span class="hljs-string">&quot;context_input_ids&quot;</span>],
<span class="hljs-meta">... </span>    context_attention_mask=docs_dict[<span class="hljs-string">&quot;context_attention_mask&quot;</span>],
<span class="hljs-meta">... </span>    doc_scores=doc_scores,
<span class="hljs-meta">... </span>    decoder_input_ids=labels,
<span class="hljs-meta">... </span>)`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,r=a(),h(l.$$.fragment)},l(s){t=m(s,"P",{"data-svelte-h":!0}),v(t)!=="svelte-11lpom8"&&(t.textContent=b),r=d(s),u(l.$$.fragment,s)},m(s,M){c(s,t,M),c(s,r,M),g(l,s,M),T=!0},p:X,i(s){T||(_(l.$$.fragment,s),T=!0)},o(s){f(l.$$.fragment,s),T=!1},d(s){s&&(o(t),o(r)),y(l,s)}}}function Tn(w){let t,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=b},l(r){t=m(r,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=b)},m(r,l){c(r,t,l)},p:X,d(r){r&&o(t)}}}function Mn(w){let t,b="Example:",r,l,T;return l=new ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSYWdSZXRyaWV2ZXIlMkMlMjBSYWdUb2tlbkZvckdlbmVyYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGcmFnLXRva2VuLW5xJTIyKSUwQXJldHJpZXZlciUyMCUzRCUyMFJhZ1JldHJpZXZlci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZmFjZWJvb2slMkZyYWctdG9rZW4tbnElMjIlMkMlMjBpbmRleF9uYW1lJTNEJTIyZXhhY3QlMjIlMkMlMjB1c2VfZHVtbXlfZGF0YXNldCUzRFRydWUlMEEpJTBBJTIzJTIwaW5pdGlhbGl6ZSUyMHdpdGglMjBSYWdSZXRyaWV2ZXIlMjB0byUyMGRvJTIwZXZlcnl0aGluZyUyMGluJTIwb25lJTIwZm9yd2FyZCUyMGNhbGwlMEFtb2RlbCUyMCUzRCUyMFJhZ1Rva2VuRm9yR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZyYWctdG9rZW4tbnElMjIlMkMlMjByZXRyaWV2ZXIlM0RyZXRyaWV2ZXIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhvdyUyMG1hbnklMjBwZW9wbGUlMjBsaXZlJTIwaW4lMjBQYXJpcyUzRiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBdGFyZ2V0cyUyMCUzRCUyMHRva2VuaXplcih0ZXh0X3RhcmdldCUzRCUyMkluJTIwUGFyaXMlMkMlMjB0aGVyZSUyMGFyZSUyMDEwJTIwbWlsbGlvbiUyMHBlb3BsZS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWlucHV0X2lkcyUyMCUzRCUyMGlucHV0cyU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQWxhYmVscyUyMCUzRCUyMHRhcmdldHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQSUwQSUyMyUyMG9yJTIwdXNlJTIwcmV0cmlldmVyJTIwc2VwYXJhdGVseSUwQW1vZGVsJTIwJTNEJTIwUmFnVG9rZW5Gb3JHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnJhZy10b2tlbi1ucSUyMiUyQyUyMHVzZV9kdW1teV9kYXRhc2V0JTNEVHJ1ZSklMEElMjMlMjAxLiUyMEVuY29kZSUwQXF1ZXN0aW9uX2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBtb2RlbC5xdWVzdGlvbl9lbmNvZGVyKGlucHV0X2lkcyklNUIwJTVEJTBBJTIzJTIwMi4lMjBSZXRyaWV2ZSUwQWRvY3NfZGljdCUyMCUzRCUyMHJldHJpZXZlcihpbnB1dF9pZHMubnVtcHkoKSUyQyUyMHF1ZXN0aW9uX2hpZGRlbl9zdGF0ZXMuZGV0YWNoKCkubnVtcHkoKSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBZG9jX3Njb3JlcyUyMCUzRCUyMHRvcmNoLmJtbSglMEElMjAlMjAlMjAlMjBxdWVzdGlvbl9oaWRkZW5fc3RhdGVzLnVuc3F1ZWV6ZSgxKSUyQyUyMGRvY3NfZGljdCU1QiUyMnJldHJpZXZlZF9kb2NfZW1iZWRzJTIyJTVELmZsb2F0KCkudHJhbnNwb3NlKDElMkMlMjAyKSUwQSkuc3F1ZWV6ZSgxKSUwQSUyMyUyMDMuJTIwRm9yd2FyZCUyMHRvJTIwZ2VuZXJhdG9yJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMGNvbnRleHRfaW5wdXRfaWRzJTNEZG9jc19kaWN0JTVCJTIyY29udGV4dF9pbnB1dF9pZHMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBjb250ZXh0X2F0dGVudGlvbl9tYXNrJTNEZG9jc19kaWN0JTVCJTIyY29udGV4dF9hdHRlbnRpb25fbWFzayUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGRvY19zY29yZXMlM0Rkb2Nfc2NvcmVzJTJDJTBBJTIwJTIwJTIwJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RsYWJlbHMlMkMlMEEpJTBBJTBBJTIzJTIwb3IlMjBkaXJlY3RseSUyMGdlbmVyYXRlJTBBZ2VuZXJhdGVkJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoJTBBJTIwJTIwJTIwJTIwY29udGV4dF9pbnB1dF9pZHMlM0Rkb2NzX2RpY3QlNUIlMjJjb250ZXh0X2lucHV0X2lkcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGNvbnRleHRfYXR0ZW50aW9uX21hc2slM0Rkb2NzX2RpY3QlNUIlMjJjb250ZXh0X2F0dGVudGlvbl9tYXNrJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwZG9jX3Njb3JlcyUzRGRvY19zY29yZXMlMkMlMEEpJTBBZ2VuZXJhdGVkX3N0cmluZyUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RagRetriever, RagTokenForGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/rag-token-nq&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>retriever = RagRetriever.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/rag-token-nq&quot;</span>, index_name=<span class="hljs-string">&quot;exact&quot;</span>, use_dummy_dataset=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># initialize with RagRetriever to do everything in one forward call</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RagTokenForGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/rag-token-nq&quot;</span>, retriever=retriever)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;How many people live in Paris?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>targets = tokenizer(text_target=<span class="hljs-string">&quot;In Paris, there are 10 million people.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = inputs[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = targets[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, labels=labels)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># or use retriever separately</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RagTokenForGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/rag-token-nq&quot;</span>, use_dummy_dataset=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># 1. Encode</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>question_hidden_states = model.question_encoder(input_ids)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># 2. Retrieve</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>doc_scores = torch.bmm(
<span class="hljs-meta">... </span>    question_hidden_states.unsqueeze(<span class="hljs-number">1</span>), docs_dict[<span class="hljs-string">&quot;retrieved_doc_embeds&quot;</span>].<span class="hljs-built_in">float</span>().transpose(<span class="hljs-number">1</span>, <span class="hljs-number">2</span>)
<span class="hljs-meta">... </span>).squeeze(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># 3. Forward to generator</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    context_input_ids=docs_dict[<span class="hljs-string">&quot;context_input_ids&quot;</span>],
<span class="hljs-meta">... </span>    context_attention_mask=docs_dict[<span class="hljs-string">&quot;context_attention_mask&quot;</span>],
<span class="hljs-meta">... </span>    doc_scores=doc_scores,
<span class="hljs-meta">... </span>    decoder_input_ids=labels,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># or directly generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generated = model.generate(
<span class="hljs-meta">... </span>    context_input_ids=docs_dict[<span class="hljs-string">&quot;context_input_ids&quot;</span>],
<span class="hljs-meta">... </span>    context_attention_mask=docs_dict[<span class="hljs-string">&quot;context_attention_mask&quot;</span>],
<span class="hljs-meta">... </span>    doc_scores=doc_scores,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_string = tokenizer.batch_decode(generated, skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),{c(){t=p("p"),t.textContent=b,r=a(),h(l.$$.fragment)},l(s){t=m(s,"P",{"data-svelte-h":!0}),v(t)!=="svelte-11lpom8"&&(t.textContent=b),r=d(s),u(l.$$.fragment,s)},m(s,M){c(s,t,M),c(s,r,M),g(l,s,M),T=!0},p:X,i(s){T||(_(l.$$.fragment,s),T=!0)},o(s){f(l.$$.fragment,s),T=!1},d(s){s&&(o(t),o(r)),y(l,s)}}}function wn(w){let t,b,r,l,T,s="<em>This model was released on 2020-05-22 and added to Hugging Face Transformers on 2020-11-16.</em>",M,ie,ft,B,zo='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/></div>',yt,ce,jo='<a href="https://huggingface.co/papers/2005.11401" rel="nofollow">Retrieval-Augmented Generation (RAG)</a> combines a pretrained language model (parametric memory) with access to an external data source (non-parametric memory) by means of a pretrained neural retriever. RAG fetches relevant passages and conditions its generation on them during inference. This often makes the answers more factual and lets you update knowledge by changing the index instead of retraining the whole model.',bt,le,Fo='You can find all the original RAG checkpoints under the <a href="https://huggingface.co/facebook/models?search=rag" rel="nofollow">AI at Meta</a> organization.',vt,L,Tt,pe,Io='The examples below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>.',Mt,H,wt,me,Go=`Quantization reduces memory by storing weights in lower precision. See the <a href="../quantization/overview">Quantization</a> overview for supported backends.
The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to quantize the weights to 4-bits.`,kt,he,Rt,ue,xt,C,ge,At,Ye,$o=`<a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig">RagConfig</a> stores the configuration of a <em>RagModel</em>. Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and
can be used to control the model outputs. Read the documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ot,S,_e,Et,Ae,Co=`Instantiate a <a href="/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig">EncoderDecoderConfig</a> (or a derived class) from a pre-trained encoder model configuration and
decoder model configuration.`,Jt,fe,qt,ye,be,Ut,ve,Zt,W,Te,Qt,Oe,Vo="Base class for retriever augmented marginalized models outputs.",zt,Me,we,jt,ke,Ft,x,Re,Pt,Ee,Wo=`Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
contents, and it formats them to be used with a RagModel.`,Dt,Y,Kt,A,xe,eo,Qe,No="Retriever initialization function. It loads the index into memory.",to,O,Je,oo,Pe,Xo="Postprocessing retrieved <code>docs</code> and combining them with <code>input_strings</code>.",no,E,qe,so,De,Bo="Retrieves documents for specified <code>question_hidden_states</code>.",It,Ue,Gt,Z,Ze,ro,Ke,Lo="The bare Rag Model outputting raw hidden-states without any specific head on top.",ao,et,Ho=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,io,tt,So=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,co,j,ze,lo,ot,Yo='The <a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagModel">RagModel</a> forward method, overrides the <code>__call__</code> special method.',po,Q,mo,P,$t,je,Ct,J,Fe,ho,nt,Ao="A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.",uo,st,Oo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,go,rt,Eo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,_o,F,Ie,fo,at,Qo='The <a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagSequenceForGeneration">RagSequenceForGeneration</a> forward method, overrides the <code>__call__</code> special method.',yo,D,bo,K,vo,ee,Ge,To,dt,Po='Implements RAG sequence “thorough” decoding. Read the <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate">generate()</a>` documentation\nfor more information on how to set other generate input parameters.',Vt,$e,Wt,q,Ce,Mo,it,Do="A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.",wo,ct,Ko=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ko,lt,en=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ro,I,Ve,xo,pt,tn='The <a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagTokenForGeneration">RagTokenForGeneration</a> forward method, overrides the <code>__call__</code> special method.',Jo,te,qo,oe,Uo,ne,We,Zo,mt,on="Implements RAG token decoding.",Nt,Ne,Xt,ut,Bt;return ie=new de({props:{title:"RAG",local:"rag",headingTag:"h1"}}),L=new St({props:{warning:!1,$$slots:{default:[hn]},$$scope:{ctx:w}}}),H=new pn({props:{id:"usage",options:["AutoModel"],$$slots:{default:[gn]},$$scope:{ctx:w}}}),he=new ht({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwUmFnVG9rZW5pemVyJTJDJTIwUmFnUmV0cmlldmVyJTJDJTIwUmFnU2VxdWVuY2VGb3JHZW5lcmF0aW9uJTBBJTBBYm5iJTIwJTNEJTIwQml0c0FuZEJ5dGVzQ29uZmlnKGxvYWRfaW5fNGJpdCUzRFRydWUlMkMlMjBibmJfNGJpdF9jb21wdXRlX2R0eXBlJTNEdG9yY2guYmZsb2F0MTYpJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwUmFnVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnJhZy1zZXF1ZW5jZS1ucSUyMiklMEFyZXRyaWV2ZXIlMjAlM0QlMjBSYWdSZXRyaWV2ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmZhY2Vib29rJTJGZHByLWN0eF9lbmNvZGVyLXNpbmdsZS1ucS1iYXNlJTIyJTJDJTIwZGF0YXNldCUzRCUyMndpa2lfZHByJTIyJTJDJTIwaW5kZXhfbmFtZSUzRCUyMmNvbXByZXNzZWQlMjIlMEEpJTBBJTBBbW9kZWwlMjAlM0QlMjBSYWdTZXF1ZW5jZUZvckdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmZhY2Vib29rJTJGcmFnLXRva2VuLW5xJTIyJTJDJTBBJTIwJTIwJTIwJTIwcmV0cmlldmVyJTNEcmV0cmlldmVyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRGJuYiUyQyUyMCUyMCUyMCUyMyUyMHF1YW50aXplcyUyMGdlbmVyYXRvciUyMHdlaWdodHMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSklMEFpbnB1dF9kaWN0JTIwJTNEJTIwdG9rZW5pemVyLnByZXBhcmVfc2VxMnNlcV9iYXRjaCglMjJIb3clMjBtYW55JTIwcGVvcGxlJTIwbGl2ZSUyMGluJTIwUGFyaXMlM0YlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWdlbmVyYXRlZCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0X2lkcyUzRGlucHV0X2RpY3QlNUIlMjJpbnB1dF9pZHMlMjIlNUQpJTBBcHJpbnQodG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZWQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklNUIwJTVEKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, RagTokenizer, RagRetriever, RagSequenceForGeneration

bnb = BitsAndBytesConfig(load_in_4bit=<span class="hljs-literal">True</span>, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = RagTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/rag-sequence-nq&quot;</span>)
retriever = RagRetriever.from_pretrained(
    <span class="hljs-string">&quot;facebook/dpr-ctx_encoder-single-nq-base&quot;</span>, dataset=<span class="hljs-string">&quot;wiki_dpr&quot;</span>, index_name=<span class="hljs-string">&quot;compressed&quot;</span>
)

model = RagSequenceForGeneration.from_pretrained(
    <span class="hljs-string">&quot;facebook/rag-token-nq&quot;</span>,
    retriever=retriever,
    quantization_config=bnb,   <span class="hljs-comment"># quantizes generator weights</span>
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)
input_dict = tokenizer.prepare_seq2seq_batch(<span class="hljs-string">&quot;How many people live in Paris?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
generated = model.generate(input_ids=input_dict[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-built_in">print</span>(tokenizer.batch_decode(generated, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>])`,wrap:!1}}),ue=new de({props:{title:"RagConfig",local:"transformers.RagConfig",headingTag:"h2"}}),ge=new U({props:{name:"class transformers.RagConfig",anchor:"transformers.RagConfig",parameters:[{name:"vocab_size",val:" = None"},{name:"is_encoder_decoder",val:" = True"},{name:"prefix",val:" = None"},{name:"bos_token_id",val:" = None"},{name:"pad_token_id",val:" = None"},{name:"eos_token_id",val:" = None"},{name:"decoder_start_token_id",val:" = None"},{name:"title_sep",val:" = ' / '"},{name:"doc_sep",val:" = ' // '"},{name:"n_docs",val:" = 5"},{name:"max_combined_length",val:" = 300"},{name:"retrieval_vector_size",val:" = 768"},{name:"retrieval_batch_size",val:" = 8"},{name:"dataset",val:" = 'wiki_dpr'"},{name:"dataset_split",val:" = 'train'"},{name:"index_name",val:" = 'compressed'"},{name:"index_path",val:" = None"},{name:"passages_path",val:" = None"},{name:"use_dummy_dataset",val:" = False"},{name:"reduce_loss",val:" = False"},{name:"label_smoothing",val:" = 0.0"},{name:"do_deduplication",val:" = True"},{name:"exclude_bos_score",val:" = False"},{name:"do_marginalize",val:" = False"},{name:"output_retrieved",val:" = False"},{name:"use_cache",val:" = True"},{name:"forced_eos_token_id",val:" = None"},{name:"dataset_revision",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RagConfig.title_sep",description:`<strong>title_sep</strong> (<code>str</code>, <em>optional</em>, defaults to  <code>&quot; / &quot;</code>) &#x2014;
Separator inserted between the title and the text of the retrieved document when calling <a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagRetriever">RagRetriever</a>.`,name:"title_sep"},{anchor:"transformers.RagConfig.doc_sep",description:`<strong>doc_sep</strong> (<code>str</code>, <em>optional</em>, defaults to  <code>&quot; // &quot;</code>) &#x2014;
Separator inserted between the text of the retrieved document and the original input when calling
<a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagRetriever">RagRetriever</a>.`,name:"doc_sep"},{anchor:"transformers.RagConfig.n_docs",description:`<strong>n_docs</strong> (<code>int</code>, <em>optional</em>, defaults to 5) &#x2014;
Number of documents to retrieve.`,name:"n_docs"},{anchor:"transformers.RagConfig.max_combined_length",description:`<strong>max_combined_length</strong> (<code>int</code>, <em>optional</em>, defaults to 300) &#x2014;
Max length of contextualized input returned by <code>__call__()</code>.`,name:"max_combined_length"},{anchor:"transformers.RagConfig.retrieval_vector_size",description:`<strong>retrieval_vector_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the document embeddings indexed by <a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagRetriever">RagRetriever</a>.`,name:"retrieval_vector_size"},{anchor:"transformers.RagConfig.retrieval_batch_size",description:`<strong>retrieval_batch_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
<a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagRetriever">RagRetriever</a>.`,name:"retrieval_batch_size"},{anchor:"transformers.RagConfig.dataset",description:`<strong>dataset</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;wiki_dpr&quot;</code>) &#x2014;
A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
using <code>datasets.list_datasets()</code>).`,name:"dataset"},{anchor:"transformers.RagConfig.dataset_split",description:`<strong>dataset_split</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;train&quot;</code>) &#x2014;
Which split of the <code>dataset</code> to load.`,name:"dataset_split"},{anchor:"transformers.RagConfig.index_name",description:`<strong>index_name</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;compressed&quot;</code>) &#x2014;
The index name of the index associated with the <code>dataset</code>. One can choose between <code>&quot;legacy&quot;</code>, <code>&quot;exact&quot;</code> and
<code>&quot;compressed&quot;</code>.`,name:"index_name"},{anchor:"transformers.RagConfig.index_path",description:`<strong>index_path</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The path to the serialized faiss index on disk.`,name:"index_path"},{anchor:"transformers.RagConfig.passages_path",description:`<strong>passages_path</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A path to text passages compatible with the faiss index. Required if using
<code>LegacyIndex</code>`,name:"passages_path"},{anchor:"transformers.RagConfig.use_dummy_dataset",description:`<strong>use_dummy_dataset</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to load a &#x201C;dummy&#x201D; variant of the dataset specified by <code>dataset</code>.`,name:"use_dummy_dataset"},{anchor:"transformers.RagConfig.label_smoothing",description:`<strong>label_smoothing</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Only relevant if <code>return_loss</code> is set to <code>True</code>. Controls the <code>epsilon</code> parameter value for label smoothing
in the loss calculation. If set to 0, no label smoothing is performed.`,name:"label_smoothing"},{anchor:"transformers.RagConfig.do_marginalize",description:`<strong>do_marginalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If <code>True</code>, the logits are marginalized over all documents by making use of
<code>torch.nn.functional.log_softmax</code>.`,name:"do_marginalize"},{anchor:"transformers.RagConfig.reduce_loss",description:`<strong>reduce_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to reduce the NLL loss using the <code>torch.Tensor.sum</code> operation.`,name:"reduce_loss"},{anchor:"transformers.RagConfig.do_deduplication",description:`<strong>do_deduplication</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to deduplicate the generations from different context documents for a given input. Has to be
set to <code>False</code> if used while training with distributed backend.`,name:"do_deduplication"},{anchor:"transformers.RagConfig.exclude_bos_score",description:`<strong>exclude_bos_score</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to disregard the BOS token when computing the loss.`,name:"exclude_bos_score"},{anchor:"transformers.RagConfig.output_retrieved(bool,",description:`<strong>output_retrieved(<code>bool</code>,</strong> <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code>, <code>retrieved_doc_embeds</code>, <code>retrieved_doc_ids</code>, <code>context_input_ids</code> and
<code>context_attention_mask</code> are returned. See returned tensors for more detail.`,name:"output_retrieved(bool,"},{anchor:"transformers.RagConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.RagConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/configuration_rag.py#L80"}}),_e=new U({props:{name:"from_question_encoder_generator_configs",anchor:"transformers.RagConfig.from_question_encoder_generator_configs",parameters:[{name:"question_encoder_config",val:": PretrainedConfig"},{name:"generator_config",val:": PretrainedConfig"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/configuration_rag.py#L172",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>An instance of a configuration object</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig"
>EncoderDecoderConfig</a></p>
`}}),fe=new de({props:{title:"RagTokenizer",local:"transformers.RagTokenizer",headingTag:"h2"}}),be=new U({props:{name:"class transformers.RagTokenizer",anchor:"transformers.RagTokenizer",parameters:[{name:"question_encoder",val:""},{name:"generator",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/tokenization_rag.py#L29"}}),ve=new de({props:{title:"Rag specific outputs",local:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput",headingTag:"h2"}}),Te=new U({props:{name:"class transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput",anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"doc_scores",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"retrieved_doc_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"retrieved_doc_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"question_encoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"question_enc_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"question_enc_attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_enc_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"generator_enc_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_enc_attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_dec_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_dec_attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_cross_attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) &#x2014;
Language modeling loss.`,name:"loss"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.`,name:"logits"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.doc_scores",description:`<strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) &#x2014;
Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>.`,name:"doc_scores"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.past_key_values",description:`<strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see <code>past_key_values</code> input) to speed up sequential decoding.`,name:"past_key_values"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.retrieved_doc_embeds",description:`<strong>retrieved_doc_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs, hidden_size)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Embedded documents retrieved by the retriever. Is used with <code>question_encoder_last_hidden_state</code> to compute
the <code>doc_scores</code>.`,name:"retrieved_doc_embeds"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.retrieved_doc_ids",description:`<strong>retrieved_doc_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.n_docs)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
The indexes of the embedded documents retrieved by the retriever.`,name:"retrieved_doc_ids"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.context_input_ids",description:`<strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.`,name:"context_input_ids"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.context_attention_mask",description:`<strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever.`,name:"context_attention_mask"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.question_encoder_last_hidden_state",description:`<strong>question_encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.`,name:"question_encoder_last_hidden_state"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.question_enc_hidden_states",description:`<strong>question_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.`,name:"question_enc_hidden_states"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.question_enc_attentions",description:`<strong>question_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.`,name:"question_enc_attentions"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.generator_enc_last_hidden_state",description:`<strong>generator_enc_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the generator encoder of the model.`,name:"generator_enc_last_hidden_state"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.generator_enc_hidden_states",description:`<strong>generator_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.`,name:"generator_enc_hidden_states"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.generator_enc_attentions",description:`<strong>generator_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.`,name:"generator_enc_attentions"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.generator_dec_hidden_states",description:`<strong>generator_dec_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.`,name:"generator_dec_hidden_states"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.generator_dec_attentions",description:`<strong>generator_dec_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.`,name:"generator_dec_attentions"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput.generator_cross_attentions",description:`<strong>generator_cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.`,name:"generator_cross_attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L43"}}),we=new U({props:{name:"class transformers.models.rag.modeling_rag.RetrievAugLMOutput",anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput",parameters:[{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"doc_scores",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"retrieved_doc_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"retrieved_doc_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"question_encoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"question_enc_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"question_enc_attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_enc_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"generator_enc_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_enc_attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_dec_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_dec_attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"},{name:"generator_cross_attentions",val:": typing.Optional[tuple[torch.FloatTensor, ...]] = None"}],parametersDescription:[{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.`,name:"logits"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.doc_scores",description:`<strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) &#x2014;
Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>.`,name:"doc_scores"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.past_key_values",description:`<strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see <code>past_key_values</code> input) to speed up sequential decoding.`,name:"past_key_values"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.retrieved_doc_embeds",description:`<strong>retrieved_doc_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs, hidden_size)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Embedded documents retrieved by the retriever. Is used with <code>question_encoder_last_hidden_state</code> to compute
the <code>doc_scores</code>.`,name:"retrieved_doc_embeds"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.retrieved_doc_ids",description:`<strong>retrieved_doc_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.n_docs)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
The indexes of the embedded documents retrieved by the retriever.`,name:"retrieved_doc_ids"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.context_input_ids",description:`<strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.`,name:"context_input_ids"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.context_attention_mask",description:`<strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever.`,name:"context_attention_mask"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.question_encoder_last_hidden_state",description:`<strong>question_encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.`,name:"question_encoder_last_hidden_state"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.question_enc_hidden_states",description:`<strong>question_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.`,name:"question_enc_hidden_states"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.question_enc_attentions",description:`<strong>question_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.`,name:"question_enc_attentions"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.generator_enc_last_hidden_state",description:`<strong>generator_enc_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the generator encoder of the model.`,name:"generator_enc_last_hidden_state"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.generator_enc_hidden_states",description:`<strong>generator_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.`,name:"generator_enc_hidden_states"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.generator_enc_attentions",description:`<strong>generator_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.`,name:"generator_enc_attentions"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.generator_dec_hidden_states",description:`<strong>generator_dec_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.`,name:"generator_dec_hidden_states"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.generator_dec_attentions",description:`<strong>generator_dec_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.`,name:"generator_dec_attentions"},{anchor:"transformers.models.rag.modeling_rag.RetrievAugLMOutput.generator_cross_attentions",description:`<strong>generator_cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.`,name:"generator_cross_attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L136"}}),ke=new de({props:{title:"RagRetriever",local:"transformers.RagRetriever",headingTag:"h2"}}),Re=new U({props:{name:"class transformers.RagRetriever",anchor:"transformers.RagRetriever",parameters:[{name:"config",val:""},{name:"question_encoder_tokenizer",val:""},{name:"generator_tokenizer",val:""},{name:"index",val:" = None"},{name:"init_retrieval",val:" = True"}],parametersDescription:[{anchor:"transformers.RagRetriever.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig">RagConfig</a>) &#x2014;
The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
<code>Index</code> to build. You can load your own custom dataset with <code>config.index_name=&quot;custom&quot;</code> or use a canonical
one (default) from the datasets library with <code>config.index_name=&quot;wiki_dpr&quot;</code> for example.`,name:"config"},{anchor:"transformers.RagRetriever.question_encoder_tokenizer",description:`<strong>question_encoder_tokenizer</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>) &#x2014;
The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
generator_tokenizer.`,name:"question_encoder_tokenizer"},{anchor:"transformers.RagRetriever.generator_tokenizer",description:`<strong>generator_tokenizer</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>) &#x2014;
The tokenizer used for the generator part of the RagModel.`,name:"generator_tokenizer"},{anchor:"transformers.RagRetriever.index",description:`<strong>index</strong> (<code>Index</code>, optional, defaults to the one defined by the configuration) &#x2014;
If specified, use this index instead of the one built using the configuration`,name:"index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/retrieval_rag.py#L349"}}),Y=new Yt({props:{anchor:"transformers.RagRetriever.example",$$slots:{default:[_n]},$$scope:{ctx:w}}}),xe=new U({props:{name:"init_retrieval",anchor:"transformers.RagRetriever.init_retrieval",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/retrieval_rag.py#L484"}}),Je=new U({props:{name:"postprocess_docs",anchor:"transformers.RagRetriever.postprocess_docs",parameters:[{name:"docs",val:""},{name:"input_strings",val:""},{name:"prefix",val:""},{name:"n_docs",val:""},{name:"return_tensors",val:" = None"}],parametersDescription:[{anchor:"transformers.RagRetriever.postprocess_docs.docs",description:`<strong>docs</strong>  (<code>dict</code>) &#x2014;
Retrieved documents.`,name:"docs"},{anchor:"transformers.RagRetriever.postprocess_docs.input_strings",description:`<strong>input_strings</strong> (<code>str</code>) &#x2014;
Input strings decoded by <code>preprocess_query</code>.`,name:"input_strings"},{anchor:"transformers.RagRetriever.postprocess_docs.prefix",description:`<strong>prefix</strong> (<code>str</code>) &#x2014;
Prefix added at the beginning of each input, typically used with T5-based models.`,name:"prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/retrieval_rag.py#L492",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>a tuple consisting of two elements: contextualized <code>input_ids</code> and a compatible
<code>attention_mask</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>tuple(tensors)</code></p>
`}}),qe=new U({props:{name:"retrieve",anchor:"transformers.RagRetriever.retrieve",parameters:[{name:"question_hidden_states",val:": ndarray"},{name:"n_docs",val:": int"}],parametersDescription:[{anchor:"transformers.RagRetriever.retrieve.question_hidden_states",description:`<strong>question_hidden_states</strong> (<code>np.ndarray</code> of shape <code>(batch_size, vector_size)</code>) &#x2014;
A batch of query vectors to retrieve with.`,name:"question_hidden_states"},{anchor:"transformers.RagRetriever.retrieve.n_docs",description:`<strong>n_docs</strong> (<code>int</code>) &#x2014;
The number of docs retrieved per query.`,name:"n_docs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/retrieval_rag.py#L564",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A tuple with the following objects:</p>
<ul>
<li><strong>retrieved_doc_embeds</strong> (<code>np.ndarray</code> of shape <code>(batch_size, n_docs, dim)</code>) — The retrieval embeddings
of the retrieved docs per query.</li>
<li><strong>doc_ids</strong> (<code>np.ndarray</code> of shape <code>(batch_size, n_docs)</code>) — The ids of the documents in the index</li>
<li><strong>doc_dicts</strong> (<code>list[dict]</code>): The <code>retrieved_doc_embeds</code> examples per query.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>tuple[np.ndarray, np.ndarray, list[dict]]</code></p>
`}}),Ue=new de({props:{title:"RagModel",local:"transformers.RagModel",headingTag:"h2"}}),Ze=new U({props:{name:"class transformers.RagModel",anchor:"transformers.RagModel",parameters:[{name:"config",val:": typing.Optional[transformers.configuration_utils.PretrainedConfig] = None"},{name:"question_encoder",val:": typing.Optional[transformers.modeling_utils.PreTrainedModel] = None"},{name:"generator",val:": typing.Optional[transformers.modeling_utils.PreTrainedModel] = None"},{name:"retriever",val:": typing.Optional[transformers.models.rag.retrieval_rag.RagRetriever] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RagModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>, <em>optional</em>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.RagModel.question_encoder",description:`<strong>question_encoder</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The model responsible for encoding the question into hidden states for retrieval.`,name:"question_encoder"},{anchor:"transformers.RagModel.generator",description:`<strong>generator</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The model responsible for generating text based on retrieved documents.`,name:"generator"},{anchor:"transformers.RagModel.retriever",description:`<strong>retriever</strong> (<code>RagRetriever</code>, <em>optional</em>) &#x2014;
The component responsible for retrieving documents from a knowledge base given the encoded question.`,name:"retriever"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L383"}}),ze=new U({props:{name:"forward",anchor:"transformers.RagModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"doc_scores",val:": typing.Optional[torch.FloatTensor] = None"},{name:"context_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_retrieved",val:": typing.Optional[bool] = None"},{name:"n_docs",val:": typing.Optional[int] = None"}],parametersDescription:[{anchor:"transformers.RagModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. <a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig">RagConfig</a>, used to initialize the model, specifies
which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
obtain the indices.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RagModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RagModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>generator_enc_last_hidden_state</code>, <em>optional</em>: <code>generator_enc_hidden_states</code>,
<em>optional</em>: <code>generator_enc_attentions</code>). <code>generator_enc_last_hidden_state</code> of shape <code>(batch_size, n_docs * sequence_length, hidden_size)</code> is a sequence of hidden-states at the output of the last layer of the
generator&#x2019;s encoder.</p>
<p>Used by the (<a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagModel">RagModel</a>) model during decoding.`,name:"encoder_outputs"},{anchor:"transformers.RagModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Provide for generation tasks. <code>None</code> by default, construct as per instructions for the generator model
you&#x2019;re using with your RAG instance.`,name:"decoder_input_ids"},{anchor:"transformers.RagModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.RagModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.RagModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RagModel.forward.doc_scores",description:`<strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) &#x2014;
Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>. If the model has is not initialized with a <code>retriever</code> <code>doc_scores</code>
has to be provided to the forward pass. <code>doc_scores</code> can be computed via
<code>question_encoder_last_hidden_state</code> and <code>retrieved_doc_embeds</code>, see examples for more information.`,name:"doc_scores"},{anchor:"transformers.RagModel.forward.context_input_ids",description:`<strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Input IDs post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever. If the model was not initialized with a <code>retriever</code> \`<code>context_input_ids</code> has to be provided to
the forward pass. <code>context_input_ids</code> are returned by <code>__call__()</code>.`,name:"context_input_ids"},{anchor:"transformers.RagModel.forward.context_attention_mask",description:`<strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>,<em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever. If the model has is not initialized with a <code>retriever</code> <code>context_attention_mask</code> has to be
provided to the forward pass. <code>context_attention_mask</code> are returned by <code>__call__()</code>.`,name:"context_attention_mask"},{anchor:"transformers.RagModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RagModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RagModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RagModel.forward.output_retrieved",description:`<strong>output_retrieved</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the <code>retrieved_doc_embeds</code>, <code>retrieved_doc_ids</code>, <code>context_input_ids</code> and
<code>context_attention_mask</code>. See returned tensors for more detail.`,name:"output_retrieved"},{anchor:"transformers.RagModel.forward.n_docs",description:`<strong>n_docs</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The number of documents to retrieve.`,name:"n_docs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L434",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMOutput"
>transformers.models.rag.modeling_rag.RetrievAugLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig"
>RagConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.</p>
</li>
<li>
<p><strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) — Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>retrieved_doc_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs, hidden_size)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Embedded documents retrieved by the retriever. Is used with <code>question_encoder_last_hidden_state</code> to compute
the <code>doc_scores</code>.</p>
</li>
<li>
<p><strong>retrieved_doc_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.n_docs)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — The indexes of the embedded documents retrieved by the retriever.</p>
</li>
<li>
<p><strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.</p>
</li>
<li>
<p><strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever.</p>
</li>
<li>
<p><strong>question_encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.</p>
</li>
<li>
<p><strong>question_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>question_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_enc_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.</p>
</li>
<li>
<p><strong>generator_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>generator_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_dec_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>generator_dec_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMOutput"
>transformers.models.rag.modeling_rag.RetrievAugLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Q=new St({props:{$$slots:{default:[fn]},$$scope:{ctx:w}}}),P=new Yt({props:{anchor:"transformers.RagModel.forward.example",$$slots:{default:[yn]},$$scope:{ctx:w}}}),je=new de({props:{title:"RagSequenceForGeneration",local:"transformers.RagSequenceForGeneration",headingTag:"h2"}}),Fe=new U({props:{name:"class transformers.RagSequenceForGeneration",anchor:"transformers.RagSequenceForGeneration",parameters:[{name:"config",val:": typing.Optional[transformers.configuration_utils.PretrainedConfig] = None"},{name:"question_encoder",val:": typing.Optional[transformers.modeling_utils.PreTrainedModel] = None"},{name:"generator",val:": typing.Optional[transformers.modeling_utils.PreTrainedModel] = None"},{name:"retriever",val:": typing.Optional[transformers.models.rag.retrieval_rag.RagRetriever] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RagSequenceForGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>, <em>optional</em>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.RagSequenceForGeneration.question_encoder",description:`<strong>question_encoder</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The model responsible for encoding the question into hidden states for retrieval.`,name:"question_encoder"},{anchor:"transformers.RagSequenceForGeneration.generator",description:`<strong>generator</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The model responsible for generating text based on retrieved documents.`,name:"generator"},{anchor:"transformers.RagSequenceForGeneration.retriever",description:`<strong>retriever</strong> (<code>RagRetriever</code>, <em>optional</em>) &#x2014;
The component responsible for retrieving documents from a knowledge base given the encoded question.`,name:"retriever"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L671"}}),Ie=new U({props:{name:"forward",anchor:"transformers.RagSequenceForGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"context_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"doc_scores",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_retrieved",val:": typing.Optional[bool] = None"},{name:"exclude_bos_score",val:": typing.Optional[bool] = None"},{name:"reduce_loss",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"n_docs",val:": typing.Optional[int] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RagSequenceForGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. <a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig">RagConfig</a>, used to initialize the model, specifies
which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
obtain the indices.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RagSequenceForGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RagSequenceForGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>generator_enc_last_hidden_state</code>, <em>optional</em>: <code>generator_enc_hidden_states</code>,
<em>optional</em>: <code>generator_enc_attentions</code>). <code>generator_enc_last_hidden_state</code> of shape <code>(batch_size, n_docs * sequence_length, hidden_size)</code> is a sequence of hidden-states at the output of the last layer of the
generator&#x2019;s encoder.</p>
<p>Used by the (<a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagModel">RagModel</a>) model during decoding.`,name:"encoder_outputs"},{anchor:"transformers.RagSequenceForGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Provide for generation tasks. <code>None</code> by default, construct as per instructions for the generator model
you&#x2019;re using with your RAG instance.`,name:"decoder_input_ids"},{anchor:"transformers.RagSequenceForGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.RagSequenceForGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.RagSequenceForGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RagSequenceForGeneration.forward.context_input_ids",description:`<strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Input IDs post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever. If the model was not initialized with a <code>retriever</code> \`<code>context_input_ids</code> has to be provided to
the forward pass. <code>context_input_ids</code> are returned by <code>__call__()</code>.`,name:"context_input_ids"},{anchor:"transformers.RagSequenceForGeneration.forward.context_attention_mask",description:`<strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>,<em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever. If the model has is not initialized with a <code>retriever</code> <code>context_attention_mask</code> has to be
provided to the forward pass. <code>context_attention_mask</code> are returned by <code>__call__()</code>.`,name:"context_attention_mask"},{anchor:"transformers.RagSequenceForGeneration.forward.doc_scores",description:`<strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) &#x2014;
Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>. If the model has is not initialized with a <code>retriever</code> <code>doc_scores</code>
has to be provided to the forward pass. <code>doc_scores</code> can be computed via
<code>question_encoder_last_hidden_state</code> and <code>retrieved_doc_embeds</code>, see examples for more information.`,name:"doc_scores"},{anchor:"transformers.RagSequenceForGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RagSequenceForGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RagSequenceForGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RagSequenceForGeneration.forward.output_retrieved",description:`<strong>output_retrieved</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the <code>retrieved_doc_embeds</code>, <code>retrieved_doc_ids</code>, <code>context_input_ids</code> and
<code>context_attention_mask</code>. See returned tensors for more detail.`,name:"output_retrieved"},{anchor:"transformers.RagSequenceForGeneration.forward.exclude_bos_score",description:`<strong>exclude_bos_score</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Only relevant if <code>labels</code> is passed. If <code>True</code>, the score of the BOS token is disregarded when computing
the loss.`,name:"exclude_bos_score"},{anchor:"transformers.RagSequenceForGeneration.forward.reduce_loss",description:`<strong>reduce_loss</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Only relevant if <code>labels</code> is passed. If <code>True</code>, the NLL loss is reduced using the <code>torch.Tensor.sum</code>
operation.`,name:"reduce_loss"},{anchor:"transformers.RagSequenceForGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.RagSequenceForGeneration.forward.n_docs",description:`<strong>n_docs</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The number of documents to retrieve.`,name:"n_docs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L708",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput"
>transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig"
>RagConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.</p>
</li>
<li>
<p><strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) — Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>retrieved_doc_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs, hidden_size)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Embedded documents retrieved by the retriever. Is used with <code>question_encoder_last_hidden_state</code> to compute
the <code>doc_scores</code>.</p>
</li>
<li>
<p><strong>retrieved_doc_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.n_docs)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — The indexes of the embedded documents retrieved by the retriever.</p>
</li>
<li>
<p><strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.</p>
</li>
<li>
<p><strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever.</p>
</li>
<li>
<p><strong>question_encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.</p>
</li>
<li>
<p><strong>question_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>question_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_enc_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.</p>
</li>
<li>
<p><strong>generator_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>generator_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_dec_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>generator_dec_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput"
>transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),D=new St({props:{$$slots:{default:[bn]},$$scope:{ctx:w}}}),K=new Yt({props:{anchor:"transformers.RagSequenceForGeneration.forward.example",$$slots:{default:[vn]},$$scope:{ctx:w}}}),Ge=new U({props:{name:"generate",anchor:"transformers.RagSequenceForGeneration.generate",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"doc_scores",val:": typing.Optional[torch.FloatTensor] = None"},{name:"do_deduplication",val:": typing.Optional[bool] = None"},{name:"num_return_sequences",val:": typing.Optional[int] = None"},{name:"num_beams",val:": typing.Optional[int] = None"},{name:"n_docs",val:": typing.Optional[int] = None"},{name:"**model_kwargs",val:""}],parametersDescription:[{anchor:"transformers.RagSequenceForGeneration.generate.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
The sequence used as a prompt for the generation. If <code>input_ids</code> is not passed, then
<code>context_input_ids</code> has to be provided.`,name:"input_ids"},{anchor:"transformers.RagSequenceForGeneration.generate.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RagSequenceForGeneration.generate.context_input_ids",description:`<strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
retriever.`,name:"context_input_ids"},{anchor:"transformers.RagSequenceForGeneration.generate.context_attention_mask",description:`<strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever.</p>
<p>If the model is not initialized with a <code>retriever</code> or <code>input_ids</code> is not given, <code>context_input_ids</code> and
<code>context_attention_mask</code> have to be provided to the forward pass. They are returned by
<code>__call__()</code>.`,name:"context_attention_mask"},{anchor:"transformers.RagSequenceForGeneration.generate.doc_scores",description:`<strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) &#x2014;
Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>.</p>
<p>If the model is not initialized with a <code>retriever</code> or <code>input_ids</code> is not given, <code>doc_scores</code> has to be
provided to the forward pass. <code>doc_scores</code> are returned by <code>__call__()</code>.`,name:"doc_scores"},{anchor:"transformers.RagSequenceForGeneration.generate.do_deduplication",description:`<strong>do_deduplication</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to deduplicate the generations from different context documents for a given input. Has
to be set to <code>False</code> if used while training with distributed backend.`,name:"do_deduplication"},{anchor:"transformers.RagSequenceForGeneration.generate.num_return_sequences(int,",description:`<strong>num_return_sequences(<code>int</code>,</strong> <em>optional</em>, defaults to 1) &#x2014;
The number of independently computed returned sequences for each element in the batch. Note that this
is not the value we pass to the <code>generator</code>&#x2019;s <code>[generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate)</code> function,
where we set <code>num_return_sequences</code> to <code>num_beams</code>.`,name:"num_return_sequences(int,"},{anchor:"transformers.RagSequenceForGeneration.generate.num_beams",description:`<strong>num_beams</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Number of beams for beam search. 1 means no beam search.`,name:"num_beams"},{anchor:"transformers.RagSequenceForGeneration.generate.n_docs",description:`<strong>n_docs</strong> (<code>int</code>, <em>optional</em>, defaults to <code>config.n_docs</code>) &#x2014;
Number of documents to retrieve and/or number of documents for which to generate an answer.`,name:"n_docs"},{anchor:"transformers.RagSequenceForGeneration.generate.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Additional kwargs will be passed to <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate">generate()</a>.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L881",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The generated
sequences. The second dimension (sequence length) is either equal to <code>max_length</code> or shorter if all batches
finished early due to the <code>eos_token_id</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>torch.LongTensor</code> of shape <code>(batch_size * num_return_sequences, sequence_length)</code></p>
`}}),$e=new de({props:{title:"RagTokenForGeneration",local:"transformers.RagTokenForGeneration",headingTag:"h2"}}),Ce=new U({props:{name:"class transformers.RagTokenForGeneration",anchor:"transformers.RagTokenForGeneration",parameters:[{name:"config",val:": typing.Optional[transformers.configuration_utils.PretrainedConfig] = None"},{name:"question_encoder",val:": typing.Optional[transformers.modeling_utils.PreTrainedModel] = None"},{name:"generator",val:": typing.Optional[transformers.modeling_utils.PreTrainedModel] = None"},{name:"retriever",val:": typing.Optional[transformers.models.rag.retrieval_rag.RagRetriever] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RagTokenForGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>, <em>optional</em>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.RagTokenForGeneration.question_encoder",description:`<strong>question_encoder</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The model responsible for encoding the question into hidden states for retrieval.`,name:"question_encoder"},{anchor:"transformers.RagTokenForGeneration.generator",description:`<strong>generator</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The model responsible for generating text based on retrieved documents.`,name:"generator"},{anchor:"transformers.RagTokenForGeneration.retriever",description:`<strong>retriever</strong> (<code>RagRetriever</code>, <em>optional</em>) &#x2014;
The component responsible for retrieving documents from a knowledge base given the encoded question.`,name:"retriever"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L1108"}}),Ve=new U({props:{name:"forward",anchor:"transformers.RagTokenForGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"context_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"doc_scores",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_retrieved",val:": typing.Optional[bool] = None"},{name:"do_marginalize",val:": typing.Optional[bool] = None"},{name:"reduce_loss",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"n_docs",val:": typing.Optional[int] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RagTokenForGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. <a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig">RagConfig</a>, used to initialize the model, specifies
which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
obtain the indices.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RagTokenForGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RagTokenForGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>generator_enc_last_hidden_state</code>, <em>optional</em>: <code>generator_enc_hidden_states</code>,
<em>optional</em>: <code>generator_enc_attentions</code>). <code>generator_enc_last_hidden_state</code> of shape <code>(batch_size, n_docs * sequence_length, hidden_size)</code> is a sequence of hidden-states at the output of the last layer of the
generator&#x2019;s encoder.</p>
<p>Used by the (<a href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagModel">RagModel</a>) model during decoding.`,name:"encoder_outputs"},{anchor:"transformers.RagTokenForGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Provide for generation tasks. <code>None</code> by default, construct as per instructions for the generator model
you&#x2019;re using with your RAG instance.`,name:"decoder_input_ids"},{anchor:"transformers.RagTokenForGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.RagTokenForGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.RagTokenForGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RagTokenForGeneration.forward.context_input_ids",description:`<strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Input IDs post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever. If the model was not initialized with a <code>retriever</code> \`<code>context_input_ids</code> has to be provided to
the forward pass. <code>context_input_ids</code> are returned by <code>__call__()</code>.`,name:"context_input_ids"},{anchor:"transformers.RagTokenForGeneration.forward.context_attention_mask",description:`<strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>,<em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever. If the model has is not initialized with a <code>retriever</code> <code>context_attention_mask</code> has to be
provided to the forward pass. <code>context_attention_mask</code> are returned by <code>__call__()</code>.`,name:"context_attention_mask"},{anchor:"transformers.RagTokenForGeneration.forward.doc_scores",description:`<strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) &#x2014;
Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>. If the model has is not initialized with a <code>retriever</code> <code>doc_scores</code>
has to be provided to the forward pass. <code>doc_scores</code> can be computed via
<code>question_encoder_last_hidden_state</code> and <code>retrieved_doc_embeds</code>, see examples for more information.`,name:"doc_scores"},{anchor:"transformers.RagTokenForGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RagTokenForGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RagTokenForGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RagTokenForGeneration.forward.output_retrieved",description:`<strong>output_retrieved</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the <code>retrieved_doc_embeds</code>, <code>retrieved_doc_ids</code>, <code>context_input_ids</code> and
<code>context_attention_mask</code>. See returned tensors for more detail.`,name:"output_retrieved"},{anchor:"transformers.RagTokenForGeneration.forward.do_marginalize",description:`<strong>do_marginalize</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If <code>True</code>, the logits are marginalized over all documents by making use of
<code>torch.nn.functional.log_softmax</code>.`,name:"do_marginalize"},{anchor:"transformers.RagTokenForGeneration.forward.reduce_loss",description:`<strong>reduce_loss</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Only relevant if <code>labels</code> is passed. If <code>True</code>, the NLL loss is reduced using the <code>torch.Tensor.sum</code>
operation.`,name:"reduce_loss"},{anchor:"transformers.RagTokenForGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.RagTokenForGeneration.forward.n_docs",description:`<strong>n_docs</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The number of documents to retrieve.`,name:"n_docs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L1220",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput"
>transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.RagConfig"
>RagConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
each vocabulary token.</p>
</li>
<li>
<p><strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) — Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_heads, sequence_length, embed_size_per_head)</code>).</p>
<p>Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
(see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>retrieved_doc_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs, hidden_size)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Embedded documents retrieved by the retriever. Is used with <code>question_encoder_last_hidden_state</code> to compute
the <code>doc_scores</code>.</p>
</li>
<li>
<p><strong>retrieved_doc_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, config.n_docs)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — The indexes of the embedded documents retrieved by the retriever.</p>
</li>
<li>
<p><strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.</p>
</li>
<li>
<p><strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) — Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever.</p>
</li>
<li>
<p><strong>question_encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
model.</p>
</li>
<li>
<p><strong>question_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>question_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_enc_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the generator encoder of the model.</p>
</li>
<li>
<p><strong>generator_enc_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>generator_enc_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_dec_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings and one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>generator_dec_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
average in the self-attention heads.</p>
</li>
<li>
<p><strong>generator_cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/rag#transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput"
>transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),te=new St({props:{$$slots:{default:[Tn]},$$scope:{ctx:w}}}),oe=new Yt({props:{anchor:"transformers.RagTokenForGeneration.forward.example",$$slots:{default:[Mn]},$$scope:{ctx:w}}}),We=new U({props:{name:"generate",anchor:"transformers.RagTokenForGeneration.generate",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"context_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"doc_scores",val:": typing.Optional[torch.FloatTensor] = None"},{name:"n_docs",val:": typing.Optional[int] = None"},{name:"generation_config",val:": typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None"},{name:"prefix_allowed_tokens_fn",val:": typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None"},{name:"logits_processor",val:": typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = []"},{name:"stopping_criteria",val:": typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = []"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RagTokenForGeneration.generate.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
The sequence used as a prompt for the generation. If <code>input_ids</code> is not passed, then
<code>context_input_ids</code> has to be provided.`,name:"input_ids"},{anchor:"transformers.RagTokenForGeneration.generate.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RagTokenForGeneration.generate.context_input_ids",description:`<strong>context_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Input IDs post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever.</p>
<p>If the model has is not initialized with a <code>retriever</code>, <code>context_input_ids</code> has to be provided to the
forward pass. <code>context_input_ids</code> are returned by <code>__call__()</code>.`,name:"context_input_ids"},{anchor:"transformers.RagTokenForGeneration.generate.context_attention_mask",description:`<strong>context_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * config.n_docs, config.max_combined_length)</code>, <em>optional</em>, returned when <em>output_retrieved=True</em>) &#x2014;
Attention mask post-processed from the retrieved documents and the question encoder <code>input_ids</code> by the
retriever.</p>
<p>If the model has is not initialized with a <code>retriever</code>, <code>context_input_ids</code> has to be provided to the
forward pass. <code>context_input_ids</code> are returned by <code>__call__()</code>.`,name:"context_attention_mask"},{anchor:"transformers.RagTokenForGeneration.generate.doc_scores",description:`<strong>doc_scores</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.n_docs)</code>) &#x2014;
Score between each retrieved document embeddings (see <code>retrieved_doc_embeds</code>) and
<code>question_encoder_last_hidden_state</code>.</p>
<p>If the model has is not initialized with a <code>retriever</code>, <code>context_input_ids</code> has to be provided to the
forward pass. <code>context_input_ids</code> are returned by <code>__call__()</code>.`,name:"doc_scores"},{anchor:"transformers.RagTokenForGeneration.generate.n_docs",description:`<strong>n_docs</strong> (<code>int</code>, <em>optional</em>, defaults to <code>config.n_docs</code>) &#x2014;
Number of documents to retrieve and/or number of documents for which to generate an answer.`,name:"n_docs"},{anchor:"transformers.RagTokenForGeneration.generate.generation_config",description:`<strong>generation_config</strong> (<code>~generation.GenerationConfig</code>, <em>optional</em>) &#x2014;
The generation configuration to be used as base parametrization for the generation call. <code>**kwargs</code>
passed to generate matching the attributes of <code>generation_config</code> will override them. If
<code>generation_config</code> is not provided, the default will be used, which has the following loading
priority: 1) from the <code>generation_config.json</code> model file, if it exists; 2) from the model
configuration. Please note that unspecified parameters will inherit <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>&#x2019;s
default values, whose documentation should be checked to parameterize generation.`,name:"generation_config"},{anchor:"transformers.RagTokenForGeneration.generate.prefix_allowed_tokens_fn",description:`<strong>prefix_allowed_tokens_fn</strong> (<code>Callable[[int, torch.Tensor], list[int]]</code>, <em>optional</em>) &#x2014;
If provided, this function constraints the beam search to allowed tokens only at each step. If not
provided no constraint is applied. This function takes 2 arguments <code>inputs_ids</code> and the batch ID
<code>batch_id</code>. It has to return a list with the allowed tokens for the next generation step conditioned on
the previously generated tokens <code>inputs_ids</code> and the batch ID <code>batch_id</code>. This argument is useful for
constrained generation conditioned on the prefix, as described in <a href="https://huggingface.co/papers/2010.00904" rel="nofollow">Autoregressive Entity
Retrieval</a>.`,name:"prefix_allowed_tokens_fn"},{anchor:"transformers.RagTokenForGeneration.generate.logits_processor",description:`<strong>logits_processor</strong> (<code>LogitsProcessorList</code>, <em>optional</em>) &#x2014;
Custom logits processors that complement the default logits processors built from arguments and a
model&#x2019;s config. If a logit processor is passed that is already created with the arguments or a model&#x2019;s
config an error is thrown.`,name:"logits_processor"},{anchor:"transformers.RagTokenForGeneration.generate.stopping_criteria",description:`<strong>stopping_criteria</strong> (<code>StoppingCriteriaList</code>, <em>optional</em>) &#x2014;
Custom stopping criteria that complement the default stopping criteria built from arguments and a
model&#x2019;s config. If a stopping criteria is passed that is already created with the arguments or a
model&#x2019;s config an error is thrown.`,name:"stopping_criteria"},{anchor:"transformers.RagTokenForGeneration.generate.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Ad hoc parametrization of <code>generate_config</code> and/or additional model-specific kwargs that will be
forwarded to the <code>forward</code> function of the model.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/rag/modeling_rag.py#L1393",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The generated
sequences. The second dimension (sequence_length) is either equal to <code>max_length</code> or shorter if all batches
finished early due to the <code>eos_token_id</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>torch.LongTensor</code> of shape <code>(batch_size * num_return_sequences, sequence_length)</code></p>
`}}),Ne=new ln({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/rag.md"}}),{c(){t=p("meta"),b=a(),r=p("p"),l=a(),T=p("p"),T.innerHTML=s,M=a(),h(ie.$$.fragment),ft=a(),B=p("div"),B.innerHTML=zo,yt=a(),ce=p("p"),ce.innerHTML=jo,bt=a(),le=p("p"),le.innerHTML=Fo,vt=a(),h(L.$$.fragment),Tt=a(),pe=p("p"),pe.innerHTML=Io,Mt=a(),h(H.$$.fragment),wt=a(),me=p("p"),me.innerHTML=Go,kt=a(),h(he.$$.fragment),Rt=a(),h(ue.$$.fragment),xt=a(),C=p("div"),h(ge.$$.fragment),At=a(),Ye=p("p"),Ye.innerHTML=$o,Ot=a(),S=p("div"),h(_e.$$.fragment),Et=a(),Ae=p("p"),Ae.innerHTML=Co,Jt=a(),h(fe.$$.fragment),qt=a(),ye=p("div"),h(be.$$.fragment),Ut=a(),h(ve.$$.fragment),Zt=a(),W=p("div"),h(Te.$$.fragment),Qt=a(),Oe=p("p"),Oe.textContent=Vo,zt=a(),Me=p("div"),h(we.$$.fragment),jt=a(),h(ke.$$.fragment),Ft=a(),x=p("div"),h(Re.$$.fragment),Pt=a(),Ee=p("p"),Ee.textContent=Wo,Dt=a(),h(Y.$$.fragment),Kt=a(),A=p("div"),h(xe.$$.fragment),eo=a(),Qe=p("p"),Qe.textContent=No,to=a(),O=p("div"),h(Je.$$.fragment),oo=a(),Pe=p("p"),Pe.innerHTML=Xo,no=a(),E=p("div"),h(qe.$$.fragment),so=a(),De=p("p"),De.innerHTML=Bo,It=a(),h(Ue.$$.fragment),Gt=a(),Z=p("div"),h(Ze.$$.fragment),ro=a(),Ke=p("p"),Ke.textContent=Lo,ao=a(),et=p("p"),et.innerHTML=Ho,io=a(),tt=p("p"),tt.innerHTML=So,co=a(),j=p("div"),h(ze.$$.fragment),lo=a(),ot=p("p"),ot.innerHTML=Yo,po=a(),h(Q.$$.fragment),mo=a(),h(P.$$.fragment),$t=a(),h(je.$$.fragment),Ct=a(),J=p("div"),h(Fe.$$.fragment),ho=a(),nt=p("p"),nt.textContent=Ao,uo=a(),st=p("p"),st.innerHTML=Oo,go=a(),rt=p("p"),rt.innerHTML=Eo,_o=a(),F=p("div"),h(Ie.$$.fragment),fo=a(),at=p("p"),at.innerHTML=Qo,yo=a(),h(D.$$.fragment),bo=a(),h(K.$$.fragment),vo=a(),ee=p("div"),h(Ge.$$.fragment),To=a(),dt=p("p"),dt.innerHTML=Po,Vt=a(),h($e.$$.fragment),Wt=a(),q=p("div"),h(Ce.$$.fragment),Mo=a(),it=p("p"),it.textContent=Do,wo=a(),ct=p("p"),ct.innerHTML=Ko,ko=a(),lt=p("p"),lt.innerHTML=en,Ro=a(),I=p("div"),h(Ve.$$.fragment),xo=a(),pt=p("p"),pt.innerHTML=tn,Jo=a(),h(te.$$.fragment),qo=a(),h(oe.$$.fragment),Uo=a(),ne=p("div"),h(We.$$.fragment),Zo=a(),mt=p("p"),mt.textContent=on,Nt=a(),h(Ne.$$.fragment),Xt=a(),ut=p("p"),this.h()},l(e){const n=dn("svelte-u9bgzb",document.head);t=m(n,"META",{name:!0,content:!0}),n.forEach(o),b=d(e),r=m(e,"P",{}),k(r).forEach(o),l=d(e),T=m(e,"P",{"data-svelte-h":!0}),v(T)!=="svelte-zgp60g"&&(T.innerHTML=s),M=d(e),u(ie.$$.fragment,e),ft=d(e),B=m(e,"DIV",{style:!0,"data-svelte-h":!0}),v(B)!=="svelte-1yr2th3"&&(B.innerHTML=zo),yt=d(e),ce=m(e,"P",{"data-svelte-h":!0}),v(ce)!=="svelte-62qzc8"&&(ce.innerHTML=jo),bt=d(e),le=m(e,"P",{"data-svelte-h":!0}),v(le)!=="svelte-a9dkzi"&&(le.innerHTML=Fo),vt=d(e),u(L.$$.fragment,e),Tt=d(e),pe=m(e,"P",{"data-svelte-h":!0}),v(pe)!=="svelte-zhs1b1"&&(pe.innerHTML=Io),Mt=d(e),u(H.$$.fragment,e),wt=d(e),me=m(e,"P",{"data-svelte-h":!0}),v(me)!=="svelte-rtoh28"&&(me.innerHTML=Go),kt=d(e),u(he.$$.fragment,e),Rt=d(e),u(ue.$$.fragment,e),xt=d(e),C=m(e,"DIV",{class:!0});var N=k(C);u(ge.$$.fragment,N),At=d(N),Ye=m(N,"P",{"data-svelte-h":!0}),v(Ye)!=="svelte-3fokmd"&&(Ye.innerHTML=$o),Ot=d(N),S=m(N,"DIV",{class:!0});var Xe=k(S);u(_e.$$.fragment,Xe),Et=d(Xe),Ae=m(Xe,"P",{"data-svelte-h":!0}),v(Ae)!=="svelte-1qxfvb0"&&(Ae.innerHTML=Co),Xe.forEach(o),N.forEach(o),Jt=d(e),u(fe.$$.fragment,e),qt=d(e),ye=m(e,"DIV",{class:!0});var gt=k(ye);u(be.$$.fragment,gt),gt.forEach(o),Ut=d(e),u(ve.$$.fragment,e),Zt=d(e),W=m(e,"DIV",{class:!0});var Be=k(W);u(Te.$$.fragment,Be),Qt=d(Be),Oe=m(Be,"P",{"data-svelte-h":!0}),v(Oe)!=="svelte-1hlc1gl"&&(Oe.textContent=Vo),Be.forEach(o),zt=d(e),Me=m(e,"DIV",{class:!0});var _t=k(Me);u(we.$$.fragment,_t),_t.forEach(o),jt=d(e),u(ke.$$.fragment,e),Ft=d(e),x=m(e,"DIV",{class:!0});var z=k(x);u(Re.$$.fragment,z),Pt=d(z),Ee=m(z,"P",{"data-svelte-h":!0}),v(Ee)!=="svelte-1t6c1is"&&(Ee.textContent=Wo),Dt=d(z),u(Y.$$.fragment,z),Kt=d(z),A=m(z,"DIV",{class:!0});var Le=k(A);u(xe.$$.fragment,Le),eo=d(Le),Qe=m(Le,"P",{"data-svelte-h":!0}),v(Qe)!=="svelte-1xrn4vk"&&(Qe.textContent=No),Le.forEach(o),to=d(z),O=m(z,"DIV",{class:!0});var He=k(O);u(Je.$$.fragment,He),oo=d(He),Pe=m(He,"P",{"data-svelte-h":!0}),v(Pe)!=="svelte-dey12y"&&(Pe.innerHTML=Xo),He.forEach(o),no=d(z),E=m(z,"DIV",{class:!0});var Se=k(E);u(qe.$$.fragment,Se),so=d(Se),De=m(Se,"P",{"data-svelte-h":!0}),v(De)!=="svelte-mluf2d"&&(De.innerHTML=Bo),Se.forEach(o),z.forEach(o),It=d(e),u(Ue.$$.fragment,e),Gt=d(e),Z=m(e,"DIV",{class:!0});var V=k(Z);u(Ze.$$.fragment,V),ro=d(V),Ke=m(V,"P",{"data-svelte-h":!0}),v(Ke)!=="svelte-120yjtw"&&(Ke.textContent=Lo),ao=d(V),et=m(V,"P",{"data-svelte-h":!0}),v(et)!=="svelte-q52n56"&&(et.innerHTML=Ho),io=d(V),tt=m(V,"P",{"data-svelte-h":!0}),v(tt)!=="svelte-hswkmf"&&(tt.innerHTML=So),co=d(V),j=m(V,"DIV",{class:!0});var se=k(j);u(ze.$$.fragment,se),lo=d(se),ot=m(se,"P",{"data-svelte-h":!0}),v(ot)!=="svelte-4t3gj3"&&(ot.innerHTML=Yo),po=d(se),u(Q.$$.fragment,se),mo=d(se),u(P.$$.fragment,se),se.forEach(o),V.forEach(o),$t=d(e),u(je.$$.fragment,e),Ct=d(e),J=m(e,"DIV",{class:!0});var G=k(J);u(Fe.$$.fragment,G),ho=d(G),nt=m(G,"P",{"data-svelte-h":!0}),v(nt)!=="svelte-146ty18"&&(nt.textContent=Ao),uo=d(G),st=m(G,"P",{"data-svelte-h":!0}),v(st)!=="svelte-q52n56"&&(st.innerHTML=Oo),go=d(G),rt=m(G,"P",{"data-svelte-h":!0}),v(rt)!=="svelte-hswkmf"&&(rt.innerHTML=Eo),_o=d(G),F=m(G,"DIV",{class:!0});var re=k(F);u(Ie.$$.fragment,re),fo=d(re),at=m(re,"P",{"data-svelte-h":!0}),v(at)!=="svelte-18mthth"&&(at.innerHTML=Qo),yo=d(re),u(D.$$.fragment,re),bo=d(re),u(K.$$.fragment,re),re.forEach(o),vo=d(G),ee=m(G,"DIV",{class:!0});var Lt=k(ee);u(Ge.$$.fragment,Lt),To=d(Lt),dt=m(Lt,"P",{"data-svelte-h":!0}),v(dt)!=="svelte-cdmo1s"&&(dt.innerHTML=Po),Lt.forEach(o),G.forEach(o),Vt=d(e),u($e.$$.fragment,e),Wt=d(e),q=m(e,"DIV",{class:!0});var $=k(q);u(Ce.$$.fragment,$),Mo=d($),it=m($,"P",{"data-svelte-h":!0}),v(it)!=="svelte-1j2ak7k"&&(it.textContent=Do),wo=d($),ct=m($,"P",{"data-svelte-h":!0}),v(ct)!=="svelte-q52n56"&&(ct.innerHTML=Ko),ko=d($),lt=m($,"P",{"data-svelte-h":!0}),v(lt)!=="svelte-hswkmf"&&(lt.innerHTML=en),Ro=d($),I=m($,"DIV",{class:!0});var ae=k(I);u(Ve.$$.fragment,ae),xo=d(ae),pt=m(ae,"P",{"data-svelte-h":!0}),v(pt)!=="svelte-vcm3x5"&&(pt.innerHTML=tn),Jo=d(ae),u(te.$$.fragment,ae),qo=d(ae),u(oe.$$.fragment,ae),ae.forEach(o),Uo=d($),ne=m($,"DIV",{class:!0});var Ht=k(ne);u(We.$$.fragment,Ht),Zo=d(Ht),mt=m(Ht,"P",{"data-svelte-h":!0}),v(mt)!=="svelte-1vsuijg"&&(mt.textContent=on),Ht.forEach(o),$.forEach(o),Nt=d(e),u(Ne.$$.fragment,e),Xt=d(e),ut=m(e,"P",{}),k(ut).forEach(o),this.h()},h(){R(t,"name","hf:doc:metadata"),R(t,"content",kn),cn(B,"float","right"),R(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){i(document.head,t),c(e,b,n),c(e,r,n),c(e,l,n),c(e,T,n),c(e,M,n),g(ie,e,n),c(e,ft,n),c(e,B,n),c(e,yt,n),c(e,ce,n),c(e,bt,n),c(e,le,n),c(e,vt,n),g(L,e,n),c(e,Tt,n),c(e,pe,n),c(e,Mt,n),g(H,e,n),c(e,wt,n),c(e,me,n),c(e,kt,n),g(he,e,n),c(e,Rt,n),g(ue,e,n),c(e,xt,n),c(e,C,n),g(ge,C,null),i(C,At),i(C,Ye),i(C,Ot),i(C,S),g(_e,S,null),i(S,Et),i(S,Ae),c(e,Jt,n),g(fe,e,n),c(e,qt,n),c(e,ye,n),g(be,ye,null),c(e,Ut,n),g(ve,e,n),c(e,Zt,n),c(e,W,n),g(Te,W,null),i(W,Qt),i(W,Oe),c(e,zt,n),c(e,Me,n),g(we,Me,null),c(e,jt,n),g(ke,e,n),c(e,Ft,n),c(e,x,n),g(Re,x,null),i(x,Pt),i(x,Ee),i(x,Dt),g(Y,x,null),i(x,Kt),i(x,A),g(xe,A,null),i(A,eo),i(A,Qe),i(x,to),i(x,O),g(Je,O,null),i(O,oo),i(O,Pe),i(x,no),i(x,E),g(qe,E,null),i(E,so),i(E,De),c(e,It,n),g(Ue,e,n),c(e,Gt,n),c(e,Z,n),g(Ze,Z,null),i(Z,ro),i(Z,Ke),i(Z,ao),i(Z,et),i(Z,io),i(Z,tt),i(Z,co),i(Z,j),g(ze,j,null),i(j,lo),i(j,ot),i(j,po),g(Q,j,null),i(j,mo),g(P,j,null),c(e,$t,n),g(je,e,n),c(e,Ct,n),c(e,J,n),g(Fe,J,null),i(J,ho),i(J,nt),i(J,uo),i(J,st),i(J,go),i(J,rt),i(J,_o),i(J,F),g(Ie,F,null),i(F,fo),i(F,at),i(F,yo),g(D,F,null),i(F,bo),g(K,F,null),i(J,vo),i(J,ee),g(Ge,ee,null),i(ee,To),i(ee,dt),c(e,Vt,n),g($e,e,n),c(e,Wt,n),c(e,q,n),g(Ce,q,null),i(q,Mo),i(q,it),i(q,wo),i(q,ct),i(q,ko),i(q,lt),i(q,Ro),i(q,I),g(Ve,I,null),i(I,xo),i(I,pt),i(I,Jo),g(te,I,null),i(I,qo),g(oe,I,null),i(q,Uo),i(q,ne),g(We,ne,null),i(ne,Zo),i(ne,mt),c(e,Nt,n),g(Ne,e,n),c(e,Xt,n),c(e,ut,n),Bt=!0},p(e,[n]){const N={};n&2&&(N.$$scope={dirty:n,ctx:e}),L.$set(N);const Xe={};n&2&&(Xe.$$scope={dirty:n,ctx:e}),H.$set(Xe);const gt={};n&2&&(gt.$$scope={dirty:n,ctx:e}),Y.$set(gt);const Be={};n&2&&(Be.$$scope={dirty:n,ctx:e}),Q.$set(Be);const _t={};n&2&&(_t.$$scope={dirty:n,ctx:e}),P.$set(_t);const z={};n&2&&(z.$$scope={dirty:n,ctx:e}),D.$set(z);const Le={};n&2&&(Le.$$scope={dirty:n,ctx:e}),K.$set(Le);const He={};n&2&&(He.$$scope={dirty:n,ctx:e}),te.$set(He);const Se={};n&2&&(Se.$$scope={dirty:n,ctx:e}),oe.$set(Se)},i(e){Bt||(_(ie.$$.fragment,e),_(L.$$.fragment,e),_(H.$$.fragment,e),_(he.$$.fragment,e),_(ue.$$.fragment,e),_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(fe.$$.fragment,e),_(be.$$.fragment,e),_(ve.$$.fragment,e),_(Te.$$.fragment,e),_(we.$$.fragment,e),_(ke.$$.fragment,e),_(Re.$$.fragment,e),_(Y.$$.fragment,e),_(xe.$$.fragment,e),_(Je.$$.fragment,e),_(qe.$$.fragment,e),_(Ue.$$.fragment,e),_(Ze.$$.fragment,e),_(ze.$$.fragment,e),_(Q.$$.fragment,e),_(P.$$.fragment,e),_(je.$$.fragment,e),_(Fe.$$.fragment,e),_(Ie.$$.fragment,e),_(D.$$.fragment,e),_(K.$$.fragment,e),_(Ge.$$.fragment,e),_($e.$$.fragment,e),_(Ce.$$.fragment,e),_(Ve.$$.fragment,e),_(te.$$.fragment,e),_(oe.$$.fragment,e),_(We.$$.fragment,e),_(Ne.$$.fragment,e),Bt=!0)},o(e){f(ie.$$.fragment,e),f(L.$$.fragment,e),f(H.$$.fragment,e),f(he.$$.fragment,e),f(ue.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(fe.$$.fragment,e),f(be.$$.fragment,e),f(ve.$$.fragment,e),f(Te.$$.fragment,e),f(we.$$.fragment,e),f(ke.$$.fragment,e),f(Re.$$.fragment,e),f(Y.$$.fragment,e),f(xe.$$.fragment,e),f(Je.$$.fragment,e),f(qe.$$.fragment,e),f(Ue.$$.fragment,e),f(Ze.$$.fragment,e),f(ze.$$.fragment,e),f(Q.$$.fragment,e),f(P.$$.fragment,e),f(je.$$.fragment,e),f(Fe.$$.fragment,e),f(Ie.$$.fragment,e),f(D.$$.fragment,e),f(K.$$.fragment,e),f(Ge.$$.fragment,e),f($e.$$.fragment,e),f(Ce.$$.fragment,e),f(Ve.$$.fragment,e),f(te.$$.fragment,e),f(oe.$$.fragment,e),f(We.$$.fragment,e),f(Ne.$$.fragment,e),Bt=!1},d(e){e&&(o(b),o(r),o(l),o(T),o(M),o(ft),o(B),o(yt),o(ce),o(bt),o(le),o(vt),o(Tt),o(pe),o(Mt),o(wt),o(me),o(kt),o(Rt),o(xt),o(C),o(Jt),o(qt),o(ye),o(Ut),o(Zt),o(W),o(zt),o(Me),o(jt),o(Ft),o(x),o(It),o(Gt),o(Z),o($t),o(Ct),o(J),o(Vt),o(Wt),o(q),o(Nt),o(Xt),o(ut)),o(t),y(ie,e),y(L,e),y(H,e),y(he,e),y(ue,e),y(ge),y(_e),y(fe,e),y(be),y(ve,e),y(Te),y(we),y(ke,e),y(Re),y(Y),y(xe),y(Je),y(qe),y(Ue,e),y(Ze),y(ze),y(Q),y(P),y(je,e),y(Fe),y(Ie),y(D),y(K),y(Ge),y($e,e),y(Ce),y(Ve),y(te),y(oe),y(We),y(Ne,e)}}}const kn='{"title":"RAG","local":"rag","sections":[{"title":"RagConfig","local":"transformers.RagConfig","sections":[],"depth":2},{"title":"RagTokenizer","local":"transformers.RagTokenizer","sections":[],"depth":2},{"title":"Rag specific outputs","local":"transformers.models.rag.modeling_rag.RetrievAugLMMarginOutput","sections":[],"depth":2},{"title":"RagRetriever","local":"transformers.RagRetriever","sections":[],"depth":2},{"title":"RagModel","local":"transformers.RagModel","sections":[],"depth":2},{"title":"RagSequenceForGeneration","local":"transformers.RagSequenceForGeneration","sections":[],"depth":2},{"title":"RagTokenForGeneration","local":"transformers.RagTokenForGeneration","sections":[],"depth":2}],"depth":1}';function Rn(w){return sn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class In extends rn{constructor(t){super(),an(this,t,Rn,wn,nn,{})}}export{In as component};
