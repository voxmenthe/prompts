import{s as uo,o as ho,n as _e}from"../chunks/scheduler.18a86fab.js";import{S as fo,i as go,g as c,s as r,r as h,m as co,A as _o,h as m,f as n,c as i,j as se,x as M,u as f,n as mo,k as ae,l as bo,y as p,a as s,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as De}from"../chunks/Tip.77304350.js";import{D as he}from"../chunks/Docstring.a1ef7999.js";import{C as ge}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as po}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as fe,E as yo}from"../chunks/getInferenceSnippets.06c2775f.js";function Mo(C){let t,u='This model was contributed by <a href="https://huggingface.co/Muennighoff" rel="nofollow">Muennighoff</a>.',a,d,T="Click on the OLMoE models in the right sidebar for more examples of how to apply OLMoE to different language tasks.";return{c(){t=c("p"),t.innerHTML=u,a=r(),d=c("p"),d.textContent=T},l(l){t=m(l,"P",{"data-svelte-h":!0}),M(t)!=="svelte-k9gtj1"&&(t.innerHTML=u),a=i(l),d=m(l,"P",{"data-svelte-h":!0}),M(d)!=="svelte-l89kk1"&&(d.textContent=T)},m(l,$){s(l,t,$),s(l,a,$),s(l,d,$)},p:_e,d(l){l&&(n(t),n(a),n(d))}}}function To(C){let t,u;return t=new ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME9sbW9lTW9kZWwlMkMlMjBPbG1vZUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBPTE1vRSUyMDdCJTIwQTFCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME9sbW9lQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwT0xNb0UlMjA3QiUyMEExQiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwT2xtb2VNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OlmoeModel, OlmoeConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a OLMoE 7B A1B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = OlmoeConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the OLMoE 7B A1B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OlmoeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){h(t.$$.fragment)},l(a){f(t.$$.fragment,a)},m(a,d){g(t,a,d),u=!0},p:_e,i(a){u||(_(t.$$.fragment,a),u=!0)},o(a){b(t.$$.fragment,a),u=!1},d(a){y(t,a)}}}function vo(C){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(a){t=m(a,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(a,d){s(a,t,d)},p:_e,d(a){a&&n(t)}}}function wo(C){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(a){t=m(a,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(a,d){s(a,t,d)},p:_e,d(a){a&&n(t)}}}function ko(C){let t,u="Example:",a,d,T;return d=new ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBPbG1vZUZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBPbG1vZUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJhbGxlbmFpJTJGT0xNb0UtMUItN0ItMDkyNCUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJhbGxlbmFpJTJGT0xNb0UtMUItN0ItMDkyNCUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, OlmoeForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = OlmoeForCausalLM.from_pretrained(<span class="hljs-string">&quot;allenai/OLMoE-1B-7B-0924&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;allenai/OLMoE-1B-7B-0924&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&#x27;Hey, are you conscious? Can you talk to me?\\nI’m not sure if you’re conscious of this, but I’m&#x27;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,a=r(),h(d.$$.fragment)},l(l){t=m(l,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=u),a=i(l),f(d.$$.fragment,l)},m(l,$){s(l,t,$),s(l,a,$),g(d,l,$),T=!0},p:_e,i(l){T||(_(d.$$.fragment,l),T=!0)},o(l){b(d.$$.fragment,l),T=!1},d(l){l&&(n(t),n(a)),y(d,l)}}}function Co(C){let t,u,a,d,T,l="<em>This model was released on 2024-09-03 and added to Hugging Face Transformers on 2024-09-03.</em>",$,I,Se='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',be,q,ye,W,Ye='<a href="https://huggingface.co/papers/2409.02060" rel="nofollow">OLMoE</a> is a sparse Mixture-of-Experts (MoE) language model with 7B parameters but only 1B parameters are used per input token. It has similar inference costs as dense models but trains ~3x faster. OLMoE uses fine-grained routing with 64 small experts in each layer and uses a dropless token-based routing algorithm.',Me,N,Ke='You can find all the original OLMoE checkpoints under the <a href="https://huggingface.co/collections/allenai/olmoe-november-2024-66cf678c047657a30c8cd3da" rel="nofollow">OLMoE</a> collection.',Te,F,ve,G,eo='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',we,R,ke,H,Ce,X,$e,Q,oo=`Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.
The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to 4-bits.`,xe,V,Je,A,je,w,P,Le,re,to=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeModel">OlmoeModel</a>. It is used to instantiate an OLMoE
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the <a href="https://huggingface.co/allenai/OLMoE-1B-7B-0924" rel="nofollow">allenai/OLMoE-1B-7B-0924</a>.`,Be,ie,no=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,qe,E,Ue,D,Oe,v,S,We,le,so="The bare Olmoe Model outputting raw hidden-states without any specific head on top.",Ne,de,ao=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ge,ce,ro=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Re,U,Y,He,me,io='The <a href="/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeModel">OlmoeModel</a> forward method, overrides the <code>__call__</code> special method.',Xe,Z,ze,K,Ie,O,ee,Qe,x,oe,Ve,pe,lo='The <a href="/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeForCausalLM">OlmoeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ae,L,Pe,B,Fe,te,Ee,ue,Ze;return q=new fe({props:{title:"OLMoE",local:"olmoe",headingTag:"h1"}}),F=new De({props:{warning:!1,$$slots:{default:[Mo]},$$scope:{ctx:C}}}),R=new ge({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyYWxsZW5haSUyRk9MTW9FLTFCLTdCLTAxMjUlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTJDJTBBKSUwQSUwQXJlc3VsdCUyMCUzRCUyMHBpcGUoJTIyRGlvbnlzdXMlMjBpcyUyMHRoZSUyMGdvZCUyMG9mJTIyKSUwQXByaW50KHJlc3VsdCk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;allenai/OLMoE-1B-7B-0125&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>,
)

result = pipe(<span class="hljs-string">&quot;Dionysus is the god of&quot;</span>)
<span class="hljs-built_in">print</span>(result)`,wrap:!1}}),H=new ge({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwaW5mZXJfZGV2aWNlJTBBJTBBZGV2aWNlJTIwJTNEJTIwaW5mZXJfZGV2aWNlKCklMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJhbGxlbmFpJTJGT0xNb0UtMUItN0ItMDkyNCUyMiUyQyUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTJDJTIwZHR5cGUlM0QlMjJhdXRvJTIyJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIpLnRvKGRldmljZSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJhbGxlbmFpJTJGT0xNb0UtMUItN0ItMDkyNCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyQml0Y29pbiUyMGlzJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFpbnB1dHMlMjAlM0QlMjAlN0JrJTNBJTIwdi50byhkZXZpY2UpJTIwZm9yJTIwayUyQyUyMHYlMjBpbiUyMGlucHV0cy5pdGVtcygpJTdEJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbGVuZ3RoJTNENjQpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, infer_device

device = infer_device()

model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;allenai/OLMoE-1B-7B-0924&quot;</span>, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, dtype=<span class="hljs-string">&quot;auto&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>).to(device)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;allenai/OLMoE-1B-7B-0924&quot;</span>)

inputs = tokenizer(<span class="hljs-string">&quot;Bitcoin is&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
inputs = {k: v.to(device) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> inputs.items()}
output = model.generate(**inputs, max_length=<span class="hljs-number">64</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>]))`,wrap:!1}}),X=new fe({props:{title:"Quantization",local:"quantization",headingTag:"h2"}}),V=new ge({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwaW5mZXJfZGV2aWNlJTBBJTBBZGV2aWNlJTIwJTNEJTIwaW5mZXJfZGV2aWNlKCklMEElMEFxdWFudGl6YXRpb25fY29uZmlnJTIwJTNEJTIwQml0c0FuZEJ5dGVzQ29uZmlnKCUwQSUyMCUyMCUyMGxvYWRfaW5fNGJpdCUzRFRydWUlMkMlMEElMjAlMjAlMjBibmJfNGJpdF9jb21wdXRlX2R0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUwQSklMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJhbGxlbmFpJTJGT0xNb0UtMUItN0ItMDkyNCUyMiUyQyUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTJDJTIwZHR5cGUlM0QlMjJhdXRvJTIyJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEcXVhbnRpemF0aW9uX2NvbmZpZykudG8oZGV2aWNlKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmFsbGVuYWklMkZPTE1vRS0xQi03Qi0wOTI0JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJCaXRjb2luJTIwaXMlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWlucHV0cyUyMCUzRCUyMCU3QmslM0ElMjB2LnRvKGRldmljZSklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwaW5wdXRzLml0ZW1zKCklN0QlMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9sZW5ndGglM0Q2NCklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, infer_device

device = infer_device()

quantization_config = BitsAndBytesConfig(
   load_in_4bit=<span class="hljs-literal">True</span>,
   bnb_4bit_compute_dtype=torch.float16,
   bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,
   bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>
)

model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;allenai/OLMoE-1B-7B-0924&quot;</span>, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, dtype=<span class="hljs-string">&quot;auto&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>, quantization_config=quantization_config).to(device)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;allenai/OLMoE-1B-7B-0924&quot;</span>)

inputs = tokenizer(<span class="hljs-string">&quot;Bitcoin is&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
inputs = {k: v.to(device) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> inputs.items()}
output = model.generate(**inputs, max_length=<span class="hljs-number">64</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>]))`,wrap:!1}}),A=new fe({props:{title:"OlmoeConfig",local:"transformers.OlmoeConfig",headingTag:"h2"}}),P=new he({props:{name:"class transformers.OlmoeConfig",anchor:"transformers.OlmoeConfig",parameters:[{name:"vocab_size",val:" = 50304"},{name:"hidden_size",val:" = 2048"},{name:"intermediate_size",val:" = 2048"},{name:"num_hidden_layers",val:" = 16"},{name:"num_attention_heads",val:" = 16"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 4096"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = None"},{name:"eos_token_id",val:" = 50279"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"clip_qkv",val:" = None"},{name:"num_experts_per_tok",val:" = 8"},{name:"num_experts",val:" = 64"},{name:"output_router_logits",val:" = False"},{name:"router_aux_loss_coef",val:" = 0.01"},{name:"norm_topk_prob",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OlmoeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50304) &#x2014;
Vocabulary size of the OLMoE model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeModel">OlmoeModel</a>`,name:"vocab_size"},{anchor:"transformers.OlmoeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.OlmoeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.OlmoeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.OlmoeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.OlmoeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.OlmoeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.OlmoeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.OlmoeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.OlmoeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.OlmoeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.OlmoeConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.OlmoeConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.OlmoeConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 50279) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.OlmoeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.OlmoeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.OlmoeConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
<code>{&quot;type&quot;: strategy name, &quot;factor&quot;: scaling factor}</code>. When using this flag, don&#x2019;t update
<code>max_position_embeddings</code> to the expected new maximum. See the following thread for more information on how
these scaling strategies behave:
<a href="https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/" rel="nofollow">https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/</a>. This is an
experimental feature, subject to breaking API changes in future versions.`,name:"rope_scaling"},{anchor:"transformers.OlmoeConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.OlmoeConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.OlmoeConfig.clip_qkv",description:`<strong>clip_qkv</strong> (<code>float</code>, <em>optional</em>) &#x2014;
If not <code>None</code>, elements of query, key and value attention states are clipped so that their
absolute value does not exceed this value.`,name:"clip_qkv"},{anchor:"transformers.OlmoeConfig.num_experts_per_tok",description:`<strong>num_experts_per_tok</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of selected experts.`,name:"num_experts_per_tok"},{anchor:"transformers.OlmoeConfig.num_experts",description:`<strong>num_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of routed experts.`,name:"num_experts"},{anchor:"transformers.OlmoeConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss, including load balancing loss and router z-loss.`,name:"output_router_logits"},{anchor:"transformers.OlmoeConfig.router_aux_loss_coef",description:`<strong>router_aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.01) &#x2014;
The aux loss factor for the total loss.`,name:"router_aux_loss_coef"},{anchor:"transformers.OlmoeConfig.norm_topk_prob",description:`<strong>norm_topk_prob</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to normalize the topk probabilities.`,name:"norm_topk_prob"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmoe/configuration_olmoe.py#L18"}}),E=new po({props:{anchor:"transformers.OlmoeConfig.example",$$slots:{default:[To]},$$scope:{ctx:C}}}),D=new fe({props:{title:"OlmoeModel",local:"transformers.OlmoeModel",headingTag:"h2"}}),S=new he({props:{name:"class transformers.OlmoeModel",anchor:"transformers.OlmoeModel",parameters:[{name:"config",val:": OlmoeConfig"}],parametersDescription:[{anchor:"transformers.OlmoeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeConfig">OlmoeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmoe/modeling_olmoe.py#L737"}}),Y=new he({props:{name:"forward",anchor:"transformers.OlmoeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.OlmoeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.OlmoeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.OlmoeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.OlmoeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.OlmoeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.OlmoeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.OlmoeModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OlmoeModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OlmoeModel.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.OlmoeModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.OlmoeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmoe/modeling_olmoe.py#L754",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeConfig"
>OlmoeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
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
<p><strong>router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_probs=True</code> and <code>config.add_router_probs=True</code> is passed or when <code>config.output_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
loss for Mixture of Experts models.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Z=new De({props:{$$slots:{default:[vo]},$$scope:{ctx:C}}}),K=new fe({props:{title:"OlmoeForCausalLM",local:"transformers.OlmoeForCausalLM",headingTag:"h2"}}),ee=new he({props:{name:"class transformers.OlmoeForCausalLM",anchor:"transformers.OlmoeForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmoe/modeling_olmoe.py#L985"}}),oe=new he({props:{name:"forward",anchor:"transformers.OlmoeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OlmoeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.OlmoeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.OlmoeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.OlmoeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.OlmoeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.OlmoeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.OlmoeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.OlmoeForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.OlmoeForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.OlmoeForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.OlmoeForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.OlmoeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.OlmoeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmoe/modeling_olmoe.py#L1000",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/olmoe#transformers.OlmoeConfig"
>OlmoeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>aux_loss</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, returned when <code>labels</code> is provided) — aux_loss for the sparse modules.</p>
</li>
<li>
<p><strong>router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_probs=True</code> and <code>config.add_router_probs=True</code> is passed or when <code>config.output_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
loss for Mixture of Experts models.</p>
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


<p><code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),L=new De({props:{$$slots:{default:[wo]},$$scope:{ctx:C}}}),B=new po({props:{anchor:"transformers.OlmoeForCausalLM.forward.example",$$slots:{default:[ko]},$$scope:{ctx:C}}}),te=new yo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/olmoe.md"}}),{c(){t=c("meta"),u=r(),a=c("p"),d=r(),T=c("p"),T.innerHTML=l,$=r(),I=c("div"),I.innerHTML=Se,be=r(),h(q.$$.fragment),ye=r(),W=c("p"),W.innerHTML=Ye,Me=r(),N=c("p"),N.innerHTML=Ke,Te=r(),h(F.$$.fragment),ve=r(),G=c("p"),G.innerHTML=eo,we=co(`
<hfoptions id="usage">
<hfoption id="Pipeline">

	`),h(R.$$.fragment),ke=co(`
</hfoption>
<hfoption id="AutoModel">

	`),h(H.$$.fragment),Ce=r(),h(X.$$.fragment),$e=r(),Q=c("p"),Q.innerHTML=oo,xe=r(),h(V.$$.fragment),Je=r(),h(A.$$.fragment),je=r(),w=c("div"),h(P.$$.fragment),Le=r(),re=c("p"),re.innerHTML=to,Be=r(),ie=c("p"),ie.innerHTML=no,qe=r(),h(E.$$.fragment),Ue=r(),h(D.$$.fragment),Oe=r(),v=c("div"),h(S.$$.fragment),We=r(),le=c("p"),le.textContent=so,Ne=r(),de=c("p"),de.innerHTML=ao,Ge=r(),ce=c("p"),ce.innerHTML=ro,Re=r(),U=c("div"),h(Y.$$.fragment),He=r(),me=c("p"),me.innerHTML=io,Xe=r(),h(Z.$$.fragment),ze=r(),h(K.$$.fragment),Ie=r(),O=c("div"),h(ee.$$.fragment),Qe=r(),x=c("div"),h(oe.$$.fragment),Ve=r(),pe=c("p"),pe.innerHTML=lo,Ae=r(),h(L.$$.fragment),Pe=r(),h(B.$$.fragment),Fe=r(),h(te.$$.fragment),Ee=r(),ue=c("p"),this.h()},l(e){const o=_o("svelte-u9bgzb",document.head);t=m(o,"META",{name:!0,content:!0}),o.forEach(n),u=i(e),a=m(e,"P",{}),se(a).forEach(n),d=i(e),T=m(e,"P",{"data-svelte-h":!0}),M(T)!=="svelte-rio3mq"&&(T.innerHTML=l),$=i(e),I=m(e,"DIV",{style:!0,"data-svelte-h":!0}),M(I)!=="svelte-1hyaprb"&&(I.innerHTML=Se),be=i(e),f(q.$$.fragment,e),ye=i(e),W=m(e,"P",{"data-svelte-h":!0}),M(W)!=="svelte-1mzh5ii"&&(W.innerHTML=Ye),Me=i(e),N=m(e,"P",{"data-svelte-h":!0}),M(N)!=="svelte-102utxm"&&(N.innerHTML=Ke),Te=i(e),f(F.$$.fragment,e),ve=i(e),G=m(e,"P",{"data-svelte-h":!0}),M(G)!=="svelte-c361bk"&&(G.innerHTML=eo),we=mo(e,`
<hfoptions id="usage">
<hfoption id="Pipeline">

	`),f(R.$$.fragment,e),ke=mo(e,`
</hfoption>
<hfoption id="AutoModel">

	`),f(H.$$.fragment,e),Ce=i(e),f(X.$$.fragment,e),$e=i(e),Q=m(e,"P",{"data-svelte-h":!0}),M(Q)!=="svelte-es10a4"&&(Q.innerHTML=oo),xe=i(e),f(V.$$.fragment,e),Je=i(e),f(A.$$.fragment,e),je=i(e),w=m(e,"DIV",{class:!0});var J=se(w);f(P.$$.fragment,J),Le=i(J),re=m(J,"P",{"data-svelte-h":!0}),M(re)!=="svelte-1cj0f76"&&(re.innerHTML=to),Be=i(J),ie=m(J,"P",{"data-svelte-h":!0}),M(ie)!=="svelte-1ek1ss9"&&(ie.innerHTML=no),qe=i(J),f(E.$$.fragment,J),J.forEach(n),Ue=i(e),f(D.$$.fragment,e),Oe=i(e),v=m(e,"DIV",{class:!0});var k=se(v);f(S.$$.fragment,k),We=i(k),le=m(k,"P",{"data-svelte-h":!0}),M(le)!=="svelte-qhbvy8"&&(le.textContent=so),Ne=i(k),de=m(k,"P",{"data-svelte-h":!0}),M(de)!=="svelte-q52n56"&&(de.innerHTML=ao),Ge=i(k),ce=m(k,"P",{"data-svelte-h":!0}),M(ce)!=="svelte-hswkmf"&&(ce.innerHTML=ro),Re=i(k),U=m(k,"DIV",{class:!0});var z=se(U);f(Y.$$.fragment,z),He=i(z),me=m(z,"P",{"data-svelte-h":!0}),M(me)!=="svelte-1fmq9jn"&&(me.innerHTML=io),Xe=i(z),f(Z.$$.fragment,z),z.forEach(n),k.forEach(n),ze=i(e),f(K.$$.fragment,e),Ie=i(e),O=m(e,"DIV",{class:!0});var ne=se(O);f(ee.$$.fragment,ne),Qe=i(ne),x=m(ne,"DIV",{class:!0});var j=se(x);f(oe.$$.fragment,j),Ve=i(j),pe=m(j,"P",{"data-svelte-h":!0}),M(pe)!=="svelte-f497jr"&&(pe.innerHTML=lo),Ae=i(j),f(L.$$.fragment,j),Pe=i(j),f(B.$$.fragment,j),j.forEach(n),ne.forEach(n),Fe=i(e),f(te.$$.fragment,e),Ee=i(e),ue=m(e,"P",{}),se(ue).forEach(n),this.h()},h(){ae(t,"name","hf:doc:metadata"),ae(t,"content",$o),bo(I,"float","right"),ae(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ae(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ae(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ae(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ae(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){p(document.head,t),s(e,u,o),s(e,a,o),s(e,d,o),s(e,T,o),s(e,$,o),s(e,I,o),s(e,be,o),g(q,e,o),s(e,ye,o),s(e,W,o),s(e,Me,o),s(e,N,o),s(e,Te,o),g(F,e,o),s(e,ve,o),s(e,G,o),s(e,we,o),g(R,e,o),s(e,ke,o),g(H,e,o),s(e,Ce,o),g(X,e,o),s(e,$e,o),s(e,Q,o),s(e,xe,o),g(V,e,o),s(e,Je,o),g(A,e,o),s(e,je,o),s(e,w,o),g(P,w,null),p(w,Le),p(w,re),p(w,Be),p(w,ie),p(w,qe),g(E,w,null),s(e,Ue,o),g(D,e,o),s(e,Oe,o),s(e,v,o),g(S,v,null),p(v,We),p(v,le),p(v,Ne),p(v,de),p(v,Ge),p(v,ce),p(v,Re),p(v,U),g(Y,U,null),p(U,He),p(U,me),p(U,Xe),g(Z,U,null),s(e,ze,o),g(K,e,o),s(e,Ie,o),s(e,O,o),g(ee,O,null),p(O,Qe),p(O,x),g(oe,x,null),p(x,Ve),p(x,pe),p(x,Ae),g(L,x,null),p(x,Pe),g(B,x,null),s(e,Fe,o),g(te,e,o),s(e,Ee,o),s(e,ue,o),Ze=!0},p(e,[o]){const J={};o&2&&(J.$$scope={dirty:o,ctx:e}),F.$set(J);const k={};o&2&&(k.$$scope={dirty:o,ctx:e}),E.$set(k);const z={};o&2&&(z.$$scope={dirty:o,ctx:e}),Z.$set(z);const ne={};o&2&&(ne.$$scope={dirty:o,ctx:e}),L.$set(ne);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),B.$set(j)},i(e){Ze||(_(q.$$.fragment,e),_(F.$$.fragment,e),_(R.$$.fragment,e),_(H.$$.fragment,e),_(X.$$.fragment,e),_(V.$$.fragment,e),_(A.$$.fragment,e),_(P.$$.fragment,e),_(E.$$.fragment,e),_(D.$$.fragment,e),_(S.$$.fragment,e),_(Y.$$.fragment,e),_(Z.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(oe.$$.fragment,e),_(L.$$.fragment,e),_(B.$$.fragment,e),_(te.$$.fragment,e),Ze=!0)},o(e){b(q.$$.fragment,e),b(F.$$.fragment,e),b(R.$$.fragment,e),b(H.$$.fragment,e),b(X.$$.fragment,e),b(V.$$.fragment,e),b(A.$$.fragment,e),b(P.$$.fragment,e),b(E.$$.fragment,e),b(D.$$.fragment,e),b(S.$$.fragment,e),b(Y.$$.fragment,e),b(Z.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(oe.$$.fragment,e),b(L.$$.fragment,e),b(B.$$.fragment,e),b(te.$$.fragment,e),Ze=!1},d(e){e&&(n(u),n(a),n(d),n(T),n($),n(I),n(be),n(ye),n(W),n(Me),n(N),n(Te),n(ve),n(G),n(we),n(ke),n(Ce),n($e),n(Q),n(xe),n(Je),n(je),n(w),n(Ue),n(Oe),n(v),n(ze),n(Ie),n(O),n(Fe),n(Ee),n(ue)),n(t),y(q,e),y(F,e),y(R,e),y(H,e),y(X,e),y(V,e),y(A,e),y(P),y(E),y(D,e),y(S),y(Y),y(Z),y(K,e),y(ee),y(oe),y(L),y(B),y(te,e)}}}const $o='{"title":"OLMoE","local":"olmoe","sections":[{"title":"Quantization","local":"quantization","sections":[],"depth":2},{"title":"OlmoeConfig","local":"transformers.OlmoeConfig","sections":[],"depth":2},{"title":"OlmoeModel","local":"transformers.OlmoeModel","sections":[],"depth":2},{"title":"OlmoeForCausalLM","local":"transformers.OlmoeForCausalLM","sections":[],"depth":2}],"depth":1}';function xo(C){return ho(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Eo extends fo{constructor(t){super(),go(this,t,xo,Co,uo,{})}}export{Eo as component};
