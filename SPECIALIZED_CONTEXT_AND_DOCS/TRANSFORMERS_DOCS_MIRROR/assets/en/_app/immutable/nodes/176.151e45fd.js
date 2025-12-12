import{s as Zo,o as Go,n as Ue}from"../chunks/scheduler.18a86fab.js";import{S as Bo,i as Vo,g as d,s,r as u,A as Wo,h as c,f as t,c as a,j as H,x as y,u as h,k as Z,l as Ro,y as l,a as r,v as f,d as g,t as _,w as M}from"../chunks/index.98837b22.js";import{T as _o}from"../chunks/Tip.77304350.js";import{D as Je}from"../chunks/Docstring.a1ef7999.js";import{C as Ee}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as zo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as G,E as No}from"../chunks/getInferenceSnippets.06c2775f.js";function Xo(E){let n,b;return n=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEVybmllNF81X01vZU1vZGVsJTJDJTIwRXJuaWU0XzVfTW9FQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEVybmllNF81X01vRSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBFcm5pZTRfNV9Nb0VDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBFUk5JRS00LjUtMjFCLUEzQiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwRXJuaWU0XzVfTW9lTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Ernie4_5_MoeModel, Ernie4_5_MoEConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Ernie4_5_MoE style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Ernie4_5_MoEConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the ERNIE-4.5-21B-A3B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Ernie4_5_MoeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){u(n.$$.fragment)},l(i){h(n.$$.fragment,i)},m(i,p){f(n,i,p),b=!0},p:Ue,i(i){b||(g(n.$$.fragment,i),b=!0)},o(i){_(n.$$.fragment,i),b=!1},d(i){M(n,i)}}}function Qo(E){let n,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=b},l(i){n=c(i,"P",{"data-svelte-h":!0}),y(n)!=="svelte-fincs2"&&(n.innerHTML=b)},m(i,p){r(i,n,p)},p:Ue,d(i){i&&t(n)}}}function Ho(E){let n,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=b},l(i){n=c(i,"P",{"data-svelte-h":!0}),y(n)!=="svelte-fincs2"&&(n.innerHTML=b)},m(i,p){r(i,n,p)},p:Ue,d(i){i&&t(n)}}}function Lo(E){let n,b="Example:",i,p,w;return p=new Ee({props:{code:"",highlighted:"",wrap:!1}}),{c(){n=d("p"),n.textContent=b,i=s(),u(p.$$.fragment)},l(m){n=c(m,"P",{"data-svelte-h":!0}),y(n)!=="svelte-11lpom8"&&(n.textContent=b),i=a(m),h(p.$$.fragment,m)},m(m,U){r(m,n,U),r(m,i,U),f(p,m,U),w=!0},p:Ue,i(m){w||(g(p.$$.fragment,m),w=!0)},o(m){_(p.$$.fragment,m),w=!1},d(m){m&&(t(n),t(i)),M(p,m)}}}function qo(E){let n,b=`Most generation-controlling parameters are set in <code>generation_config</code> which, if not passed, will be set to the
model’s default generation configuration. You can override any <code>generation_config</code> by passing the corresponding
parameters to generate(), e.g. <code>.generate(inputs, num_beams=4, do_sample=True)</code>.`,i,p,w=`For an overview of generation strategies and code examples, check out the <a href="../generation_strategies">following
guide</a>.`;return{c(){n=d("p"),n.innerHTML=b,i=s(),p=d("p"),p.innerHTML=w},l(m){n=c(m,"P",{"data-svelte-h":!0}),y(n)!=="svelte-1c5u34l"&&(n.innerHTML=b),i=a(m),p=c(m,"P",{"data-svelte-h":!0}),y(p)!=="svelte-fvlq1g"&&(p.innerHTML=w)},m(m,U){r(m,n,U),r(m,i,U),r(m,p,U)},p:Ue,d(m){m&&(t(n),t(i),t(p))}}}function Ao(E){let n,b,i,p,w,m="<em>This model was released on 2025-06-30 and added to Hugging Face Transformers on 2025-07-21.</em>",U,B,Mo='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',Ce,L,$e,q,je,A,yo=`The Ernie 4.5 Moe model was released in the <a href="https://ernie.baidu.com/blog/posts/ernie4.5/" rel="nofollow">Ernie 4.5 Model Family</a> release by baidu.
This family of models contains multiple different architectures and model sizes. This model in specific targets the base text
model with mixture of experts (moe) - one with 21B total, 3B active parameters and another one with 300B total, 47B active parameters.
It uses the standard <a href="./llama">Llama</a> at its core combined with a specialized MoE based on <a href="./mixtral">Mixtral</a> with additional shared
experts.`,Ie,S,bo='Other models from the family can be found at <a href="./ernie4_5">Ernie 4.5</a>.',Fe,V,To='<img src="https://ernie.baidu.com/blog/posts/ernie4.5/overview.png"/>',ze,P,Ze,Y,Ge,O,Be,D,Ve,K,We,ee,Re,oe,Ne,te,wo=`This model was contributed by <a href="https://huggingface.co/AntonV" rel="nofollow">Anton Vlasjuk</a>.
The original code can be found <a href="https://github.com/PaddlePaddle/ERNIE" rel="nofollow">here</a>.`,Xe,ne,Qe,J,se,Oe,ue,vo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeModel">Ernie4_5_MoeModel</a>. It is used to instantiate a
Ernie 4.5 MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of <a href="https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT" rel="nofollow">baidu/ERNIE-4.5-21B-A3B-PT</a>.`,De,he,ko=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ke,W,He,ae,Le,v,re,eo,fe,Jo="The bare Ernie4 5 Moe Model outputting raw hidden-states without any specific head on top.",oo,ge,xo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,to,_e,Eo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,no,I,ie,so,Me,Uo='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeModel">Ernie4_5_MoeModel</a> forward method, overrides the <code>__call__</code> special method.',ao,R,qe,le,Ae,T,de,ro,ye,Co="The Ernie4 5 Moe Model for causal language modeling.",io,be,$o=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,lo,Te,jo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,co,C,ce,mo,we,Io='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeForCausalLM">Ernie4_5_MoeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',po,N,uo,X,ho,F,me,fo,ve,Fo="Generates sequences of token ids for models with a language modeling head.",go,Q,Se,pe,Pe,xe,Ye;return L=new G({props:{title:"Ernie 4.5 Moe",local:"ernie-45-moe",headingTag:"h1"}}),q=new G({props:{title:"Overview",local:"overview",headingTag:"h2"}}),P=new G({props:{title:"Usage Tips",local:"usage-tips",headingTag:"h2"}}),Y=new G({props:{title:"Generate text",local:"generate-text",headingTag:"h3"}}),O=new Ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWxfbmFtZSUyMCUzRCUyMCUyMmJhaWR1JTJGRVJOSUUtNC41LTIxQi1BM0ItUFQlMjIlMEElMEElMjMlMjBsb2FkJTIwdGhlJTIwdG9rZW5pemVyJTIwYW5kJTIwdGhlJTIwbW9kZWwlMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMG1vZGVsX25hbWUlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEEpJTBBJTBBJTIzJTIwcHJlcGFyZSUyMHRoZSUyMG1vZGVsJTIwaW5wdXQlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjBwcm9tcHQlN0QlMEElNUQlMEF0ZXh0JTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZSUzREZhbHNlJTJDJTBBJTIwJTIwJTIwJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUwQSklMEFtb2RlbF9pbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTVCdGV4dCU1RCUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQSUyMyUyMGNvbmR1Y3QlMjB0ZXh0JTIwY29tcGxldGlvbiUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjAqKm1vZGVsX2lucHV0cyUyQyUwQSUyMCUyMCUyMCUyMG1heF9uZXdfdG9rZW5zJTNEMzIlMkMlMEEpJTBBb3V0cHV0X2lkcyUyMCUzRCUyMGdlbmVyYXRlZF9pZHMlNUIwJTVEJTVCbGVuKG1vZGVsX2lucHV0cy5pbnB1dF9pZHMlNUIwJTVEKSUzQSU1RC50b2xpc3QoKSUwQSUwQSUyMyUyMGRlY29kZSUyMHRoZSUyMGdlbmVyYXRlZCUyMGlkcyUwQWdlbmVyYXRlX3RleHQlMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKG91dHB1dF9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model_name = <span class="hljs-string">&quot;baidu/ERNIE-4.5-21B-A3B-PT&quot;</span>

<span class="hljs-comment"># load the tokenizer and the model</span>
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=torch.bfloat16,
)

<span class="hljs-comment"># prepare the model input</span>
inputs = tokenizer(<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">False</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>
)
model_inputs = tokenizer([text], add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-comment"># conduct text completion</span>
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=<span class="hljs-number">32</span>,
)
output_ids = generated_ids[<span class="hljs-number">0</span>][<span class="hljs-built_in">len</span>(model_inputs.input_ids[<span class="hljs-number">0</span>]):].tolist()

<span class="hljs-comment"># decode the generated ids</span>
generate_text = tokenizer.decode(output_ids, skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),D=new G({props:{title:"Distributed Generation with Tensor Parallelism",local:"distributed-generation-with-tensor-parallelism",headingTag:"h3"}}),K=new Ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWxfbmFtZSUyMCUzRCUyMCUyMmJhaWR1JTJGRVJOSUUtNC41LTIxQi1BM0ItUFQlMjIlMEElMEElMjMlMjBsb2FkJTIwdGhlJTIwdG9rZW5pemVyJTIwYW5kJTIwdGhlJTIwbW9kZWwlMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMG1vZGVsX25hbWUlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjB0cF9wbGFuJTNEJTIyYXV0byUyMiUyQyUwQSklMEElMEElMjMlMjBwcmVwYXJlJTIwdGhlJTIwbW9kZWwlMjBpbnB1dCUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBcHJvbXB0JTIwJTNEJTIwJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlMjIlMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMHByb21wdCU3RCUwQSU1RCUwQXRleHQlMjAlM0QlMjB0b2tlbml6ZXIuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjBtZXNzYWdlcyUyQyUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNERmFsc2UlMkMlMEElMjAlMjAlMjAlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTBBKSUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglNUJ0ZXh0JTVEJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBJTIzJTIwY29uZHVjdCUyMHRleHQlMjBjb21wbGV0aW9uJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCUwQSUyMCUyMCUyMCUyMCoqbW9kZWxfaW5wdXRzJTJDJTBBJTIwJTIwJTIwJTIwbWF4X25ld190b2tlbnMlM0QzMiUyQyUwQSklMEFvdXRwdXRfaWRzJTIwJTNEJTIwZ2VuZXJhdGVkX2lkcyU1QjAlNUQlNUJsZW4obW9kZWxfaW5wdXRzLmlucHV0X2lkcyU1QjAlNUQpJTNBJTVELnRvbGlzdCgpJTBBJTBBJTIzJTIwZGVjb2RlJTIwdGhlJTIwZ2VuZXJhdGVkJTIwaWRzJTBBZ2VuZXJhdGVfdGV4dCUyMCUzRCUyMHRva2VuaXplci5kZWNvZGUob3V0cHV0X2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model_name = <span class="hljs-string">&quot;baidu/ERNIE-4.5-21B-A3B-PT&quot;</span>

<span class="hljs-comment"># load the tokenizer and the model</span>
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=torch.bfloat16,
    tp_plan=<span class="hljs-string">&quot;auto&quot;</span>,
)

<span class="hljs-comment"># prepare the model input</span>
inputs = tokenizer(<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">False</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>
)
model_inputs = tokenizer([text], add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-comment"># conduct text completion</span>
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=<span class="hljs-number">32</span>,
)
output_ids = generated_ids[<span class="hljs-number">0</span>][<span class="hljs-built_in">len</span>(model_inputs.input_ids[<span class="hljs-number">0</span>]):].tolist()

<span class="hljs-comment"># decode the generated ids</span>
generate_text = tokenizer.decode(output_ids, skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),ee=new G({props:{title:"Quantization with Bitsandbytes",local:"quantization-with-bitsandbytes",headingTag:"h3"}}),oe=new Ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWxfbmFtZSUyMCUzRCUyMCUyMmJhaWR1JTJGRVJOSUUtNC41LTIxQi1BM0ItUFQlMjIlMEElMEElMjMlMjBsb2FkJTIwdGhlJTIwdG9rZW5pemVyJTIwYW5kJTIwdGhlJTIwbW9kZWwlMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMG1vZGVsX25hbWUlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RCaXRzQW5kQnl0ZXNDb25maWcobG9hZF9pbl80Yml0JTNEVHJ1ZSklMkMlMEEpJTBBJTBBJTIzJTIwcHJlcGFyZSUyMHRoZSUyMG1vZGVsJTIwaW5wdXQlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjBwcm9tcHQlN0QlMEElNUQlMEF0ZXh0JTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZSUzREZhbHNlJTJDJTBBJTIwJTIwJTIwJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUwQSklMEFtb2RlbF9pbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTVCdGV4dCU1RCUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQSUyMyUyMGNvbmR1Y3QlMjB0ZXh0JTIwY29tcGxldGlvbiUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjAqKm1vZGVsX2lucHV0cyUyQyUwQSUyMCUyMCUyMCUyMG1heF9uZXdfdG9rZW5zJTNEMzIlMkMlMEEpJTBBb3V0cHV0X2lkcyUyMCUzRCUyMGdlbmVyYXRlZF9pZHMlNUIwJTVEJTVCbGVuKG1vZGVsX2lucHV0cy5pbnB1dF9pZHMlNUIwJTVEKSUzQSU1RC50b2xpc3QoKSUwQSUwQSUyMyUyMGRlY29kZSUyMHRoZSUyMGdlbmVyYXRlZCUyMGlkcyUwQWdlbmVyYXRlX3RleHQlMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKG91dHB1dF9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

model_name = <span class="hljs-string">&quot;baidu/ERNIE-4.5-21B-A3B-PT&quot;</span>

<span class="hljs-comment"># load the tokenizer and the model</span>
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=BitsAndBytesConfig(load_in_4bit=<span class="hljs-literal">True</span>),
)

<span class="hljs-comment"># prepare the model input</span>
inputs = tokenizer(<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">False</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>
)
model_inputs = tokenizer([text], add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-comment"># conduct text completion</span>
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=<span class="hljs-number">32</span>,
)
output_ids = generated_ids[<span class="hljs-number">0</span>][<span class="hljs-built_in">len</span>(model_inputs.input_ids[<span class="hljs-number">0</span>]):].tolist()

<span class="hljs-comment"># decode the generated ids</span>
generate_text = tokenizer.decode(output_ids, skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),ne=new G({props:{title:"Ernie4_5_MoeConfig",local:"transformers.Ernie4_5_MoeConfig",headingTag:"h2"}}),se=new Je({props:{name:"class transformers.Ernie4_5_MoeConfig",anchor:"transformers.Ernie4_5_MoeConfig",parameters:[{name:"vocab_size",val:" = 103424"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"hidden_size",val:" = 2560"},{name:"intermediate_size",val:" = 12288"},{name:"num_hidden_layers",val:" = 28"},{name:"num_attention_heads",val:" = 20"},{name:"num_key_value_heads",val:" = 4"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 131072"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = True"},{name:"rope_theta",val:" = 500000.0"},{name:"rope_scaling",val:" = None"},{name:"use_bias",val:" = False"},{name:"moe_intermediate_size",val:" = 1536"},{name:"moe_k",val:" = 6"},{name:"moe_num_experts",val:" = 64"},{name:"moe_num_shared_experts",val:" = 2"},{name:"moe_layer_start_index",val:" = 1"},{name:"moe_layer_end_index",val:" = -1"},{name:"moe_layer_interval",val:" = 1"},{name:"moe_norm_min",val:" = 1e-12"},{name:"output_router_logits",val:" = False"},{name:"router_aux_loss_coef",val:" = 0.001"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Ernie4_5_MoeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 103424) &#x2014;
Vocabulary size of the Ernie 4.5 MoE model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeModel">Ernie4_5_MoeModel</a>`,name:"vocab_size"},{anchor:"transformers.Ernie4_5_MoeConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.Ernie4_5_MoeConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Ernie4_5_MoeConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Ernie4_5_MoeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2560) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Ernie4_5_MoeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 12288) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Ernie4_5_MoeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 28) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Ernie4_5_MoeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 20) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Ernie4_5_MoeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>32</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Ernie4_5_MoeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Ernie4_5_MoeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 131072) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Ernie4_5_MoeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Ernie4_5_MoeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Ernie4_5_MoeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Ernie4_5_MoeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.Ernie4_5_MoeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 500000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Ernie4_5_MoeConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
and you expect the model to work on longer <code>max_position_embeddings</code>, we recommend you to update this value
accordingly.
Expected contents:
<code>rope_type</code> (<code>str</code>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;linear&#x2019;, &#x2018;dynamic&#x2019;, &#x2018;yarn&#x2019;, &#x2018;longrope&#x2019;,
&#x2018;llama3&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<code>factor</code> (<code>float</code>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings. In
most scaling types, a <code>factor</code> of x will enable the model to handle sequences of length x <em>
original maximum pre-trained length.
<code>original_max_position_embeddings</code> (<code>int</code>, </em>optional<em>):
Used with &#x2018;dynamic&#x2019;, &#x2018;longrope&#x2019; and &#x2018;llama3&#x2019;. The original max position embeddings used during
pretraining.
<code>attention_factor</code> (<code>float</code>, </em>optional<em>):
Used with &#x2018;yarn&#x2019; and &#x2018;longrope&#x2019;. The scaling factor to be applied on the attention
computation. If unspecified, it defaults to value recommended by the implementation, using the
<code>factor</code> field to infer the suggested value.
<code>beta_fast</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for extrapolation (only) in the linear
ramp function. If unspecified, it defaults to 32.
<code>beta_slow</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for interpolation (only) in the linear
ramp function. If unspecified, it defaults to 1.
<code>short_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Ernie4_5_MoeConfig.use_bias",description:`<strong>use_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in any of the projections including mlp and attention for example.`,name:"use_bias"},{anchor:"transformers.Ernie4_5_MoeConfig.moe_intermediate_size",description:`<strong>moe_intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1536) &#x2014;
Intermediate size of the routed expert.`,name:"moe_intermediate_size"},{anchor:"transformers.Ernie4_5_MoeConfig.moe_k",description:`<strong>moe_k</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of selected experts.`,name:"moe_k"},{anchor:"transformers.Ernie4_5_MoeConfig.moe_num_experts",description:`<strong>moe_num_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of routed experts.`,name:"moe_num_experts"},{anchor:"transformers.Ernie4_5_MoeConfig.moe_num_shared_experts",description:`<strong>moe_num_shared_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The number of experts that are shared for all MoE forwards.`,name:"moe_num_shared_experts"},{anchor:"transformers.Ernie4_5_MoeConfig.moe_layer_start_index",description:`<strong>moe_layer_start_index</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The first index at which MoE layers start to appear.`,name:"moe_layer_start_index"},{anchor:"transformers.Ernie4_5_MoeConfig.moe_layer_end_index",description:`<strong>moe_layer_end_index</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
The last possible index for a MoE layer.`,name:"moe_layer_end_index"},{anchor:"transformers.Ernie4_5_MoeConfig.moe_layer_interval",description:`<strong>moe_layer_interval</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The intervals between MoE layers to appear.`,name:"moe_layer_interval"},{anchor:"transformers.Ernie4_5_MoeConfig.moe_norm_min",description:`<strong>moe_norm_min</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
Minimum division value during routing normalization.`,name:"moe_norm_min"},{anchor:"transformers.Ernie4_5_MoeConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss, including load balancing loss and router z-loss.`,name:"output_router_logits"},{anchor:"transformers.Ernie4_5_MoeConfig.router_aux_loss_coef",description:`<strong>router_aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014;
The aux loss factor for the total loss.`,name:"router_aux_loss_coef"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5_moe/configuration_ernie4_5_moe.py#L24"}}),W=new zo({props:{anchor:"transformers.Ernie4_5_MoeConfig.example",$$slots:{default:[Xo]},$$scope:{ctx:E}}}),ae=new G({props:{title:"Ernie4_5_MoeModel",local:"transformers.Ernie4_5_MoeModel",headingTag:"h2"}}),re=new Je({props:{name:"class transformers.Ernie4_5_MoeModel",anchor:"transformers.Ernie4_5_MoeModel",parameters:[{name:"config",val:": Ernie4_5_MoeConfig"}],parametersDescription:[{anchor:"transformers.Ernie4_5_MoeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeConfig">Ernie4_5_MoeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py#L496"}}),ie=new Je({props:{name:"forward",anchor:"transformers.Ernie4_5_MoeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Ernie4_5_MoeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Ernie4_5_MoeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Ernie4_5_MoeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Ernie4_5_MoeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Ernie4_5_MoeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Ernie4_5_MoeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Ernie4_5_MoeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py#L513",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeConfig"
>Ernie4_5_MoeConfig</a>) and inputs.</p>
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
`}}),R=new _o({props:{$$slots:{default:[Qo]},$$scope:{ctx:E}}}),le=new G({props:{title:"Ernie4_5_MoeForCausalLM",local:"transformers.Ernie4_5_MoeForCausalLM",headingTag:"h2"}}),de=new Je({props:{name:"class transformers.Ernie4_5_MoeForCausalLM",anchor:"transformers.Ernie4_5_MoeForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Ernie4_5_MoeForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeForCausalLM">Ernie4_5_MoeForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py#L660"}}),ce=new Je({props:{name:"forward",anchor:"transformers.Ernie4_5_MoeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py#L678",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5_moe#transformers.Ernie4_5_MoeConfig"
>Ernie4_5_MoeConfig</a>) and inputs.</p>
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
`}}),N=new _o({props:{$$slots:{default:[Ho]},$$scope:{ctx:E}}}),X=new zo({props:{anchor:"transformers.Ernie4_5_MoeForCausalLM.forward.example",$$slots:{default:[Lo]},$$scope:{ctx:E}}}),me=new Je({props:{name:"generate",anchor:"transformers.Ernie4_5_MoeForCausalLM.generate",parameters:[{name:"inputs",val:": typing.Optional[torch.Tensor] = None"},{name:"generation_config",val:": typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None"},{name:"logits_processor",val:": typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None"},{name:"stopping_criteria",val:": typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None"},{name:"prefix_allowed_tokens_fn",val:": typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None"},{name:"synced_gpus",val:": typing.Optional[bool] = None"},{name:"assistant_model",val:": typing.Optional[ForwardRef('PreTrainedModel')] = None"},{name:"streamer",val:": typing.Optional[ForwardRef('BaseStreamer')] = None"},{name:"negative_prompt_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"negative_prompt_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"use_model_defaults",val:": typing.Optional[bool] = None"},{name:"custom_generate",val:": typing.Union[str, typing.Callable, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.inputs",description:`<strong>inputs</strong> (<code>torch.Tensor</code> of varying shape depending on the modality, <em>optional</em>) &#x2014;
The sequence used as a prompt for the generation or as model inputs to the encoder. If <code>None</code> the
method initializes it with <code>bos_token_id</code> and a batch size of 1. For decoder-only models <code>inputs</code>
should be in the format of <code>input_ids</code>. For encoder-decoder models <em>inputs</em> can represent any of
<code>input_ids</code>, <code>input_values</code>, <code>input_features</code>, or <code>pixel_values</code>.`,name:"inputs"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.generation_config",description:`<strong>generation_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>, <em>optional</em>) &#x2014;
The generation configuration to be used as base parametrization for the generation call. <code>**kwargs</code>
passed to generate matching the attributes of <code>generation_config</code> will override them. If
<code>generation_config</code> is not provided, the default will be used, which has the following loading
priority: 1) from the <code>generation_config.json</code> model file, if it exists; 2) from the model
configuration. Please note that unspecified parameters will inherit <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>&#x2019;s
default values, whose documentation should be checked to parameterize generation.`,name:"generation_config"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.logits_processor",description:`<strong>logits_processor</strong> (<code>LogitsProcessorList</code>, <em>optional</em>) &#x2014;
Custom logits processors that complement the default logits processors built from arguments and
generation config. If a logit processor is passed that is already created with the arguments or a
generation config an error is thrown. This feature is intended for advanced users.`,name:"logits_processor"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.stopping_criteria",description:`<strong>stopping_criteria</strong> (<code>StoppingCriteriaList</code>, <em>optional</em>) &#x2014;
Custom stopping criteria that complements the default stopping criteria built from arguments and a
generation config. If a stopping criteria is passed that is already created with the arguments or a
generation config an error is thrown. If your stopping criteria depends on the <code>scores</code> input, make
sure you pass <code>return_dict_in_generate=True, output_scores=True</code> to <code>generate</code>. This feature is
intended for advanced users.`,name:"stopping_criteria"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.prefix_allowed_tokens_fn",description:`<strong>prefix_allowed_tokens_fn</strong> (<code>Callable[[int, torch.Tensor], list[int]]</code>, <em>optional</em>) &#x2014;
If provided, this function constraints the beam search to allowed tokens only at each step. If not
provided no constraint is applied. This function takes 2 arguments: the batch ID <code>batch_id</code> and
<code>input_ids</code>. It has to return a list with the allowed tokens for the next generation step conditioned
on the batch ID <code>batch_id</code> and the previously generated tokens <code>inputs_ids</code>. This argument is useful
for constrained generation conditioned on the prefix, as described in <a href="https://huggingface.co/papers/2010.00904" rel="nofollow">Autoregressive Entity
Retrieval</a>.`,name:"prefix_allowed_tokens_fn"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.synced_gpus",description:`<strong>synced_gpus</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
to <code>True</code> if using <code>FullyShardedDataParallel</code> or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to <code>False</code>.`,name:"synced_gpus"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.assistant_model",description:`<strong>assistant_model</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
An assistant model that can be used to accelerate generation. The assistant model must have the exact
same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
is much faster than running generation with the model you&#x2019;re calling generate from. As such, the
assistant model should be much smaller.`,name:"assistant_model"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.streamer",description:`<strong>streamer</strong> (<code>BaseStreamer</code>, <em>optional</em>) &#x2014;
Streamer object that will be used to stream the generated sequences. Generated tokens are passed
through <code>streamer.put(token_ids)</code> and the streamer is responsible for any further processing.`,name:"streamer"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.negative_prompt_ids",description:`<strong>negative_prompt_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
The negative prompt needed for some processors such as CFG. The batch size must match the input batch
size. This is an experimental feature, subject to breaking API changes in future versions.`,name:"negative_prompt_ids"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.negative_prompt_attention_mask",description:`<strong>negative_prompt_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Attention_mask for <code>negative_prompt_ids</code>.`,name:"negative_prompt_attention_mask"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.use_model_defaults",description:`<strong>use_model_defaults</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
When it is <code>True</code>, unset parameters in <code>generation_config</code> will be set to the model-specific default
generation configuration (<code>model.generation_config</code>), as opposed to the global defaults
(<code>GenerationConfig()</code>). If unset, models saved starting from <code>v4.50</code> will consider this flag to be
<code>True</code>.`,name:"use_model_defaults"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.custom_generate",description:`<strong>custom_generate</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>) &#x2014;
One of the following:<ul>
<li><code>str</code> (Hugging Face Hub repository name): runs the custom <code>generate</code> function defined at
<code>custom_generate/generate.py</code> in that repository instead of the standard <code>generate</code> method. The
repository fully replaces the generation logic, and the return type may differ.</li>
<li><code>str</code> (local repository path): same as above but from a local path, <code>trust_remote_code</code> not required.</li>
<li><code>Callable</code>: <code>generate</code> will perform the usual input preparation steps, then call the provided callable to
run the decoding loop.
For more information, see <a href="../../generation_strategies#custom-generation-methods">the docs</a>.</li>
</ul>`,name:"custom_generate"},{anchor:"transformers.Ernie4_5_MoeForCausalLM.generate.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Ad hoc parametrization of <code>generation_config</code> and/or additional model-specific kwargs that will be
forwarded to the <code>forward</code> function of the model. If the model is an encoder-decoder model, encoder
specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with <em>decoder_</em>.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L2140",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> (if <code>return_dict_in_generate=True</code>
or when <code>config.return_dict_in_generate=True</code>) or a <code>torch.LongTensor</code>.</p>
<p>If the model is <em>not</em> an encoder-decoder model (<code>model.config.is_encoder_decoder=False</code>), the possible
<a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> types are:</p>
<ul>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput"
>GenerateDecoderOnlyOutput</a>,</li>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput"
>GenerateBeamDecoderOnlyOutput</a></li>
</ul>
<p>If the model is an encoder-decoder model (<code>model.config.is_encoder_decoder=True</code>), the possible
<a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> types are:</p>
<ul>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput"
>GenerateEncoderDecoderOutput</a>,</li>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput"
>GenerateBeamEncoderDecoderOutput</a></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> or <code>torch.LongTensor</code></p>
`}}),Q=new _o({props:{warning:!0,$$slots:{default:[qo]},$$scope:{ctx:E}}}),pe=new No({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/ernie4_5_moe.md"}}),{c(){n=d("meta"),b=s(),i=d("p"),p=s(),w=d("p"),w.innerHTML=m,U=s(),B=d("div"),B.innerHTML=Mo,Ce=s(),u(L.$$.fragment),$e=s(),u(q.$$.fragment),je=s(),A=d("p"),A.innerHTML=yo,Ie=s(),S=d("p"),S.innerHTML=bo,Fe=s(),V=d("div"),V.innerHTML=To,ze=s(),u(P.$$.fragment),Ze=s(),u(Y.$$.fragment),Ge=s(),u(O.$$.fragment),Be=s(),u(D.$$.fragment),Ve=s(),u(K.$$.fragment),We=s(),u(ee.$$.fragment),Re=s(),u(oe.$$.fragment),Ne=s(),te=d("p"),te.innerHTML=wo,Xe=s(),u(ne.$$.fragment),Qe=s(),J=d("div"),u(se.$$.fragment),Oe=s(),ue=d("p"),ue.innerHTML=vo,De=s(),he=d("p"),he.innerHTML=ko,Ke=s(),u(W.$$.fragment),He=s(),u(ae.$$.fragment),Le=s(),v=d("div"),u(re.$$.fragment),eo=s(),fe=d("p"),fe.textContent=Jo,oo=s(),ge=d("p"),ge.innerHTML=xo,to=s(),_e=d("p"),_e.innerHTML=Eo,no=s(),I=d("div"),u(ie.$$.fragment),so=s(),Me=d("p"),Me.innerHTML=Uo,ao=s(),u(R.$$.fragment),qe=s(),u(le.$$.fragment),Ae=s(),T=d("div"),u(de.$$.fragment),ro=s(),ye=d("p"),ye.textContent=Co,io=s(),be=d("p"),be.innerHTML=$o,lo=s(),Te=d("p"),Te.innerHTML=jo,co=s(),C=d("div"),u(ce.$$.fragment),mo=s(),we=d("p"),we.innerHTML=Io,po=s(),u(N.$$.fragment),uo=s(),u(X.$$.fragment),ho=s(),F=d("div"),u(me.$$.fragment),fo=s(),ve=d("p"),ve.textContent=Fo,go=s(),u(Q.$$.fragment),Se=s(),u(pe.$$.fragment),Pe=s(),xe=d("p"),this.h()},l(e){const o=Wo("svelte-u9bgzb",document.head);n=c(o,"META",{name:!0,content:!0}),o.forEach(t),b=a(e),i=c(e,"P",{}),H(i).forEach(t),p=a(e),w=c(e,"P",{"data-svelte-h":!0}),y(w)!=="svelte-z3yyfn"&&(w.innerHTML=m),U=a(e),B=c(e,"DIV",{style:!0,"data-svelte-h":!0}),y(B)!=="svelte-11gpmgv"&&(B.innerHTML=Mo),Ce=a(e),h(L.$$.fragment,e),$e=a(e),h(q.$$.fragment,e),je=a(e),A=c(e,"P",{"data-svelte-h":!0}),y(A)!=="svelte-q3c1bh"&&(A.innerHTML=yo),Ie=a(e),S=c(e,"P",{"data-svelte-h":!0}),y(S)!=="svelte-1oat5mq"&&(S.innerHTML=bo),Fe=a(e),V=c(e,"DIV",{class:!0,"data-svelte-h":!0}),y(V)!=="svelte-xmifer"&&(V.innerHTML=To),ze=a(e),h(P.$$.fragment,e),Ze=a(e),h(Y.$$.fragment,e),Ge=a(e),h(O.$$.fragment,e),Be=a(e),h(D.$$.fragment,e),Ve=a(e),h(K.$$.fragment,e),We=a(e),h(ee.$$.fragment,e),Re=a(e),h(oe.$$.fragment,e),Ne=a(e),te=c(e,"P",{"data-svelte-h":!0}),y(te)!=="svelte-wizmvu"&&(te.innerHTML=wo),Xe=a(e),h(ne.$$.fragment,e),Qe=a(e),J=c(e,"DIV",{class:!0});var $=H(J);h(se.$$.fragment,$),Oe=a($),ue=c($,"P",{"data-svelte-h":!0}),y(ue)!=="svelte-l12ded"&&(ue.innerHTML=vo),De=a($),he=c($,"P",{"data-svelte-h":!0}),y(he)!=="svelte-1ek1ss9"&&(he.innerHTML=ko),Ke=a($),h(W.$$.fragment,$),$.forEach(t),He=a(e),h(ae.$$.fragment,e),Le=a(e),v=c(e,"DIV",{class:!0});var x=H(v);h(re.$$.fragment,x),eo=a(x),fe=c(x,"P",{"data-svelte-h":!0}),y(fe)!=="svelte-1dlcazj"&&(fe.textContent=Jo),oo=a(x),ge=c(x,"P",{"data-svelte-h":!0}),y(ge)!=="svelte-q52n56"&&(ge.innerHTML=xo),to=a(x),_e=c(x,"P",{"data-svelte-h":!0}),y(_e)!=="svelte-hswkmf"&&(_e.innerHTML=Eo),no=a(x),I=c(x,"DIV",{class:!0});var z=H(I);h(ie.$$.fragment,z),so=a(z),Me=c(z,"P",{"data-svelte-h":!0}),y(Me)!=="svelte-4l8wrw"&&(Me.innerHTML=Uo),ao=a(z),h(R.$$.fragment,z),z.forEach(t),x.forEach(t),qe=a(e),h(le.$$.fragment,e),Ae=a(e),T=c(e,"DIV",{class:!0});var k=H(T);h(de.$$.fragment,k),ro=a(k),ye=c(k,"P",{"data-svelte-h":!0}),y(ye)!=="svelte-1xa5h1q"&&(ye.textContent=Co),io=a(k),be=c(k,"P",{"data-svelte-h":!0}),y(be)!=="svelte-q52n56"&&(be.innerHTML=$o),lo=a(k),Te=c(k,"P",{"data-svelte-h":!0}),y(Te)!=="svelte-hswkmf"&&(Te.innerHTML=jo),co=a(k),C=c(k,"DIV",{class:!0});var j=H(C);h(ce.$$.fragment,j),mo=a(j),we=c(j,"P",{"data-svelte-h":!0}),y(we)!=="svelte-9lktqc"&&(we.innerHTML=Io),po=a(j),h(N.$$.fragment,j),uo=a(j),h(X.$$.fragment,j),j.forEach(t),ho=a(k),F=c(k,"DIV",{class:!0});var ke=H(F);h(me.$$.fragment,ke),fo=a(ke),ve=c(ke,"P",{"data-svelte-h":!0}),y(ve)!=="svelte-s5ko3x"&&(ve.textContent=Fo),go=a(ke),h(Q.$$.fragment,ke),ke.forEach(t),k.forEach(t),Se=a(e),h(pe.$$.fragment,e),Pe=a(e),xe=c(e,"P",{}),H(xe).forEach(t),this.h()},h(){Z(n,"name","hf:doc:metadata"),Z(n,"content",So),Ro(B,"float","right"),Z(V,"class","flex justify-center"),Z(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){l(document.head,n),r(e,b,o),r(e,i,o),r(e,p,o),r(e,w,o),r(e,U,o),r(e,B,o),r(e,Ce,o),f(L,e,o),r(e,$e,o),f(q,e,o),r(e,je,o),r(e,A,o),r(e,Ie,o),r(e,S,o),r(e,Fe,o),r(e,V,o),r(e,ze,o),f(P,e,o),r(e,Ze,o),f(Y,e,o),r(e,Ge,o),f(O,e,o),r(e,Be,o),f(D,e,o),r(e,Ve,o),f(K,e,o),r(e,We,o),f(ee,e,o),r(e,Re,o),f(oe,e,o),r(e,Ne,o),r(e,te,o),r(e,Xe,o),f(ne,e,o),r(e,Qe,o),r(e,J,o),f(se,J,null),l(J,Oe),l(J,ue),l(J,De),l(J,he),l(J,Ke),f(W,J,null),r(e,He,o),f(ae,e,o),r(e,Le,o),r(e,v,o),f(re,v,null),l(v,eo),l(v,fe),l(v,oo),l(v,ge),l(v,to),l(v,_e),l(v,no),l(v,I),f(ie,I,null),l(I,so),l(I,Me),l(I,ao),f(R,I,null),r(e,qe,o),f(le,e,o),r(e,Ae,o),r(e,T,o),f(de,T,null),l(T,ro),l(T,ye),l(T,io),l(T,be),l(T,lo),l(T,Te),l(T,co),l(T,C),f(ce,C,null),l(C,mo),l(C,we),l(C,po),f(N,C,null),l(C,uo),f(X,C,null),l(T,ho),l(T,F),f(me,F,null),l(F,fo),l(F,ve),l(F,go),f(Q,F,null),r(e,Se,o),f(pe,e,o),r(e,Pe,o),r(e,xe,o),Ye=!0},p(e,[o]){const $={};o&2&&($.$$scope={dirty:o,ctx:e}),W.$set($);const x={};o&2&&(x.$$scope={dirty:o,ctx:e}),R.$set(x);const z={};o&2&&(z.$$scope={dirty:o,ctx:e}),N.$set(z);const k={};o&2&&(k.$$scope={dirty:o,ctx:e}),X.$set(k);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),Q.$set(j)},i(e){Ye||(g(L.$$.fragment,e),g(q.$$.fragment,e),g(P.$$.fragment,e),g(Y.$$.fragment,e),g(O.$$.fragment,e),g(D.$$.fragment,e),g(K.$$.fragment,e),g(ee.$$.fragment,e),g(oe.$$.fragment,e),g(ne.$$.fragment,e),g(se.$$.fragment,e),g(W.$$.fragment,e),g(ae.$$.fragment,e),g(re.$$.fragment,e),g(ie.$$.fragment,e),g(R.$$.fragment,e),g(le.$$.fragment,e),g(de.$$.fragment,e),g(ce.$$.fragment,e),g(N.$$.fragment,e),g(X.$$.fragment,e),g(me.$$.fragment,e),g(Q.$$.fragment,e),g(pe.$$.fragment,e),Ye=!0)},o(e){_(L.$$.fragment,e),_(q.$$.fragment,e),_(P.$$.fragment,e),_(Y.$$.fragment,e),_(O.$$.fragment,e),_(D.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(se.$$.fragment,e),_(W.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(ie.$$.fragment,e),_(R.$$.fragment,e),_(le.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(N.$$.fragment,e),_(X.$$.fragment,e),_(me.$$.fragment,e),_(Q.$$.fragment,e),_(pe.$$.fragment,e),Ye=!1},d(e){e&&(t(b),t(i),t(p),t(w),t(U),t(B),t(Ce),t($e),t(je),t(A),t(Ie),t(S),t(Fe),t(V),t(ze),t(Ze),t(Ge),t(Be),t(Ve),t(We),t(Re),t(Ne),t(te),t(Xe),t(Qe),t(J),t(He),t(Le),t(v),t(qe),t(Ae),t(T),t(Se),t(Pe),t(xe)),t(n),M(L,e),M(q,e),M(P,e),M(Y,e),M(O,e),M(D,e),M(K,e),M(ee,e),M(oe,e),M(ne,e),M(se),M(W),M(ae,e),M(re),M(ie),M(R),M(le,e),M(de),M(ce),M(N),M(X),M(me),M(Q),M(pe,e)}}}const So='{"title":"Ernie 4.5 Moe","local":"ernie-45-moe","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage Tips","local":"usage-tips","sections":[{"title":"Generate text","local":"generate-text","sections":[],"depth":3},{"title":"Distributed Generation with Tensor Parallelism","local":"distributed-generation-with-tensor-parallelism","sections":[],"depth":3},{"title":"Quantization with Bitsandbytes","local":"quantization-with-bitsandbytes","sections":[],"depth":3}],"depth":2},{"title":"Ernie4_5_MoeConfig","local":"transformers.Ernie4_5_MoeConfig","sections":[],"depth":2},{"title":"Ernie4_5_MoeModel","local":"transformers.Ernie4_5_MoeModel","sections":[],"depth":2},{"title":"Ernie4_5_MoeForCausalLM","local":"transformers.Ernie4_5_MoeForCausalLM","sections":[],"depth":2}],"depth":1}';function Po(E){return Go(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class nt extends Bo{constructor(n){super(),Vo(this,n,Po,Ao,Zo,{})}}export{nt as component};
