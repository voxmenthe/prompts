import{s as De,o as Ke,n as Ge}from"../chunks/scheduler.18a86fab.js";import{S as et,i as tt,g as l,s as r,r as m,A as ot,h as c,f as n,c as s,j as K,x as v,u,k as ee,y as d,a as i,v as f,d as g,t as _,w as M}from"../chunks/index.98837b22.js";import{T as Oe}from"../chunks/Tip.77304350.js";import{D as le}from"../chunks/Docstring.a1ef7999.js";import{C as Le}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Qe}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ce,E as nt}from"../chunks/getInferenceSnippets.06c2775f.js";function at(x){let o,p;return o=new Le({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdyYW5pdGVNb2VTaGFyZWRNb2RlbCUyQyUyMEdyYW5pdGVNb2VTaGFyZWRDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwR3Jhbml0ZU1vZVNoYXJlZCUyMGdyYW5pdGVtb2UtM2IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwR3Jhbml0ZU1vZVNoYXJlZENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMGdyYW5pdGVtb2UtN2IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEdyYW5pdGVNb2VTaGFyZWRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> GraniteMoeSharedModel, GraniteMoeSharedConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a GraniteMoeShared granitemoe-3b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = GraniteMoeSharedConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the granitemoe-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GraniteMoeSharedModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){m(o.$$.fragment)},l(a){u(o.$$.fragment,a)},m(a,h){f(o,a,h),p=!0},p:Ge,i(a){p||(g(o.$$.fragment,a),p=!0)},o(a){_(o.$$.fragment,a),p=!1},d(a){M(o,a)}}}function rt(x){let o,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=p},l(a){o=c(a,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=p)},m(a,h){i(a,o,h)},p:Ge,d(a){a&&n(o)}}}function st(x){let o,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=p},l(a){o=c(a,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=p)},m(a,h){i(a,o,h)},p:Ge,d(a){a&&n(o)}}}function it(x){let o,p="Example:",a,h,k;return h=new Le({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHcmFuaXRlTW9lU2hhcmVkRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMEdyYW5pdGVNb2VTaGFyZWRGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyaWJtJTJGUG93ZXJNb0UtM2IlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyaWJtJTJGUG93ZXJNb0UtM2IlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNEMzApJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GraniteMoeSharedForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = GraniteMoeSharedForCausalLM.from_pretrained(<span class="hljs-string">&quot;ibm/PowerMoE-3b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;ibm/PowerMoE-3b&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=l("p"),o.textContent=p,a=r(),m(h.$$.fragment)},l(b){o=c(b,"P",{"data-svelte-h":!0}),v(o)!=="svelte-11lpom8"&&(o.textContent=p),a=s(b),u(h.$$.fragment,b)},m(b,z){i(b,o,z),i(b,a,z),f(h,b,z),k=!0},p:Ge,i(b){k||(g(h.$$.fragment,b),k=!0)},o(b){_(h.$$.fragment,b),k=!1},d(b){b&&(n(o),n(a)),M(h,b)}}}function dt(x){let o,p,a,h,k,b="<em>This model was released on 2024-08-23 and added to Hugging Face Transformers on 2025-02-14.</em>",z,U,pe,B,he,L,Ne='The GraniteMoe model was proposed in <a href="https://huggingface.co/papers/2408.13359" rel="nofollow">Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler</a> by Yikang Shen, Matthew Stallone, Mayank Mishra, Gaoyuan Zhang, Shawn Tan, Aditya Prasad, Adriana Meza Soria, David D. Cox and Rameswar Panda.',me,N,qe="Additionally this class GraniteMoeSharedModel adds shared experts for Moe.",ue,q,fe,H,He='This HF implementation is contributed by <a href="https://huggingface.co/mayank-mishra" rel="nofollow">Mayank Mishra</a>, <a href="https://huggingface.co/shawntan" rel="nofollow">Shawn Tan</a> and <a href="https://huggingface.co/SukritiSharma" rel="nofollow">Sukriti Sharma</a>.',ge,R,_e,T,X,$e,te,Re=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedModel">GraniteMoeSharedModel</a>. It is used to instantiate an GraniteMoeShared
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the <a href="https://huggingface.co/ibm-research/moe-7b-1b-active-shared-experts" rel="nofollow">ibm-research/moe-7b-1b-active-shared-experts</a>.`,xe,oe,Xe=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ce,I,Me,V,be,y,E,Se,ne,Ve="The bare Granitemoeshared Model outputting raw hidden-states without any specific head on top.",Je,ae,Ee=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ze,re,Pe=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ie,C,P,Fe,se,Ye='The <a href="/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedModel">GraniteMoeSharedModel</a> forward method, overrides the <code>__call__</code> special method.',je,F,ye,Y,ve,S,A,Ze,G,O,We,ie,Ae='The <a href="/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedForCausalLM">GraniteMoeSharedForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ue,j,Be,Z,Te,Q,we,de,ke;return U=new ce({props:{title:"GraniteMoeShared",local:"granitemoeshared",headingTag:"h1"}}),B=new ce({props:{title:"Overview",local:"overview",headingTag:"h2"}}),q=new Le({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWxfcGF0aCUyMCUzRCUyMCUyMmlibS1yZXNlYXJjaCUyRm1vZS03Yi0xYi1hY3RpdmUtc2hhcmVkLWV4cGVydHMlMjIlMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9wYXRoKSUwQSUwQSUyMyUyMGRyb3AlMjBkZXZpY2VfbWFwJTIwaWYlMjBydW5uaW5nJTIwb24lMjBDUFUlMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZChtb2RlbF9wYXRoJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIpJTBBbW9kZWwuZXZhbCgpJTBBJTBBJTIzJTIwY2hhbmdlJTIwaW5wdXQlMjB0ZXh0JTIwYXMlMjBkZXNpcmVkJTBBcHJvbXB0JTIwJTNEJTIwJTIyV3JpdGUlMjBhJTIwY29kZSUyMHRvJTIwZmluZCUyMHRoZSUyMG1heGltdW0lMjB2YWx1ZSUyMGluJTIwYSUyMGxpc3QlMjBvZiUyMG51bWJlcnMuJTIyJTBBJTBBJTIzJTIwdG9rZW5pemUlMjB0aGUlMjB0ZXh0JTBBaW5wdXRfdG9rZW5zJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTIzJTIwZ2VuZXJhdGUlMjBvdXRwdXQlMjB0b2tlbnMlMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X3Rva2VucyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMTAwKSUwQSUyMyUyMGRlY29kZSUyMG91dHB1dCUyMHRva2VucyUyMGludG8lMjB0ZXh0JTBBb3V0cHV0JTIwJTNEJTIwdG9rZW5pemVyLmJhdGNoX2RlY29kZShvdXRwdXQpJTBBJTIzJTIwbG9vcCUyMG92ZXIlMjB0aGUlMjBiYXRjaCUyMHRvJTIwcHJpbnQlMkMlMjBpbiUyMHRoaXMlMjBleGFtcGxlJTIwdGhlJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBZm9yJTIwaSUyMGluJTIwb3V0cHV0JTNBJTBBJTIwJTIwJTIwJTIwcHJpbnQoaSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model_path = <span class="hljs-string">&quot;ibm-research/moe-7b-1b-active-shared-experts&quot;</span>
tokenizer = AutoTokenizer.from_pretrained(model_path)

<span class="hljs-comment"># drop device_map if running on CPU</span>
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=<span class="hljs-string">&quot;auto&quot;</span>)
model.<span class="hljs-built_in">eval</span>()

<span class="hljs-comment"># change input text as desired</span>
prompt = <span class="hljs-string">&quot;Write a code to find the maximum value in a list of numbers.&quot;</span>

<span class="hljs-comment"># tokenize the text</span>
input_tokens = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-comment"># generate output tokens</span>
output = model.generate(**input_tokens, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-comment"># decode output tokens into text</span>
output = tokenizer.batch_decode(output)
<span class="hljs-comment"># loop over the batch to print, in this example the batch size is 1</span>
<span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> output:
    <span class="hljs-built_in">print</span>(i)`,wrap:!1}}),R=new ce({props:{title:"GraniteMoeSharedConfig",local:"transformers.GraniteMoeSharedConfig",headingTag:"h2"}}),X=new le({props:{name:"class transformers.GraniteMoeSharedConfig",anchor:"transformers.GraniteMoeSharedConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 11008"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"embedding_multiplier",val:" = 1.0"},{name:"logits_scaling",val:" = 1.0"},{name:"residual_multiplier",val:" = 1.0"},{name:"attention_multiplier",val:" = 1.0"},{name:"num_local_experts",val:" = 8"},{name:"num_experts_per_tok",val:" = 2"},{name:"output_router_logits",val:" = False"},{name:"router_aux_loss_coef",val:" = 0.001"},{name:"shared_intermediate_size",val:" = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GraniteMoeSharedConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the GraniteMoeShared model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedModel">GraniteMoeSharedModel</a>`,name:"vocab_size"},{anchor:"transformers.GraniteMoeSharedConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.GraniteMoeSharedConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.GraniteMoeSharedConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.GraniteMoeSharedConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.GraniteMoeSharedConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.GraniteMoeSharedConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.GraniteMoeSharedConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.GraniteMoeSharedConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.GraniteMoeSharedConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.GraniteMoeSharedConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.GraniteMoeSharedConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.GraniteMoeSharedConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.GraniteMoeSharedConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.GraniteMoeSharedConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.GraniteMoeSharedConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.GraniteMoeSharedConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
<code>{&quot;type&quot;: strategy name, &quot;factor&quot;: scaling factor}</code>. When using this flag, don&#x2019;t update
<code>max_position_embeddings</code> to the expected new maximum. See the following thread for more information on how
these scaling strategies behave:
<a href="https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/" rel="nofollow">https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/</a>. This is an
experimental feature, subject to breaking API changes in future versions.`,name:"rope_scaling"},{anchor:"transformers.GraniteMoeSharedConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.GraniteMoeSharedConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.GraniteMoeSharedConfig.embedding_multiplier",description:"<strong>embedding_multiplier</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014; embedding multiplier",name:"embedding_multiplier"},{anchor:"transformers.GraniteMoeSharedConfig.logits_scaling",description:"<strong>logits_scaling</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014; divisor for output logits",name:"logits_scaling"},{anchor:"transformers.GraniteMoeSharedConfig.residual_multiplier",description:"<strong>residual_multiplier</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014; residual multiplier",name:"residual_multiplier"},{anchor:"transformers.GraniteMoeSharedConfig.attention_multiplier",description:"<strong>attention_multiplier</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014; attention multiplier",name:"attention_multiplier"},{anchor:"transformers.GraniteMoeSharedConfig.num_local_experts",description:"<strong>num_local_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014; total number of experts",name:"num_local_experts"},{anchor:"transformers.GraniteMoeSharedConfig.num_experts_per_tok",description:"<strong>num_experts_per_tok</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014; number of experts per token",name:"num_experts_per_tok"},{anchor:"transformers.GraniteMoeSharedConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss.`,name:"output_router_logits"},{anchor:"transformers.GraniteMoeSharedConfig.router_aux_loss_coef",description:"<strong>router_aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014; router auxiliary loss coefficient",name:"router_aux_loss_coef"},{anchor:"transformers.GraniteMoeSharedConfig.shared_intermediate_size",description:`<strong>shared_intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014; intermediate size for shared experts. 0 implies
no shared experts.`,name:"shared_intermediate_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granitemoeshared/configuration_granitemoeshared.py#L30"}}),I=new Qe({props:{anchor:"transformers.GraniteMoeSharedConfig.example",$$slots:{default:[at]},$$scope:{ctx:x}}}),V=new ce({props:{title:"GraniteMoeSharedModel",local:"transformers.GraniteMoeSharedModel",headingTag:"h2"}}),E=new le({props:{name:"class transformers.GraniteMoeSharedModel",anchor:"transformers.GraniteMoeSharedModel",parameters:[{name:"config",val:": GraniteMoeSharedConfig"}],parametersDescription:[{anchor:"transformers.GraniteMoeSharedModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedConfig">GraniteMoeSharedConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granitemoeshared/modeling_granitemoeshared.py#L589"}}),P=new le({props:{name:"forward",anchor:"transformers.GraniteMoeSharedModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GraniteMoeSharedModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GraniteMoeSharedModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GraniteMoeSharedModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GraniteMoeSharedModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GraniteMoeSharedModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GraniteMoeSharedModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GraniteMoeSharedModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GraniteMoeSharedModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GraniteMoeSharedModel.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.GraniteMoeSharedModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.GraniteMoeSharedModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granitemoeshared/modeling_granitemoeshared.py#L615",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedConfig"
>GraniteMoeSharedConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),F=new Oe({props:{$$slots:{default:[rt]},$$scope:{ctx:x}}}),Y=new ce({props:{title:"GraniteMoeSharedForCausalLM",local:"transformers.GraniteMoeSharedForCausalLM",headingTag:"h2"}}),A=new le({props:{name:"class transformers.GraniteMoeSharedForCausalLM",anchor:"transformers.GraniteMoeSharedForCausalLM",parameters:[{name:"config",val:": GraniteMoeSharedConfig"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granitemoeshared/modeling_granitemoeshared.py#L936"}}),O=new le({props:{name:"forward",anchor:"transformers.GraniteMoeSharedForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/granitemoeshared/modeling_granitemoeshared.py#L952",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/granitemoeshared#transformers.GraniteMoeSharedConfig"
>GraniteMoeSharedConfig</a>) and inputs.</p>
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
`}}),j=new Oe({props:{$$slots:{default:[st]},$$scope:{ctx:x}}}),Z=new Qe({props:{anchor:"transformers.GraniteMoeSharedForCausalLM.forward.example",$$slots:{default:[it]},$$scope:{ctx:x}}}),Q=new nt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/granitemoeshared.md"}}),{c(){o=l("meta"),p=r(),a=l("p"),h=r(),k=l("p"),k.innerHTML=b,z=r(),m(U.$$.fragment),pe=r(),m(B.$$.fragment),he=r(),L=l("p"),L.innerHTML=Ne,me=r(),N=l("p"),N.textContent=qe,ue=r(),m(q.$$.fragment),fe=r(),H=l("p"),H.innerHTML=He,ge=r(),m(R.$$.fragment),_e=r(),T=l("div"),m(X.$$.fragment),$e=r(),te=l("p"),te.innerHTML=Re,xe=r(),oe=l("p"),oe.innerHTML=Xe,Ce=r(),m(I.$$.fragment),Me=r(),m(V.$$.fragment),be=r(),y=l("div"),m(E.$$.fragment),Se=r(),ne=l("p"),ne.textContent=Ve,Je=r(),ae=l("p"),ae.innerHTML=Ee,ze=r(),re=l("p"),re.innerHTML=Pe,Ie=r(),C=l("div"),m(P.$$.fragment),Fe=r(),se=l("p"),se.innerHTML=Ye,je=r(),m(F.$$.fragment),ye=r(),m(Y.$$.fragment),ve=r(),S=l("div"),m(A.$$.fragment),Ze=r(),G=l("div"),m(O.$$.fragment),We=r(),ie=l("p"),ie.innerHTML=Ae,Ue=r(),m(j.$$.fragment),Be=r(),m(Z.$$.fragment),Te=r(),m(Q.$$.fragment),we=r(),de=l("p"),this.h()},l(e){const t=ot("svelte-u9bgzb",document.head);o=c(t,"META",{name:!0,content:!0}),t.forEach(n),p=s(e),a=c(e,"P",{}),K(a).forEach(n),h=s(e),k=c(e,"P",{"data-svelte-h":!0}),v(k)!=="svelte-1hwh779"&&(k.innerHTML=b),z=s(e),u(U.$$.fragment,e),pe=s(e),u(B.$$.fragment,e),he=s(e),L=c(e,"P",{"data-svelte-h":!0}),v(L)!=="svelte-1381kj"&&(L.innerHTML=Ne),me=s(e),N=c(e,"P",{"data-svelte-h":!0}),v(N)!=="svelte-1wp58mh"&&(N.textContent=qe),ue=s(e),u(q.$$.fragment,e),fe=s(e),H=c(e,"P",{"data-svelte-h":!0}),v(H)!=="svelte-1koc4oh"&&(H.innerHTML=He),ge=s(e),u(R.$$.fragment,e),_e=s(e),T=c(e,"DIV",{class:!0});var $=K(T);u(X.$$.fragment,$),$e=s($),te=c($,"P",{"data-svelte-h":!0}),v(te)!=="svelte-1qylv92"&&(te.innerHTML=Re),xe=s($),oe=c($,"P",{"data-svelte-h":!0}),v(oe)!=="svelte-1ek1ss9"&&(oe.innerHTML=Xe),Ce=s($),u(I.$$.fragment,$),$.forEach(n),Me=s(e),u(V.$$.fragment,e),be=s(e),y=c(e,"DIV",{class:!0});var w=K(y);u(E.$$.fragment,w),Se=s(w),ne=c(w,"P",{"data-svelte-h":!0}),v(ne)!=="svelte-1ysrmto"&&(ne.textContent=Ve),Je=s(w),ae=c(w,"P",{"data-svelte-h":!0}),v(ae)!=="svelte-q52n56"&&(ae.innerHTML=Ee),ze=s(w),re=c(w,"P",{"data-svelte-h":!0}),v(re)!=="svelte-hswkmf"&&(re.innerHTML=Pe),Ie=s(w),C=c(w,"DIV",{class:!0});var J=K(C);u(P.$$.fragment,J),Fe=s(J),se=c(J,"P",{"data-svelte-h":!0}),v(se)!=="svelte-5cov27"&&(se.innerHTML=Ye),je=s(J),u(F.$$.fragment,J),J.forEach(n),w.forEach(n),ye=s(e),u(Y.$$.fragment,e),ve=s(e),S=c(e,"DIV",{class:!0});var D=K(S);u(A.$$.fragment,D),Ze=s(D),G=c(D,"DIV",{class:!0});var W=K(G);u(O.$$.fragment,W),We=s(W),ie=c(W,"P",{"data-svelte-h":!0}),v(ie)!=="svelte-1wx7d57"&&(ie.innerHTML=Ae),Ue=s(W),u(j.$$.fragment,W),Be=s(W),u(Z.$$.fragment,W),W.forEach(n),D.forEach(n),Te=s(e),u(Q.$$.fragment,e),we=s(e),de=c(e,"P",{}),K(de).forEach(n),this.h()},h(){ee(o,"name","hf:doc:metadata"),ee(o,"content",lt),ee(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ee(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ee(y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ee(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ee(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){d(document.head,o),i(e,p,t),i(e,a,t),i(e,h,t),i(e,k,t),i(e,z,t),f(U,e,t),i(e,pe,t),f(B,e,t),i(e,he,t),i(e,L,t),i(e,me,t),i(e,N,t),i(e,ue,t),f(q,e,t),i(e,fe,t),i(e,H,t),i(e,ge,t),f(R,e,t),i(e,_e,t),i(e,T,t),f(X,T,null),d(T,$e),d(T,te),d(T,xe),d(T,oe),d(T,Ce),f(I,T,null),i(e,Me,t),f(V,e,t),i(e,be,t),i(e,y,t),f(E,y,null),d(y,Se),d(y,ne),d(y,Je),d(y,ae),d(y,ze),d(y,re),d(y,Ie),d(y,C),f(P,C,null),d(C,Fe),d(C,se),d(C,je),f(F,C,null),i(e,ye,t),f(Y,e,t),i(e,ve,t),i(e,S,t),f(A,S,null),d(S,Ze),d(S,G),f(O,G,null),d(G,We),d(G,ie),d(G,Ue),f(j,G,null),d(G,Be),f(Z,G,null),i(e,Te,t),f(Q,e,t),i(e,we,t),i(e,de,t),ke=!0},p(e,[t]){const $={};t&2&&($.$$scope={dirty:t,ctx:e}),I.$set($);const w={};t&2&&(w.$$scope={dirty:t,ctx:e}),F.$set(w);const J={};t&2&&(J.$$scope={dirty:t,ctx:e}),j.$set(J);const D={};t&2&&(D.$$scope={dirty:t,ctx:e}),Z.$set(D)},i(e){ke||(g(U.$$.fragment,e),g(B.$$.fragment,e),g(q.$$.fragment,e),g(R.$$.fragment,e),g(X.$$.fragment,e),g(I.$$.fragment,e),g(V.$$.fragment,e),g(E.$$.fragment,e),g(P.$$.fragment,e),g(F.$$.fragment,e),g(Y.$$.fragment,e),g(A.$$.fragment,e),g(O.$$.fragment,e),g(j.$$.fragment,e),g(Z.$$.fragment,e),g(Q.$$.fragment,e),ke=!0)},o(e){_(U.$$.fragment,e),_(B.$$.fragment,e),_(q.$$.fragment,e),_(R.$$.fragment,e),_(X.$$.fragment,e),_(I.$$.fragment,e),_(V.$$.fragment,e),_(E.$$.fragment,e),_(P.$$.fragment,e),_(F.$$.fragment,e),_(Y.$$.fragment,e),_(A.$$.fragment,e),_(O.$$.fragment,e),_(j.$$.fragment,e),_(Z.$$.fragment,e),_(Q.$$.fragment,e),ke=!1},d(e){e&&(n(p),n(a),n(h),n(k),n(z),n(pe),n(he),n(L),n(me),n(N),n(ue),n(fe),n(H),n(ge),n(_e),n(T),n(Me),n(be),n(y),n(ye),n(ve),n(S),n(Te),n(we),n(de)),n(o),M(U,e),M(B,e),M(q,e),M(R,e),M(X),M(I),M(V,e),M(E),M(P),M(F),M(Y,e),M(A),M(O),M(j),M(Z),M(Q,e)}}}const lt='{"title":"GraniteMoeShared","local":"granitemoeshared","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"GraniteMoeSharedConfig","local":"transformers.GraniteMoeSharedConfig","sections":[],"depth":2},{"title":"GraniteMoeSharedModel","local":"transformers.GraniteMoeSharedModel","sections":[],"depth":2},{"title":"GraniteMoeSharedForCausalLM","local":"transformers.GraniteMoeSharedForCausalLM","sections":[],"depth":2}],"depth":1}';function ct(x){return Ke(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Mt extends et{constructor(o){super(),tt(this,o,ct,dt,De,{})}}export{Mt as component};
