import{s as Fc,o as jc,n as Ee}from"../chunks/scheduler.18a86fab.js";import{S as Wc,i as Jc,g as r,s as n,r as l,A as Uc,h as s,f as i,c as o,j as x,u as p,x as a,k as y,y as e,a as k,v as m,d as h,t as u,w as f}from"../chunks/index.98837b22.js";import{T as Ic}from"../chunks/Tip.77304350.js";import{D as z}from"../chunks/Docstring.a1ef7999.js";import{C as Xt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as nr}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as rr,E as Nc}from"../chunks/getInferenceSnippets.06c2775f.js";function Bc(J){let c,I="Examples:",b,_,P;return _=new Xt({props:{code:"JTIzJTIwTGV0J3MlMjBzZWUlMjBob3clMjB0byUyMGluY3JlYXNlJTIwdGhlJTIwdm9jYWJ1bGFyeSUyMG9mJTIwQmVydCUyMG1vZGVsJTIwYW5kJTIwdG9rZW5pemVyJTBBdG9rZW5pemVyJTIwJTNEJTIwQmVydFRva2VuaXplckZhc3QuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0TW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBbnVtX2FkZGVkX3Rva3MlMjAlM0QlMjB0b2tlbml6ZXIuYWRkX3Rva2VucyglNUIlMjJuZXdfdG9rMSUyMiUyQyUyMCUyMm15X25ldy10b2syJTIyJTVEKSUwQXByaW50KCUyMldlJTIwaGF2ZSUyMGFkZGVkJTIyJTJDJTIwbnVtX2FkZGVkX3Rva3MlMkMlMjAlMjJ0b2tlbnMlMjIpJTBBJTIzJTIwTm90aWNlJTNBJTIwcmVzaXplX3Rva2VuX2VtYmVkZGluZ3MlMjBleHBlY3QlMjB0byUyMHJlY2VpdmUlMjB0aGUlMjBmdWxsJTIwc2l6ZSUyMG9mJTIwdGhlJTIwbmV3JTIwdm9jYWJ1bGFyeSUyQyUyMGkuZS4lMkMlMjB0aGUlMjBsZW5ndGglMjBvZiUyMHRoZSUyMHRva2VuaXplci4lMEFtb2RlbC5yZXNpemVfdG9rZW5fZW1iZWRkaW5ncyhsZW4odG9rZW5pemVyKSk=",highlighted:`<span class="hljs-comment"># Let&#x27;s see how to increase the vocabulary of Bert model and tokenizer</span>
tokenizer = BertTokenizerFast.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = BertModel.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

num_added_toks = tokenizer.add_tokens([<span class="hljs-string">&quot;new_tok1&quot;</span>, <span class="hljs-string">&quot;my_new-tok2&quot;</span>])
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;We have added&quot;</span>, num_added_toks, <span class="hljs-string">&quot;tokens&quot;</span>)
<span class="hljs-comment"># Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.</span>
model.resize_token_embeddings(<span class="hljs-built_in">len</span>(tokenizer))`,wrap:!1}}),{c(){c=r("p"),c.textContent=I,b=n(),l(_.$$.fragment)},l(d){c=s(d,"P",{"data-svelte-h":!0}),a(c)!=="svelte-kvfsh7"&&(c.textContent=I),b=o(d),p(_.$$.fragment,d)},m(d,C){k(d,c,C),k(d,b,C),m(_,d,C),P=!0},p:Ee,i(d){P||(h(_.$$.fragment,d),P=!0)},o(d){u(_.$$.fragment,d),P=!1},d(d){d&&(i(c),i(b)),f(_,d)}}}function Vc(J){let c,I="Examples:",b,_,P;return _=new Xt({props:{code:"JTIzJTIwTGV0J3MlMjBzZWUlMjBob3clMjB0byUyMGFkZCUyMGElMjBuZXclMjBjbGFzc2lmaWNhdGlvbiUyMHRva2VuJTIwdG8lMjBHUFQtMiUwQXRva2VuaXplciUyMCUzRCUyMEdQVDJUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haS1jb21tdW5pdHklMkZncHQyJTIyKSUwQW1vZGVsJTIwJTNEJTIwR1BUMk1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMiklMEElMEFzcGVjaWFsX3Rva2Vuc19kaWN0JTIwJTNEJTIwJTdCJTIyY2xzX3Rva2VuJTIyJTNBJTIwJTIyJTNDQ0xTJTNFJTIyJTdEJTBBJTBBbnVtX2FkZGVkX3Rva3MlMjAlM0QlMjB0b2tlbml6ZXIuYWRkX3NwZWNpYWxfdG9rZW5zKHNwZWNpYWxfdG9rZW5zX2RpY3QpJTBBcHJpbnQoJTIyV2UlMjBoYXZlJTIwYWRkZWQlMjIlMkMlMjBudW1fYWRkZWRfdG9rcyUyQyUyMCUyMnRva2VucyUyMiklMEElMjMlMjBOb3RpY2UlM0ElMjByZXNpemVfdG9rZW5fZW1iZWRkaW5ncyUyMGV4cGVjdCUyMHRvJTIwcmVjZWl2ZSUyMHRoZSUyMGZ1bGwlMjBzaXplJTIwb2YlMjB0aGUlMjBuZXclMjB2b2NhYnVsYXJ5JTJDJTIwaS5lLiUyQyUyMHRoZSUyMGxlbmd0aCUyMG9mJTIwdGhlJTIwdG9rZW5pemVyLiUwQW1vZGVsLnJlc2l6ZV90b2tlbl9lbWJlZGRpbmdzKGxlbih0b2tlbml6ZXIpKSUwQSUwQWFzc2VydCUyMHRva2VuaXplci5jbHNfdG9rZW4lMjAlM0QlM0QlMjAlMjIlM0NDTFMlM0UlMjI=",highlighted:`<span class="hljs-comment"># Let&#x27;s see how to add a new classification token to GPT-2</span>
tokenizer = GPT2Tokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
model = GPT2Model.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

special_tokens_dict = {<span class="hljs-string">&quot;cls_token&quot;</span>: <span class="hljs-string">&quot;&lt;CLS&gt;&quot;</span>}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;We have added&quot;</span>, num_added_toks, <span class="hljs-string">&quot;tokens&quot;</span>)
<span class="hljs-comment"># Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.</span>
model.resize_token_embeddings(<span class="hljs-built_in">len</span>(tokenizer))

<span class="hljs-keyword">assert</span> tokenizer.cls_token == <span class="hljs-string">&quot;&lt;CLS&gt;&quot;</span>`,wrap:!1}}),{c(){c=r("p"),c.textContent=I,b=n(),l(_.$$.fragment)},l(d){c=s(d,"P",{"data-svelte-h":!0}),a(c)!=="svelte-kvfsh7"&&(c.textContent=I),b=o(d),p(_.$$.fragment,d)},m(d,C){k(d,c,C),k(d,b,C),m(_,d,C),P=!0},p:Ee,i(d){P||(h(_.$$.fragment,d),P=!0)},o(d){u(_.$$.fragment,d),P=!1},d(d){d&&(i(c),i(b)),f(_,d)}}}function Lc(J){let c,I="Examples:",b,_,P;return _=new Xt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS1jYXNlZCUyMiklMEElMEElMjMlMjBQdXNoJTIwdGhlJTIwdG9rZW5pemVyJTIwdG8lMjB5b3VyJTIwbmFtZXNwYWNlJTIwd2l0aCUyMHRoZSUyMG5hbWUlMjAlMjJteS1maW5ldHVuZWQtYmVydCUyMi4lMEF0b2tlbml6ZXIucHVzaF90b19odWIoJTIybXktZmluZXR1bmVkLWJlcnQlMjIpJTBBJTBBJTIzJTIwUHVzaCUyMHRoZSUyMHRva2VuaXplciUyMHRvJTIwYW4lMjBvcmdhbml6YXRpb24lMjB3aXRoJTIwdGhlJTIwbmFtZSUyMCUyMm15LWZpbmV0dW5lZC1iZXJ0JTIyLiUwQXRva2VuaXplci5wdXNoX3RvX2h1YiglMjJodWdnaW5nZmFjZSUyRm15LWZpbmV0dW5lZC1iZXJ0JTIyKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-cased&quot;</span>)

<span class="hljs-comment"># Push the tokenizer to your namespace with the name &quot;my-finetuned-bert&quot;.</span>
tokenizer.push_to_hub(<span class="hljs-string">&quot;my-finetuned-bert&quot;</span>)

<span class="hljs-comment"># Push the tokenizer to an organization with the name &quot;my-finetuned-bert&quot;.</span>
tokenizer.push_to_hub(<span class="hljs-string">&quot;huggingface/my-finetuned-bert&quot;</span>)`,wrap:!1}}),{c(){c=r("p"),c.textContent=I,b=n(),l(_.$$.fragment)},l(d){c=s(d,"P",{"data-svelte-h":!0}),a(c)!=="svelte-kvfsh7"&&(c.textContent=I),b=o(d),p(_.$$.fragment,d)},m(d,C){k(d,c,C),k(d,b,C),m(_,d,C),P=!0},p:Ee,i(d){P||(h(_.$$.fragment,d),P=!0)},o(d){u(_.$$.fragment,d),P=!1},d(d){d&&(i(c),i(b)),f(_,d)}}}function Ec(J){let c,I=`This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.`;return{c(){c=r("p"),c.textContent=I},l(b){c=s(b,"P",{"data-svelte-h":!0}),a(c)!=="svelte-1yi8eve"&&(c.textContent=I)},m(b,_){k(b,c,_)},p:Ee,d(b){b&&i(c)}}}function Zc(J){let c,I="Examples:",b,_,P;return _=new Xt({props:{code:"JTIzJTIwTGV0J3MlMjBzZWUlMjBob3clMjB0byUyMGluY3JlYXNlJTIwdGhlJTIwdm9jYWJ1bGFyeSUyMG9mJTIwQmVydCUyMG1vZGVsJTIwYW5kJTIwdG9rZW5pemVyJTBBdG9rZW5pemVyJTIwJTNEJTIwQmVydFRva2VuaXplckZhc3QuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0TW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBbnVtX2FkZGVkX3Rva3MlMjAlM0QlMjB0b2tlbml6ZXIuYWRkX3Rva2VucyglNUIlMjJuZXdfdG9rMSUyMiUyQyUyMCUyMm15X25ldy10b2syJTIyJTVEKSUwQXByaW50KCUyMldlJTIwaGF2ZSUyMGFkZGVkJTIyJTJDJTIwbnVtX2FkZGVkX3Rva3MlMkMlMjAlMjJ0b2tlbnMlMjIpJTBBJTIzJTIwTm90aWNlJTNBJTIwcmVzaXplX3Rva2VuX2VtYmVkZGluZ3MlMjBleHBlY3QlMjB0byUyMHJlY2VpdmUlMjB0aGUlMjBmdWxsJTIwc2l6ZSUyMG9mJTIwdGhlJTIwbmV3JTIwdm9jYWJ1bGFyeSUyQyUyMGkuZS4lMkMlMjB0aGUlMjBsZW5ndGglMjBvZiUyMHRoZSUyMHRva2VuaXplci4lMEFtb2RlbC5yZXNpemVfdG9rZW5fZW1iZWRkaW5ncyhsZW4odG9rZW5pemVyKSk=",highlighted:`<span class="hljs-comment"># Let&#x27;s see how to increase the vocabulary of Bert model and tokenizer</span>
tokenizer = BertTokenizerFast.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
model = BertModel.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

num_added_toks = tokenizer.add_tokens([<span class="hljs-string">&quot;new_tok1&quot;</span>, <span class="hljs-string">&quot;my_new-tok2&quot;</span>])
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;We have added&quot;</span>, num_added_toks, <span class="hljs-string">&quot;tokens&quot;</span>)
<span class="hljs-comment"># Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.</span>
model.resize_token_embeddings(<span class="hljs-built_in">len</span>(tokenizer))`,wrap:!1}}),{c(){c=r("p"),c.textContent=I,b=n(),l(_.$$.fragment)},l(d){c=s(d,"P",{"data-svelte-h":!0}),a(c)!=="svelte-kvfsh7"&&(c.textContent=I),b=o(d),p(_.$$.fragment,d)},m(d,C){k(d,c,C),k(d,b,C),m(_,d,C),P=!0},p:Ee,i(d){P||(h(_.$$.fragment,d),P=!0)},o(d){u(_.$$.fragment,d),P=!1},d(d){d&&(i(c),i(b)),f(_,d)}}}function Dc(J){let c,I="Examples:",b,_,P;return _=new Xt({props:{code:"JTIzJTIwTGV0J3MlMjBzZWUlMjBob3clMjB0byUyMGFkZCUyMGElMjBuZXclMjBjbGFzc2lmaWNhdGlvbiUyMHRva2VuJTIwdG8lMjBHUFQtMiUwQXRva2VuaXplciUyMCUzRCUyMEdQVDJUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm9wZW5haS1jb21tdW5pdHklMkZncHQyJTIyKSUwQW1vZGVsJTIwJTNEJTIwR1BUMk1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJvcGVuYWktY29tbXVuaXR5JTJGZ3B0MiUyMiklMEElMEFzcGVjaWFsX3Rva2Vuc19kaWN0JTIwJTNEJTIwJTdCJTIyY2xzX3Rva2VuJTIyJTNBJTIwJTIyJTNDQ0xTJTNFJTIyJTdEJTBBJTBBbnVtX2FkZGVkX3Rva3MlMjAlM0QlMjB0b2tlbml6ZXIuYWRkX3NwZWNpYWxfdG9rZW5zKHNwZWNpYWxfdG9rZW5zX2RpY3QpJTBBcHJpbnQoJTIyV2UlMjBoYXZlJTIwYWRkZWQlMjIlMkMlMjBudW1fYWRkZWRfdG9rcyUyQyUyMCUyMnRva2VucyUyMiklMEElMjMlMjBOb3RpY2UlM0ElMjByZXNpemVfdG9rZW5fZW1iZWRkaW5ncyUyMGV4cGVjdCUyMHRvJTIwcmVjZWl2ZSUyMHRoZSUyMGZ1bGwlMjBzaXplJTIwb2YlMjB0aGUlMjBuZXclMjB2b2NhYnVsYXJ5JTJDJTIwaS5lLiUyQyUyMHRoZSUyMGxlbmd0aCUyMG9mJTIwdGhlJTIwdG9rZW5pemVyLiUwQW1vZGVsLnJlc2l6ZV90b2tlbl9lbWJlZGRpbmdzKGxlbih0b2tlbml6ZXIpKSUwQSUwQWFzc2VydCUyMHRva2VuaXplci5jbHNfdG9rZW4lMjAlM0QlM0QlMjAlMjIlM0NDTFMlM0UlMjI=",highlighted:`<span class="hljs-comment"># Let&#x27;s see how to add a new classification token to GPT-2</span>
tokenizer = GPT2Tokenizer.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)
model = GPT2Model.from_pretrained(<span class="hljs-string">&quot;openai-community/gpt2&quot;</span>)

special_tokens_dict = {<span class="hljs-string">&quot;cls_token&quot;</span>: <span class="hljs-string">&quot;&lt;CLS&gt;&quot;</span>}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
<span class="hljs-built_in">print</span>(<span class="hljs-string">&quot;We have added&quot;</span>, num_added_toks, <span class="hljs-string">&quot;tokens&quot;</span>)
<span class="hljs-comment"># Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.</span>
model.resize_token_embeddings(<span class="hljs-built_in">len</span>(tokenizer))

<span class="hljs-keyword">assert</span> tokenizer.cls_token == <span class="hljs-string">&quot;&lt;CLS&gt;&quot;</span>`,wrap:!1}}),{c(){c=r("p"),c.textContent=I,b=n(),l(_.$$.fragment)},l(d){c=s(d,"P",{"data-svelte-h":!0}),a(c)!=="svelte-kvfsh7"&&(c.textContent=I),b=o(d),p(_.$$.fragment,d)},m(d,C){k(d,c,C),k(d,b,C),m(_,d,C),P=!0},p:Ee,i(d){P||(h(_.$$.fragment,d),P=!0)},o(d){u(_.$$.fragment,d),P=!1},d(d){d&&(i(c),i(b)),f(_,d)}}}function Hc(J){let c,I="Examples:",b,_,P;return _=new Xt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS1jYXNlZCUyMiklMEElMEElMjMlMjBQdXNoJTIwdGhlJTIwdG9rZW5pemVyJTIwdG8lMjB5b3VyJTIwbmFtZXNwYWNlJTIwd2l0aCUyMHRoZSUyMG5hbWUlMjAlMjJteS1maW5ldHVuZWQtYmVydCUyMi4lMEF0b2tlbml6ZXIucHVzaF90b19odWIoJTIybXktZmluZXR1bmVkLWJlcnQlMjIpJTBBJTBBJTIzJTIwUHVzaCUyMHRoZSUyMHRva2VuaXplciUyMHRvJTIwYW4lMjBvcmdhbml6YXRpb24lMjB3aXRoJTIwdGhlJTIwbmFtZSUyMCUyMm15LWZpbmV0dW5lZC1iZXJ0JTIyLiUwQXRva2VuaXplci5wdXNoX3RvX2h1YiglMjJodWdnaW5nZmFjZSUyRm15LWZpbmV0dW5lZC1iZXJ0JTIyKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-cased&quot;</span>)

<span class="hljs-comment"># Push the tokenizer to your namespace with the name &quot;my-finetuned-bert&quot;.</span>
tokenizer.push_to_hub(<span class="hljs-string">&quot;my-finetuned-bert&quot;</span>)

<span class="hljs-comment"># Push the tokenizer to an organization with the name &quot;my-finetuned-bert&quot;.</span>
tokenizer.push_to_hub(<span class="hljs-string">&quot;huggingface/my-finetuned-bert&quot;</span>)`,wrap:!1}}),{c(){c=r("p"),c.textContent=I,b=n(),l(_.$$.fragment)},l(d){c=s(d,"P",{"data-svelte-h":!0}),a(c)!=="svelte-kvfsh7"&&(c.textContent=I),b=o(d),p(_.$$.fragment,d)},m(d,C){k(d,c,C),k(d,b,C),m(_,d,C),P=!0},p:Ee,i(d){P||(h(_.$$.fragment,d),P=!0)},o(d){u(_.$$.fragment,d),P=!1},d(d){d&&(i(c),i(b)),f(_,d)}}}function Sc(J){let c,I=`This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
this inside your training loop.`;return{c(){c=r("p"),c.textContent=I},l(b){c=s(b,"P",{"data-svelte-h":!0}),a(c)!=="svelte-1yi8eve"&&(c.textContent=I)},m(b,_){k(b,c,_)},p:Ee,d(b){b&&i(c)}}}function Ac(J){let c,I,b,_,P,d,C,Mi=`A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most
of the tokenizers are available in two flavors: a full python implementation and a “Fast” implementation based on the
Rust library <a href="https://github.com/huggingface/tokenizers" rel="nofollow">🤗 Tokenizers</a>. The “Fast” implementations allows:`,sr,Ze,qi=`<li>a significant speed-up in particular when doing batched tokenization and</li> <li>additional methods to map between the original string (character and words) and the token space (e.g. getting the
index of the token comprising a given character or the span of characters corresponding to a given token).</li>`,ar,De,Ci=`The base classes <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>
implement the common methods for encoding string inputs in model inputs (see below) and instantiating/saving python and
“Fast” tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library
(downloaded from HuggingFace’s AWS S3 repository). They both rely on
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase">PreTrainedTokenizerBase</a> that contains the common methods, and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.SpecialTokensMixin">SpecialTokensMixin</a>.`,ir,He,Ii=`<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> thus implement the main
methods for using all the tokenizers:`,dr,Se,Fi=`<li>Tokenizing (splitting strings in sub-word token strings), converting tokens strings to ids and back, and
encoding/decoding (i.e., tokenizing and converting to integers).</li> <li>Adding new tokens to the vocabulary in a way that is independent of the underlying structure (BPE, SentencePiece…).</li> <li>Managing special tokens (like mask, beginning-of-sentence, etc.): adding them, assigning them to attributes in the
tokenizer for easy access and making sure they are not split during tokenization.</li>`,cr,Ae,ji=`<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a> holds the output of the
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase">PreTrainedTokenizerBase</a>’s encoding methods (<code>__call__</code>,
<code>encode_plus</code> and <code>batch_encode_plus</code>) and is derived from a Python dictionary. When the tokenizer is a pure python
tokenizer, this class behaves just like a standard python dictionary and holds the various model inputs computed by
these methods (<code>input_ids</code>, <code>attention_mask</code>…). When the tokenizer is a “Fast” tokenizer (i.e., backed by
HuggingFace <a href="https://github.com/huggingface/tokenizers" rel="nofollow">tokenizers library</a>), this class provides in addition
several advanced alignment methods which can be used to map between the original string (character and words) and the
token space (e.g., getting the index of the token comprising a given character or the span of characters corresponding
to a given token).`,lr,Ge,pr,Re,Wi=`Apart from that each tokenizer can be a “multimodal” tokenizer which means that the tokenizer will hold all relevant special tokens
as part of tokenizer attributes for easier access. For example, if the tokenizer is loaded from a vision-language model like LLaVA, you will
be able to access <code>tokenizer.image_token_id</code> to obtain the special image token used as a placeholder.`,mr,Xe,Ji=`To enable extra special tokens for any type of tokenizer, you have to add the following lines and save the tokenizer. Extra special tokens do not
have to be modality related and can ne anything that the model often needs access to. In the below code, tokenizer at <code>output_dir</code> will have direct access
to three more special tokens.`,hr,Ye,ur,Oe,fr,T,Qe,Vr,Yt,Ui="Base class for all slow tokenizers.",Lr,Ot,Ni='Inherits from <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase">PreTrainedTokenizerBase</a>.',Er,Qt,Bi=`Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
pretrained tokenizers as well as adding tokens to the vocabulary.`,Zr,Kt,Vi=`This class also contain the added tokens in a unified way on top of all tokenizers so we don’t have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).`,Dr,en,Li="Class attributes (overridden by derived classes)",Hr,tn,Ei=`<li><strong>vocab_files_names</strong> (<code>dict[str, str]</code>) — A dictionary with, as keys, the <code>__init__</code> keyword name of each
vocabulary file required by the model, and as associated values, the filename for saving the associated file
(string).</li> <li><strong>pretrained_vocab_files_map</strong> (<code>dict[str, dict[str, str]]</code>) — A dictionary of dictionaries, with the
high-level keys being the <code>__init__</code> keyword name of each vocabulary file required by the model, the
low-level being the <code>short-cut-names</code> of the pretrained models with, as associated values, the <code>url</code> to the
associated pretrained vocabulary file.</li> <li><strong>model_input_names</strong> (<code>list[str]</code>) — A list of inputs expected in the forward pass of the model.</li> <li><strong>padding_side</strong> (<code>str</code>) — The default value for the side on which the model should have padding applied.
Should be <code>&#39;right&#39;</code> or <code>&#39;left&#39;</code>.</li> <li><strong>truncation_side</strong> (<code>str</code>) — The default value for the side on which the model should have truncation
applied. Should be <code>&#39;right&#39;</code> or <code>&#39;left&#39;</code>.</li>`,Sr,fe,Ke,Ar,nn,Zi=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`,Gr,L,et,Rr,on,Di=`Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
it with indices starting from length of the current vocabulary and will be isolated before the tokenization
algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
not treated in the same way.`,Xr,rn,Hi=`Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
of the model so that its embedding matrix matches the tokenizer.`,Yr,sn,Si='In order to do that, please use the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings">resize_token_embeddings()</a> method.',Or,ge,Qr,F,tt,Kr,an,Ai=`Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
current vocabulary).`,es,dn,Gi=`When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
model so that its embedding matrix matches the tokenizer.`,ts,cn,Ri='In order to do that, please use the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings">resize_token_embeddings()</a> method.',ns,ln,Xi="Using <code>add_special_tokens</code> will ensure your special tokens can be used in several ways:",os,pn,Yi=`<li>Special tokens can be skipped when decoding using <code>skip_special_tokens = True</code>.</li> <li>Special tokens are carefully handled by the tokenizer (they are never split), similar to <code>AddedTokens</code>.</li> <li>You can easily refer to special tokens using tokenizer class attributes like <code>tokenizer.cls_token</code>. This
makes it easy to develop model-agnostic training and fine-tuning scripts.</li>`,rs,mn,Oi=`When possible, special tokens are already registered for provided pretrained models (for instance
<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a> <code>cls_token</code> is already registered to be <code>&#39;[CLS]&#39;</code> and XLM’s one is also registered to be
<code>&#39;&lt;/s&gt;&#39;</code>).`,ss,_e,as,ke,nt,is,hn,Qi=`Converts a list of dictionaries with <code>&quot;role&quot;</code> and <code>&quot;content&quot;</code> keys to a list of token
ids. This method is intended for use with chat models, and will read the tokenizer’s chat_template attribute to
determine the format and control tokens to use when converting.`,ds,be,ot,cs,un,Ki="Convert a list of lists of token ids into a list of strings by calling decode.",ls,Y,rt,ps,fn,ed=`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`,ms,gn,td="Similar to doing <code>self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))</code>.",hs,O,st,us,_n,nd="Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.",fs,kn,od="Same as doing <code>self.convert_tokens_to_ids(self.tokenize(text))</code>.",gs,Q,at,_s,bn,rd="Upload the tokenizer files to the 🤗 Model Hub.",ks,Te,bs,ve,it,Ts,Tn,sd=`Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
added tokens.`,vs,xe,dt,xs,vn,ad=`Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
vocabulary.`,ys,ye,ct,ws,xn,id=`Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
something we should change.`,zs,K,lt,$s,yn,dd="Returns the number of added tokens when encoding a sequence with special tokens.",Ps,we,Ms,ee,pt,qs,wn,cd="Performs any necessary transformations before tokenization.",Cs,zn,ld=`This method should pop the arguments from kwargs and return the remaining <code>kwargs</code> as well. We test the
<code>kwargs</code> at the end of the encoding process to be sure all the arguments have been used.`,Is,te,mt,Fs,$n,pd="Converts a string into a sequence of tokens, using the tokenizer.",js,Pn,md=`Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
(BPE/SentencePieces/WordPieces). Takes care of added tokens.`,gr,ht,_r,ut,hd=`The <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> depend on the <a href="https://huggingface.co/docs/tokenizers" rel="nofollow">tokenizers</a> library. The tokenizers obtained from the 🤗 tokenizers library can be
loaded very simply into 🤗 transformers. Take a look at the <a href="../fast_tokenizers">Using tokenizers from 🤗 tokenizers</a> page to understand how this is done.`,kr,v,ft,Ws,Mn,ud="Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).",Js,qn,fd='Inherits from <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase">PreTrainedTokenizerBase</a>.',Us,Cn,gd=`Handles all the shared methods for tokenization and special tokens, as well as methods for
downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.`,Ns,In,_d=`This class also contains the added tokens in a unified way on top of all tokenizers so we don’t have to handle the
specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece…).`,Bs,Fn,kd="Class attributes (overridden by derived classes)",Vs,jn,bd=`<li><strong>vocab_files_names</strong> (<code>dict[str, str]</code>) — A dictionary with, as keys, the <code>__init__</code> keyword name of each
vocabulary file required by the model, and as associated values, the filename for saving the associated file
(string).</li> <li><strong>pretrained_vocab_files_map</strong> (<code>dict[str, dict[str, str]]</code>) — A dictionary of dictionaries, with the
high-level keys being the <code>__init__</code> keyword name of each vocabulary file required by the model, the
low-level being the <code>short-cut-names</code> of the pretrained models with, as associated values, the <code>url</code> to the
associated pretrained vocabulary file.</li> <li><strong>model_input_names</strong> (<code>list[str]</code>) — A list of inputs expected in the forward pass of the model.</li> <li><strong>padding_side</strong> (<code>str</code>) — The default value for the side on which the model should have padding applied.
Should be <code>&#39;right&#39;</code> or <code>&#39;left&#39;</code>.</li> <li><strong>truncation_side</strong> (<code>str</code>) — The default value for the side on which the model should have truncation
applied. Should be <code>&#39;right&#39;</code> or <code>&#39;left&#39;</code>.</li>`,Ls,ze,gt,Es,Wn,Td=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`,Zs,E,_t,Ds,Jn,vd=`Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
it with indices starting from length of the current vocabulary and will be isolated before the tokenization
algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
not treated in the same way.`,Hs,Un,xd=`Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
of the model so that its embedding matrix matches the tokenizer.`,Ss,Nn,yd='In order to do that, please use the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings">resize_token_embeddings()</a> method.',As,$e,Gs,j,kt,Rs,Bn,wd=`Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
current vocabulary).`,Xs,Vn,zd=`When adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the
model so that its embedding matrix matches the tokenizer.`,Ys,Ln,$d='In order to do that, please use the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings">resize_token_embeddings()</a> method.',Os,En,Pd="Using <code>add_special_tokens</code> will ensure your special tokens can be used in several ways:",Qs,Zn,Md=`<li>Special tokens can be skipped when decoding using <code>skip_special_tokens = True</code>.</li> <li>Special tokens are carefully handled by the tokenizer (they are never split), similar to <code>AddedTokens</code>.</li> <li>You can easily refer to special tokens using tokenizer class attributes like <code>tokenizer.cls_token</code>. This
makes it easy to develop model-agnostic training and fine-tuning scripts.</li>`,Ks,Dn,qd=`When possible, special tokens are already registered for provided pretrained models (for instance
<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a> <code>cls_token</code> is already registered to be <code>&#39;[CLS]&#39;</code> and XLM’s one is also registered to be
<code>&#39;&lt;/s&gt;&#39;</code>).`,ea,Pe,ta,Me,bt,na,Hn,Cd=`Converts a list of dictionaries with <code>&quot;role&quot;</code> and <code>&quot;content&quot;</code> keys to a list of token
ids. This method is intended for use with chat models, and will read the tokenizer’s chat_template attribute to
determine the format and control tokens to use when converting.`,oa,qe,Tt,ra,Sn,Id="Convert a list of lists of token ids into a list of strings by calling decode.",sa,ne,vt,aa,An,Fd=`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`,ia,Gn,jd="Similar to doing <code>self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))</code>.",da,oe,xt,ca,Rn,Wd="Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.",la,Xn,Jd="Same as doing <code>self.convert_tokens_to_ids(self.tokenize(text))</code>.",pa,re,yt,ma,Yn,Ud="Upload the tokenizer files to the 🤗 Model Hub.",ha,Ce,ua,Ie,wt,fa,On,Nd=`Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
added tokens.`,ga,Fe,zt,_a,Qn,Bd=`Converts a token string (or a sequence of tokens) in a single integer id (or a Iterable of ids), using the
vocabulary.`,ka,je,$t,ba,Kn,Vd="Returns the added tokens in the vocabulary as a dictionary of token to index.",Ta,se,Pt,va,eo,Ld="Returns the number of added tokens when encoding a sequence with special tokens.",xa,We,ya,ae,Mt,wa,to,Ed=`Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
library) and restore the tokenizer settings afterwards.`,za,no,Zd=`The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
section.`,$a,Je,qt,Pa,oo,Dd=`Trains a tokenizer on a new corpus with the same defaults (in terms of special tokens or tokenization pipeline)
as the current one.`,br,Ct,Tr,M,It,Ma,ro,Hd=`Holds the output of the <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__"><strong>call</strong>()</a>,
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus">encode_plus()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_encode_plus">batch_encode_plus()</a> methods (tokens, attention_masks, etc).`,qa,so,Sd=`This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
utility methods to map from word/character space to token space.`,Ca,Z,Ft,Ia,ao,Ad=`Get the index of the token in the encoded output comprising a character in the original string for a sequence
of the batch.`,Fa,io,Gd="Can be called as:",ja,co,Rd="<li><code>self.char_to_token(char_index)</code> if batch size is 1</li> <li><code>self.char_to_token(batch_index, char_index)</code> if batch size is greater or equal to 1</li>",Wa,lo,Xd=`This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.`,Ja,D,jt,Ua,po,Yd=`Get the word in the original string corresponding to a character in the original string of a sequence of the
batch.`,Na,mo,Od="Can be called as:",Ba,ho,Qd="<li><code>self.char_to_word(char_index)</code> if batch size is 1</li> <li><code>self.char_to_word(batch_index, char_index)</code> if batch size is greater than 1</li>",Va,uo,Kd=`This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.`,La,Ue,Wt,Ea,fo,ec="Convert the inner content to tensors.",Za,ie,Jt,Da,go,tc="Return a list mapping the tokens to the id of their original sentences:",Ha,_o,nc=`<li><code>None</code> for special tokens added around or between sequences,</li> <li><code>0</code> for tokens corresponding to words in the first sequence,</li> <li><code>1</code> for tokens corresponding to words in the second sequence when a pair of sequences was jointly
encoded.</li>`,Sa,Ne,Ut,Aa,ko,oc="Send all values to device by calling <code>v.to(device, non_blocking=non_blocking)</code> (PyTorch only).",Ga,N,Nt,Ra,bo,rc="Get the character span corresponding to an encoded token in a sequence of the batch.",Xa,To,sc='Character spans are returned as a <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.CharSpan">CharSpan</a> with:',Ya,vo,ac=`<li><strong>start</strong> — Index of the first character in the original string associated to the token.</li> <li><strong>end</strong> — Index of the character following the last character in the original string associated to the
token.</li>`,Oa,xo,ic="Can be called as:",Qa,yo,dc="<li><code>self.token_to_chars(token_index)</code> if batch size is 1</li> <li><code>self.token_to_chars(batch_index, token_index)</code> if batch size is greater or equal to 1</li>",Ka,H,Bt,ei,wo,cc=`Get the index of the sequence represented by the given token. In the general use case, this method returns <code>0</code>
for a single sequence or the first sequence of a pair, and <code>1</code> for the second sequence of a pair`,ti,zo,lc="Can be called as:",ni,$o,pc="<li><code>self.token_to_sequence(token_index)</code> if batch size is 1</li> <li><code>self.token_to_sequence(batch_index, token_index)</code> if batch size is greater than 1</li>",oi,Po,mc=`This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
words are defined by the user). In this case it allows to easily associate encoded tokens with provided
tokenized words.`,ri,S,Vt,si,Mo,hc="Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.",ai,qo,uc="Can be called as:",ii,Co,fc="<li><code>self.token_to_word(token_index)</code> if batch size is 1</li> <li><code>self.token_to_word(batch_index, token_index)</code> if batch size is greater than 1</li>",di,Io,gc=`This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
words are defined by the user). In this case it allows to easily associate encoded tokens with provided
tokenized words.`,ci,Be,Lt,li,Fo,_c=`Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
integer indices) at a given batch index (only works for the output of a fast tokenizer).`,pi,Ve,Et,mi,jo,kc="Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.",hi,B,Zt,ui,Wo,bc="Get the character span in the original string corresponding to given word in a sequence of the batch.",fi,Jo,Tc="Character spans are returned as a CharSpan NamedTuple with:",gi,Uo,vc="<li>start: index of the first character in the original string</li> <li>end: index of the character following the last character in the original string</li>",_i,No,xc="Can be called as:",ki,Bo,yc="<li><code>self.word_to_chars(word_index)</code> if batch size is 1</li> <li><code>self.word_to_chars(batch_index, word_index)</code> if batch size is greater or equal to 1</li>",bi,U,Dt,Ti,Vo,wc="Get the encoded token span corresponding to a word in a sequence of the batch.",vi,Lo,zc='Token spans are returned as a <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.TokenSpan">TokenSpan</a> with:',xi,Eo,$c="<li><strong>start</strong> — Index of the first token.</li> <li><strong>end</strong> — Index of the token following the last token.</li>",yi,Zo,Pc="Can be called as:",wi,Do,Mc=`<li><code>self.word_to_tokens(word_index, sequence_index: int = 0)</code> if batch size is 1</li> <li><code>self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)</code> if batch size is greater or equal to
1</li>`,zi,Ho,qc=`This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
words.`,$i,Le,Ht,Pi,So,Cc="Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.",vr,St,xr,or,yr;return P=new rr({props:{title:"Tokenizer",local:"tokenizer",headingTag:"h1"}}),Ge=new rr({props:{title:"Multimodal Tokenizer",local:"multimodal-tokenizer",headingTag:"h1"}}),Ye=new Xt({props:{code:"dmlzaW9uX3Rva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmxsYXZhLWhmJTJGbGxhdmEtMS41LTdiLWhmJTIyJTJDJTBBJTIwJTIwJTIwJTIwZXh0cmFfc3BlY2lhbF90b2tlbnMlM0QlN0IlMjJpbWFnZV90b2tlbiUyMiUzQSUyMCUyMiUzQ2ltYWdlJTNFJTIyJTJDJTIwJTIyYm9pX3Rva2VuJTIyJTNBJTIwJTIyJTNDaW1hZ2Vfc3RhcnQlM0UlMjIlMkMlMjAlMjJlb2lfdG9rZW4lMjIlM0ElMjAlMjIlM0NpbWFnZV9lbmQlM0UlMjIlN0QlMEEpJTBBcHJpbnQodmlzaW9uX3Rva2VuaXplci5pbWFnZV90b2tlbiUyQyUyMHZpc2lvbl90b2tlbml6ZXIuaW1hZ2VfdG9rZW5faWQpJTBBKCUyMiUzQ2ltYWdlJTNFJTIyJTJDJTIwMzIwMDAp",highlighted:`vision_tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;llava-hf/llava-1.5-7b-hf&quot;</span>,
    extra_special_tokens={<span class="hljs-string">&quot;image_token&quot;</span>: <span class="hljs-string">&quot;&lt;image&gt;&quot;</span>, <span class="hljs-string">&quot;boi_token&quot;</span>: <span class="hljs-string">&quot;&lt;image_start&gt;&quot;</span>, <span class="hljs-string">&quot;eoi_token&quot;</span>: <span class="hljs-string">&quot;&lt;image_end&gt;&quot;</span>}
)
<span class="hljs-built_in">print</span>(vision_tokenizer.image_token, vision_tokenizer.image_token_id)
(<span class="hljs-string">&quot;&lt;image&gt;&quot;</span>, <span class="hljs-number">32000</span>)`,wrap:!1}}),Oe=new rr({props:{title:"PreTrainedTokenizer",local:"transformers.PreTrainedTokenizer",headingTag:"h2"}}),Qe=new z({props:{name:"class transformers.PreTrainedTokenizer",anchor:"transformers.PreTrainedTokenizer",parameters:[{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.model_max_length",description:`<strong>model_max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
loaded with <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained">from_pretrained()</a>, this will be set to the
value stored for the associated model in <code>max_model_input_sizes</code> (see above). If no value is provided, will
default to VERY_LARGE_INTEGER (<code>int(1e30)</code>).`,name:"model_max_length"},{anchor:"transformers.PreTrainedTokenizer.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizer.truncation_side",description:`<strong>truncation_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have truncation applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"truncation_side"},{anchor:"transformers.PreTrainedTokenizer.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A Jinja template string that will be used to format lists of chat messages. See
<a href="https://huggingface.co/docs/transformers/chat_templating" rel="nofollow">https://huggingface.co/docs/transformers/chat_templating</a> for a full description.`,name:"chat_template"},{anchor:"transformers.PreTrainedTokenizer.model_input_names",description:`<strong>model_input_names</strong> (<code>list[string]</code>, <em>optional</em>) &#x2014;
The list of inputs accepted by the forward pass of the model (like <code>&quot;token_type_ids&quot;</code> or
<code>&quot;attention_mask&quot;</code>). Default value is picked from the class attribute of the same name.`,name:"model_input_names"},{anchor:"transformers.PreTrainedTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the beginning of a sentence. Will be associated to <code>self.bos_token</code> and
<code>self.bos_token_id</code>.`,name:"bos_token"},{anchor:"transformers.PreTrainedTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the end of a sentence. Will be associated to <code>self.eos_token</code> and
<code>self.eos_token_id</code>.`,name:"eos_token"},{anchor:"transformers.PreTrainedTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing an out-of-vocabulary token. Will be associated to <code>self.unk_token</code> and
<code>self.unk_token_id</code>.`,name:"unk_token"},{anchor:"transformers.PreTrainedTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token separating two different sentences in the same input (used by BERT for instance). Will be
associated to <code>self.sep_token</code> and <code>self.sep_token_id</code>.`,name:"sep_token"},{anchor:"transformers.PreTrainedTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation. Will be associated to <code>self.pad_token</code> and <code>self.pad_token_id</code>.`,name:"pad_token"},{anchor:"transformers.PreTrainedTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the class of the input (used by BERT for instance). Will be associated to
<code>self.cls_token</code> and <code>self.cls_token_id</code>.`,name:"cls_token"},{anchor:"transformers.PreTrainedTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT). Will be associated to <code>self.mask_token</code> and <code>self.mask_token_id</code>.`,name:"mask_token"},{anchor:"transformers.PreTrainedTokenizer.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (tuple or list of <code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A tuple or a list of additional special tokens. Add them here to ensure they are skipped when decoding with
<code>skip_special_tokens</code> is set to True. If they are not part of the vocabulary, they will be added at the end
of the vocabulary.`,name:"additional_special_tokens"},{anchor:"transformers.PreTrainedTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should cleanup the spaces that were added when splitting the input text during the
tokenization process.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizer.split_special_tokens",description:`<strong>split_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the special tokens should be split during the tokenization process. Passing will affect the
internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
<code>&lt;s&gt;</code> is the <code>bos_token</code>, then <code>tokenizer.tokenize(&quot;&lt;s&gt;&quot;) = [&apos;&lt;s&gt;</code>]. Otherwise, if
<code>split_special_tokens=True</code>, then <code>tokenizer.tokenize(&quot;&lt;s&gt;&quot;)</code> will be give <code>[&apos;&lt;&apos;,&apos;s&apos;, &apos;&gt;&apos;]</code>.`,name:"split_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils.py#L407"}}),Ke=new z({props:{name:"__call__",anchor:"transformers.PreTrainedTokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.PreTrainedTokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.PreTrainedTokenizer.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.PreTrainedTokenizer.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.PreTrainedTokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizer.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizer.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.PreTrainedTokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.PreTrainedTokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.PreTrainedTokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.PreTrainedTokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.PreTrainedTokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.PreTrainedTokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.PreTrainedTokenizer.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>token_type_ids</strong> — List of token type ids to be fed to a model (when <code>return_token_type_ids=True</code> or
if <em>“token_type_ids”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a></p>
</li>
<li>
<p><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>overflowing_tokens</strong> — List of overflowing tokens sequences (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>num_truncated_tokens</strong> — Number of tokens truncated (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>special_tokens_mask</strong> — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when <code>add_special_tokens=True</code> and <code>return_special_tokens_mask=True</code>).</p>
</li>
<li>
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>)</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),et=new z({props:{name:"add_tokens",anchor:"transformers.PreTrainedTokenizer.add_tokens",parameters:[{name:"new_tokens",val:": typing.Union[str, tokenizers.AddedToken, collections.abc.Sequence[typing.Union[str, tokenizers.AddedToken]]]"},{name:"special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.add_tokens.new_tokens",description:`<strong>new_tokens</strong> (<code>str</code>, <code>tokenizers.AddedToken</code> or a sequence of <em>str</em> or <code>tokenizers.AddedToken</code>) &#x2014;
Tokens are only added if they are not already in the vocabulary. <code>tokenizers.AddedToken</code> wraps a string
token to let you personalize its behavior: whether this token should only match against a single word,
whether this token should strip all potential whitespaces on the left side, whether this token should
strip all potential whitespaces on the right side, etc.`,name:"new_tokens"},{anchor:"transformers.PreTrainedTokenizer.add_tokens.special_tokens",description:`<strong>special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Can be used to specify if the token is a special token. This mostly change the normalization behavior
(special tokens like CLS or [MASK] are usually not lower-cased for instance).</p>
<p>See details for <code>tokenizers.AddedToken</code> in HuggingFace tokenizers library.`,name:"special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L994",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of tokens added to the vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),ge=new nr({props:{anchor:"transformers.PreTrainedTokenizer.add_tokens.example",$$slots:{default:[Bc]},$$scope:{ctx:J}}}),tt=new z({props:{name:"add_special_tokens",anchor:"transformers.PreTrainedTokenizer.add_special_tokens",parameters:[{name:"special_tokens_dict",val:": dict"},{name:"replace_additional_special_tokens",val:" = True"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.add_special_tokens.special_tokens_dict",description:`<strong>special_tokens_dict</strong> (dictionary <em>str</em> to <em>str</em>, <code>tokenizers.AddedToken</code>, or <code>Sequence[Union[str, AddedToken]]</code>) &#x2014;
Keys should be in the list of predefined special attributes: [<code>bos_token</code>, <code>eos_token</code>, <code>unk_token</code>,
<code>sep_token</code>, <code>pad_token</code>, <code>cls_token</code>, <code>mask_token</code>, <code>additional_special_tokens</code>].</p>
<p>Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
assign the index of the <code>unk_token</code> to them).`,name:"special_tokens_dict"},{anchor:"transformers.PreTrainedTokenizer.add_special_tokens.replace_additional_special_tokens",description:`<strong>replace_additional_special_tokens</strong> (<code>bool</code>, <em>optional</em>,, defaults to <code>True</code>) &#x2014;
If <code>True</code>, the existing list of additional special tokens will be replaced by the list provided in
<code>special_tokens_dict</code>. Otherwise, <code>self._special_tokens_map[&quot;additional_special_tokens&quot;]</code> is just extended. In the former
case, the tokens will NOT be removed from the tokenizer&#x2019;s full vocabulary - they are only being flagged
as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
<code>added_tokens_encoder</code> and <code>added_tokens_decoder</code>. This means that the previous
<code>additional_special_tokens</code> are still added tokens, and will not be split by the model.`,name:"replace_additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L890",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of tokens added to the vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),_e=new nr({props:{anchor:"transformers.PreTrainedTokenizer.add_special_tokens.example",$$slots:{default:[Vc]},$$scope:{ctx:J}}}),nt=new z({props:{name:"apply_chat_template",anchor:"transformers.PreTrainedTokenizer.apply_chat_template",parameters:[{name:"conversation",val:": typing.Union[list[dict[str, str]], list[list[dict[str, str]]]]"},{name:"tools",val:": typing.Optional[list[typing.Union[dict, typing.Callable]]] = None"},{name:"documents",val:": typing.Optional[list[dict[str, str]]] = None"},{name:"chat_template",val:": typing.Optional[str] = None"},{name:"add_generation_prompt",val:": bool = False"},{name:"continue_final_message",val:": bool = False"},{name:"tokenize",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": bool = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_dict",val:": bool = False"},{name:"return_assistant_tokens_mask",val:": bool = False"},{name:"tokenizer_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.conversation",description:`<strong>conversation</strong> (Union[list[dict[str, str]], list[list[dict[str, str]]]]) &#x2014; A list of dicts
with &#x201C;role&#x201D; and &#x201C;content&#x201D; keys, representing the chat history so far.`,name:"conversation"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.tools",description:`<strong>tools</strong> (<code>list[Union[Dict, Callable]]</code>, <em>optional</em>) &#x2014;
A list of tools (callable functions) that will be accessible to the model. If the template does not
support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
giving the name, description and argument types for the tool. See our
<a href="https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use" rel="nofollow">chat templating guide</a>
for more information.`,name:"tools"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.documents",description:`<strong>documents</strong> (<code>list[dict[str, str]]</code>, <em>optional</em>) &#x2014;
A list of dicts representing documents that will be accessible to the model if it is performing RAG
(retrieval-augmented generation). If the template does not support RAG, this argument will have no
effect. We recommend that each document should be a dict containing &#x201C;title&#x201D; and &#x201C;text&#x201D; keys. Please
see the RAG section of the <a href="https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG" rel="nofollow">chat templating guide</a>
for examples of passing documents with chat templates.`,name:"documents"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
argument, as the model&#x2019;s template will be used by default.`,name:"chat_template"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.add_generation_prompt",description:`<strong>add_generation_prompt</strong> (bool, <em>optional</em>) &#x2014;
If this is set, a prompt with the token(s) that indicate
the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
Note that this argument will be passed to the chat template, and so it must be supported in the
template for this argument to have any effect.`,name:"add_generation_prompt"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.continue_final_message",description:`<strong>continue_final_message</strong> (bool, <em>optional</em>) &#x2014;
If this is set, the chat will be formatted so that the final
message in the chat is open-ended, without any EOS tokens. The model will continue this message
rather than starting a new one. This allows you to &#x201C;prefill&#x201D; part of
the model&#x2019;s response for it. Cannot be used at the same time as <code>add_generation_prompt</code>.`,name:"continue_final_message"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.tokenize",description:`<strong>tokenize</strong> (<code>bool</code>, defaults to <code>True</code>) &#x2014;
Whether to tokenize the output. If <code>False</code>, the output will be a string.`,name:"tokenize"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.truncation",description:`<strong>truncation</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to truncate sequences at the maximum length. Has no effect if tokenize is <code>False</code>.`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is <code>False</code>. If
not specified, the tokenizer&#x2019;s <code>max_length</code> attribute will be used as a default.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Has no effect if tokenize is <code>False</code>. Acceptable
values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.Tensor</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to return a dictionary with named outputs. Has no effect if tokenize is <code>False</code>.`,name:"return_dict"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.tokenizer_kwargs",description:"<strong>tokenizer_kwargs</strong> (<code>dict[str -- Any]</code>, <em>optional</em>): Additional kwargs to pass to the tokenizer.",name:"tokenizer_kwargs"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.return_assistant_tokens_mask",description:`<strong>return_assistant_tokens_mask</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
the mask will contain 1. For user and system tokens, the mask will contain 0.
This functionality is only available for chat templates that support it via the <code>{% generation %}</code> keyword.`,name:"return_assistant_tokens_mask"},{anchor:"transformers.PreTrainedTokenizer.apply_chat_template.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; Additional kwargs to pass to the template renderer. Will be accessible by the chat template.",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1518",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of token ids representing the tokenized chat so far, including control tokens. This
output is ready to pass to the model, either directly or via methods like <code>generate()</code>. If <code>return_dict</code> is
set, will return a dict of tokenizer outputs instead.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Union[list[int], Dict]</code></p>
`}}),ot=new z({props:{name:"batch_decode",anchor:"transformers.PreTrainedTokenizer.batch_decode",parameters:[{name:"sequences",val:": typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.batch_decode.sequences",description:`<strong>sequences</strong> (<code>Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"sequences"},{anchor:"transformers.PreTrainedTokenizer.batch_decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.PreTrainedTokenizer.batch_decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizer.batch_decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3833",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of decoded sentences.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code></p>
`}}),rt=new z({props:{name:"decode",anchor:"transformers.PreTrainedTokenizer.decode",parameters:[{name:"token_ids",val:": typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.decode.token_ids",description:`<strong>token_ids</strong> (<code>Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"token_ids"},{anchor:"transformers.PreTrainedTokenizer.decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.PreTrainedTokenizer.decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizer.decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3867",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded sentence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),st=new z({props:{name:"encode",anchor:"transformers.PreTrainedTokenizer.encode",parameters:[{name:"text",val:": typing.Union[str, list[str], list[int]]"},{name:"text_pair",val:": typing.Union[str, list[str], list[int], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.encode.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code> or <code>list[int]</code>) &#x2014;
The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
<code>tokenize</code> method) or a list of integers (tokenized string ids using the <code>convert_tokens_to_ids</code>
method).`,name:"text"},{anchor:"transformers.PreTrainedTokenizer.encode.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code> or <code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
the <code>tokenize</code> method) or a list of integers (tokenized string ids using the <code>convert_tokens_to_ids</code>
method).`,name:"text_pair"},{anchor:"transformers.PreTrainedTokenizer.encode.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizer.encode.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizer.encode.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizer.encode.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizer.encode.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizer.encode.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizer.encode.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizer.encode.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizer.encode.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizer.encode.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; Passed along to the <code>.tokenize()</code> method.",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2667",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The tokenized ids of the text.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code>, <code>torch.Tensor</code>, <code>tf.Tensor</code> or <code>np.ndarray</code></p>
`}}),at=new z({props:{name:"push_to_hub",anchor:"transformers.PreTrainedTokenizer.push_to_hub",parameters:[{name:"repo_id",val:": str"},{name:"use_temp_dir",val:": typing.Optional[bool] = None"},{name:"commit_message",val:": typing.Optional[str] = None"},{name:"private",val:": typing.Optional[bool] = None"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"max_shard_size",val:": typing.Union[str, int, NoneType] = '5GB'"},{name:"create_pr",val:": bool = False"},{name:"safe_serialization",val:": bool = True"},{name:"revision",val:": typing.Optional[str] = None"},{name:"commit_description",val:": typing.Optional[str] = None"},{name:"tags",val:": typing.Optional[list[str]] = None"},{name:"**deprecated_kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.push_to_hub.repo_id",description:`<strong>repo_id</strong> (<code>str</code>) &#x2014;
The name of the repository you want to push your tokenizer to. It should contain your organization name
when pushing to a given organization.`,name:"repo_id"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.use_temp_dir",description:`<strong>use_temp_dir</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
Will default to <code>True</code> if there is no directory named like <code>repo_id</code>, <code>False</code> otherwise.`,name:"use_temp_dir"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.commit_message",description:`<strong>commit_message</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Message to commit while pushing. Will default to <code>&quot;Upload tokenizer&quot;</code>.`,name:"commit_message"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.private",description:`<strong>private</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to make the repo private. If <code>None</code> (default), the repo will be public unless the organization&#x2019;s default is private. This value is ignored if the repo already exists.`,name:"private"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.token",description:`<strong>token</strong> (<code>bool</code> or <code>str</code>, <em>optional</em>) &#x2014;
The token to use as HTTP bearer authorization for remote files. If <code>True</code>, will use the token generated
when running <code>hf auth login</code> (stored in <code>~/.huggingface</code>). Will default to <code>True</code> if <code>repo_url</code>
is not specified.`,name:"token"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.max_shard_size",description:`<strong>max_shard_size</strong> (<code>int</code> or <code>str</code>, <em>optional</em>, defaults to <code>&quot;5GB&quot;</code>) &#x2014;
Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
will then be each of size lower than this size. If expressed as a string, needs to be digits followed
by a unit (like <code>&quot;5MB&quot;</code>). We default it to <code>&quot;5GB&quot;</code> so that users can easily load models on free-tier
Google Colab instances without any CPU OOM issues.`,name:"max_shard_size"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.create_pr",description:`<strong>create_pr</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to create a PR with the uploaded files or directly commit.`,name:"create_pr"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.safe_serialization",description:`<strong>safe_serialization</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to convert the model weights in safetensors format for safer serialization.`,name:"safe_serialization"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.revision",description:`<strong>revision</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Branch to push the uploaded files to.`,name:"revision"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.commit_description",description:`<strong>commit_description</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The description of the commit that will be created`,name:"commit_description"},{anchor:"transformers.PreTrainedTokenizer.push_to_hub.tags",description:`<strong>tags</strong> (<code>list[str]</code>, <em>optional</em>) &#x2014;
List of tags to push on the Hub.`,name:"tags"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L847"}}),Te=new nr({props:{anchor:"transformers.PreTrainedTokenizer.push_to_hub.example",$$slots:{default:[Lc]},$$scope:{ctx:J}}}),it=new z({props:{name:"convert_ids_to_tokens",anchor:"transformers.PreTrainedTokenizer.convert_ids_to_tokens",parameters:[{name:"ids",val:": typing.Union[int, list[int]]"},{name:"skip_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.convert_ids_to_tokens.ids",description:`<strong>ids</strong> (<code>int</code> or <code>list[int]</code>) &#x2014;
The token id (or token ids) to convert to tokens.`,name:"ids"},{anchor:"transformers.PreTrainedTokenizer.convert_ids_to_tokens.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils.py#L1044",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded token(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code> or <code>list[str]</code></p>
`}}),dt=new z({props:{name:"convert_tokens_to_ids",anchor:"transformers.PreTrainedTokenizer.convert_tokens_to_ids",parameters:[{name:"tokens",val:": typing.Union[str, list[str]]"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.convert_tokens_to_ids.tokens",description:"<strong>tokens</strong> (<code>str</code> or <code>list[str]</code>) &#x2014; One or several token(s) to convert to token id(s).",name:"tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils.py#L710",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token id or list of token ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code> or <code>list[int]</code></p>
`}}),ct=new z({props:{name:"get_added_vocab",anchor:"transformers.PreTrainedTokenizer.get_added_vocab",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils.py#L487",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The added tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>dict[str, int]</code></p>
`}}),lt=new z({props:{name:"num_special_tokens_to_add",anchor:"transformers.PreTrainedTokenizer.num_special_tokens_to_add",parameters:[{name:"pair",val:": bool = False"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.num_special_tokens_to_add.pair",description:`<strong>pair</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the number of added tokens should be computed in the case of a sequence pair or a single
sequence.`,name:"pair"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils.py#L598",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of special tokens added to sequences.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),we=new Ic({props:{$$slots:{default:[Ec]},$$scope:{ctx:J}}}),pt=new z({props:{name:"prepare_for_tokenization",anchor:"transformers.PreTrainedTokenizer.prepare_for_tokenization",parameters:[{name:"text",val:": str"},{name:"is_split_into_words",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.prepare_for_tokenization.text",description:`<strong>text</strong> (<code>str</code>) &#x2014;
The text to prepare.`,name:"text"},{anchor:"transformers.PreTrainedTokenizer.prepare_for_tokenization.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizer.prepare_for_tokenization.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Keyword arguments to use for the tokenization.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils.py#L984",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The prepared text and the unused kwargs.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>tuple[str, dict[str, Any]]</code></p>
`}}),mt=new z({props:{name:"tokenize",anchor:"transformers.PreTrainedTokenizer.tokenize",parameters:[{name:"text",val:": str"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizer.tokenize.text",description:`<strong>text</strong> (<code>str</code>) &#x2014;
The sequence to be encoded.`,name:"text"},{anchor:"transformers.PreTrainedTokenizer.tokenize.*kwargs",description:`*<strong>*kwargs</strong> (additional keyword arguments) &#x2014;
Passed along to the model-specific <code>prepare_for_tokenization</code> preprocessing method.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils.py#L621",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code></p>
`}}),ht=new rr({props:{title:"PreTrainedTokenizerFast",local:"transformers.PreTrainedTokenizerFast",headingTag:"h2"}}),ft=new z({props:{name:"class transformers.PreTrainedTokenizerFast",anchor:"transformers.PreTrainedTokenizerFast",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.model_max_length",description:`<strong>model_max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
loaded with <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained">from_pretrained()</a>, this will be set to the
value stored for the associated model in <code>max_model_input_sizes</code> (see above). If no value is provided, will
default to VERY_LARGE_INTEGER (<code>int(1e30)</code>).`,name:"model_max_length"},{anchor:"transformers.PreTrainedTokenizerFast.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerFast.truncation_side",description:`<strong>truncation_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have truncation applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"truncation_side"},{anchor:"transformers.PreTrainedTokenizerFast.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A Jinja template string that will be used to format lists of chat messages. See
<a href="https://huggingface.co/docs/transformers/chat_templating" rel="nofollow">https://huggingface.co/docs/transformers/chat_templating</a> for a full description.`,name:"chat_template"},{anchor:"transformers.PreTrainedTokenizerFast.model_input_names",description:`<strong>model_input_names</strong> (<code>list[string]</code>, <em>optional</em>) &#x2014;
The list of inputs accepted by the forward pass of the model (like <code>&quot;token_type_ids&quot;</code> or
<code>&quot;attention_mask&quot;</code>). Default value is picked from the class attribute of the same name.`,name:"model_input_names"},{anchor:"transformers.PreTrainedTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the beginning of a sentence. Will be associated to <code>self.bos_token</code> and
<code>self.bos_token_id</code>.`,name:"bos_token"},{anchor:"transformers.PreTrainedTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the end of a sentence. Will be associated to <code>self.eos_token</code> and
<code>self.eos_token_id</code>.`,name:"eos_token"},{anchor:"transformers.PreTrainedTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing an out-of-vocabulary token. Will be associated to <code>self.unk_token</code> and
<code>self.unk_token_id</code>.`,name:"unk_token"},{anchor:"transformers.PreTrainedTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token separating two different sentences in the same input (used by BERT for instance). Will be
associated to <code>self.sep_token</code> and <code>self.sep_token_id</code>.`,name:"sep_token"},{anchor:"transformers.PreTrainedTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation. Will be associated to <code>self.pad_token</code> and <code>self.pad_token_id</code>.`,name:"pad_token"},{anchor:"transformers.PreTrainedTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing the class of the input (used by BERT for instance). Will be associated to
<code>self.cls_token</code> and <code>self.cls_token_id</code>.`,name:"cls_token"},{anchor:"transformers.PreTrainedTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token representing a masked token (used by masked-language modeling pretraining objectives, like
BERT). Will be associated to <code>self.mask_token</code> and <code>self.mask_token_id</code>.`,name:"mask_token"},{anchor:"transformers.PreTrainedTokenizerFast.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (tuple or list of <code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A tuple or a list of additional special tokens. Add them here to ensure they are skipped when decoding with
<code>skip_special_tokens</code> is set to True. If they are not part of the vocabulary, they will be added at the end
of the vocabulary.`,name:"additional_special_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should cleanup the spaces that were added when splitting the input text during the
tokenization process.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizerFast.split_special_tokens",description:`<strong>split_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the special tokens should be split during the tokenization process. Passing will affect the
internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
<code>&lt;s&gt;</code> is the <code>bos_token</code>, then <code>tokenizer.tokenize(&quot;&lt;s&gt;&quot;) = [&apos;&lt;s&gt;</code>]. Otherwise, if
<code>split_special_tokens=True</code>, then <code>tokenizer.tokenize(&quot;&lt;s&gt;&quot;)</code> will be give <code>[&apos;&lt;&apos;,&apos;s&apos;, &apos;&gt;&apos;]</code>.`,name:"split_special_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.tokenizer_object",description:`<strong>tokenizer_object</strong> (<code>tokenizers.Tokenizer</code>) &#x2014;
A <code>tokenizers.Tokenizer</code> object from &#x1F917; tokenizers to instantiate from. See <a href="../fast_tokenizers">Using tokenizers from &#x1F917;
tokenizers</a> for more information.`,name:"tokenizer_object"},{anchor:"transformers.PreTrainedTokenizerFast.tokenizer_file",description:`<strong>tokenizer_file</strong> (<code>str</code>) &#x2014;
A path to a local JSON file representing a previously serialized <code>tokenizers.Tokenizer</code> object from &#x1F917;
tokenizers.`,name:"tokenizer_file"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_fast.py#L82"}}),gt=new z({props:{name:"__call__",anchor:"transformers.PreTrainedTokenizerFast.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.PreTrainedTokenizerFast.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>token_type_ids</strong> — List of token type ids to be fed to a model (when <code>return_token_type_ids=True</code> or
if <em>“token_type_ids”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a></p>
</li>
<li>
<p><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>overflowing_tokens</strong> — List of overflowing tokens sequences (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>num_truncated_tokens</strong> — Number of tokens truncated (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>special_tokens_mask</strong> — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when <code>add_special_tokens=True</code> and <code>return_special_tokens_mask=True</code>).</p>
</li>
<li>
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>)</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),_t=new z({props:{name:"add_tokens",anchor:"transformers.PreTrainedTokenizerFast.add_tokens",parameters:[{name:"new_tokens",val:": typing.Union[str, tokenizers.AddedToken, collections.abc.Sequence[typing.Union[str, tokenizers.AddedToken]]]"},{name:"special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.add_tokens.new_tokens",description:`<strong>new_tokens</strong> (<code>str</code>, <code>tokenizers.AddedToken</code> or a sequence of <em>str</em> or <code>tokenizers.AddedToken</code>) &#x2014;
Tokens are only added if they are not already in the vocabulary. <code>tokenizers.AddedToken</code> wraps a string
token to let you personalize its behavior: whether this token should only match against a single word,
whether this token should strip all potential whitespaces on the left side, whether this token should
strip all potential whitespaces on the right side, etc.`,name:"new_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.add_tokens.special_tokens",description:`<strong>special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Can be used to specify if the token is a special token. This mostly change the normalization behavior
(special tokens like CLS or [MASK] are usually not lower-cased for instance).</p>
<p>See details for <code>tokenizers.AddedToken</code> in HuggingFace tokenizers library.`,name:"special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L994",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of tokens added to the vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),$e=new nr({props:{anchor:"transformers.PreTrainedTokenizerFast.add_tokens.example",$$slots:{default:[Zc]},$$scope:{ctx:J}}}),kt=new z({props:{name:"add_special_tokens",anchor:"transformers.PreTrainedTokenizerFast.add_special_tokens",parameters:[{name:"special_tokens_dict",val:": dict"},{name:"replace_additional_special_tokens",val:" = True"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.add_special_tokens.special_tokens_dict",description:`<strong>special_tokens_dict</strong> (dictionary <em>str</em> to <em>str</em>, <code>tokenizers.AddedToken</code>, or <code>Sequence[Union[str, AddedToken]]</code>) &#x2014;
Keys should be in the list of predefined special attributes: [<code>bos_token</code>, <code>eos_token</code>, <code>unk_token</code>,
<code>sep_token</code>, <code>pad_token</code>, <code>cls_token</code>, <code>mask_token</code>, <code>additional_special_tokens</code>].</p>
<p>Tokens are only added if they are not already in the vocabulary (tested by checking if the tokenizer
assign the index of the <code>unk_token</code> to them).`,name:"special_tokens_dict"},{anchor:"transformers.PreTrainedTokenizerFast.add_special_tokens.replace_additional_special_tokens",description:`<strong>replace_additional_special_tokens</strong> (<code>bool</code>, <em>optional</em>,, defaults to <code>True</code>) &#x2014;
If <code>True</code>, the existing list of additional special tokens will be replaced by the list provided in
<code>special_tokens_dict</code>. Otherwise, <code>self._special_tokens_map[&quot;additional_special_tokens&quot;]</code> is just extended. In the former
case, the tokens will NOT be removed from the tokenizer&#x2019;s full vocabulary - they are only being flagged
as non-special tokens. Remember, this only affects which tokens are skipped during decoding, not the
<code>added_tokens_encoder</code> and <code>added_tokens_decoder</code>. This means that the previous
<code>additional_special_tokens</code> are still added tokens, and will not be split by the model.`,name:"replace_additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L890",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of tokens added to the vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),Pe=new nr({props:{anchor:"transformers.PreTrainedTokenizerFast.add_special_tokens.example",$$slots:{default:[Dc]},$$scope:{ctx:J}}}),bt=new z({props:{name:"apply_chat_template",anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template",parameters:[{name:"conversation",val:": typing.Union[list[dict[str, str]], list[list[dict[str, str]]]]"},{name:"tools",val:": typing.Optional[list[typing.Union[dict, typing.Callable]]] = None"},{name:"documents",val:": typing.Optional[list[dict[str, str]]] = None"},{name:"chat_template",val:": typing.Optional[str] = None"},{name:"add_generation_prompt",val:": bool = False"},{name:"continue_final_message",val:": bool = False"},{name:"tokenize",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": bool = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_dict",val:": bool = False"},{name:"return_assistant_tokens_mask",val:": bool = False"},{name:"tokenizer_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.conversation",description:`<strong>conversation</strong> (Union[list[dict[str, str]], list[list[dict[str, str]]]]) &#x2014; A list of dicts
with &#x201C;role&#x201D; and &#x201C;content&#x201D; keys, representing the chat history so far.`,name:"conversation"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.tools",description:`<strong>tools</strong> (<code>list[Union[Dict, Callable]]</code>, <em>optional</em>) &#x2014;
A list of tools (callable functions) that will be accessible to the model. If the template does not
support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
giving the name, description and argument types for the tool. See our
<a href="https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use" rel="nofollow">chat templating guide</a>
for more information.`,name:"tools"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.documents",description:`<strong>documents</strong> (<code>list[dict[str, str]]</code>, <em>optional</em>) &#x2014;
A list of dicts representing documents that will be accessible to the model if it is performing RAG
(retrieval-augmented generation). If the template does not support RAG, this argument will have no
effect. We recommend that each document should be a dict containing &#x201C;title&#x201D; and &#x201C;text&#x201D; keys. Please
see the RAG section of the <a href="https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG" rel="nofollow">chat templating guide</a>
for examples of passing documents with chat templates.`,name:"documents"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A Jinja template to use for this conversion. It is usually not necessary to pass anything to this
argument, as the model&#x2019;s template will be used by default.`,name:"chat_template"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.add_generation_prompt",description:`<strong>add_generation_prompt</strong> (bool, <em>optional</em>) &#x2014;
If this is set, a prompt with the token(s) that indicate
the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
Note that this argument will be passed to the chat template, and so it must be supported in the
template for this argument to have any effect.`,name:"add_generation_prompt"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.continue_final_message",description:`<strong>continue_final_message</strong> (bool, <em>optional</em>) &#x2014;
If this is set, the chat will be formatted so that the final
message in the chat is open-ended, without any EOS tokens. The model will continue this message
rather than starting a new one. This allows you to &#x201C;prefill&#x201D; part of
the model&#x2019;s response for it. Cannot be used at the same time as <code>add_generation_prompt</code>.`,name:"continue_final_message"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.tokenize",description:`<strong>tokenize</strong> (<code>bool</code>, defaults to <code>True</code>) &#x2014;
Whether to tokenize the output. If <code>False</code>, the output will be a string.`,name:"tokenize"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.truncation",description:`<strong>truncation</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to truncate sequences at the maximum length. Has no effect if tokenize is <code>False</code>.`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is <code>False</code>. If
not specified, the tokenizer&#x2019;s <code>max_length</code> attribute will be used as a default.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Has no effect if tokenize is <code>False</code>. Acceptable
values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.Tensor</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to return a dictionary with named outputs. Has no effect if tokenize is <code>False</code>.`,name:"return_dict"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.tokenizer_kwargs",description:"<strong>tokenizer_kwargs</strong> (<code>dict[str -- Any]</code>, <em>optional</em>): Additional kwargs to pass to the tokenizer.",name:"tokenizer_kwargs"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.return_assistant_tokens_mask",description:`<strong>return_assistant_tokens_mask</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
the mask will contain 1. For user and system tokens, the mask will contain 0.
This functionality is only available for chat templates that support it via the <code>{% generation %}</code> keyword.`,name:"return_assistant_tokens_mask"},{anchor:"transformers.PreTrainedTokenizerFast.apply_chat_template.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; Additional kwargs to pass to the template renderer. Will be accessible by the chat template.",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L1518",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of token ids representing the tokenized chat so far, including control tokens. This
output is ready to pass to the model, either directly or via methods like <code>generate()</code>. If <code>return_dict</code> is
set, will return a dict of tokenizer outputs instead.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Union[list[int], Dict]</code></p>
`}}),Tt=new z({props:{name:"batch_decode",anchor:"transformers.PreTrainedTokenizerFast.batch_decode",parameters:[{name:"sequences",val:": typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.batch_decode.sequences",description:`<strong>sequences</strong> (<code>Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"sequences"},{anchor:"transformers.PreTrainedTokenizerFast.batch_decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.batch_decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizerFast.batch_decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3833",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of decoded sentences.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code></p>
`}}),vt=new z({props:{name:"decode",anchor:"transformers.PreTrainedTokenizerFast.decode",parameters:[{name:"token_ids",val:": typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.decode.token_ids",description:`<strong>token_ids</strong> (<code>Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"token_ids"},{anchor:"transformers.PreTrainedTokenizerFast.decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.PreTrainedTokenizerFast.decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Will be passed to the underlying model specific decode method.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3867",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded sentence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),xt=new z({props:{name:"encode",anchor:"transformers.PreTrainedTokenizerFast.encode",parameters:[{name:"text",val:": typing.Union[str, list[str], list[int]]"},{name:"text_pair",val:": typing.Union[str, list[str], list[int], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.encode.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code> or <code>list[int]</code>) &#x2014;
The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
<code>tokenize</code> method) or a list of integers (tokenized string ids using the <code>convert_tokens_to_ids</code>
method).`,name:"text"},{anchor:"transformers.PreTrainedTokenizerFast.encode.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code> or <code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
the <code>tokenize</code> method) or a list of integers (tokenized string ids using the <code>convert_tokens_to_ids</code>
method).`,name:"text_pair"},{anchor:"transformers.PreTrainedTokenizerFast.encode.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.encode.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.PreTrainedTokenizerFast.encode.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.PreTrainedTokenizerFast.encode.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerFast.encode.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizerFast.encode.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.PreTrainedTokenizerFast.encode.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerFast.encode.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.PreTrainedTokenizerFast.encode.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.PreTrainedTokenizerFast.encode.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; Passed along to the <code>.tokenize()</code> method.",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2667",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The tokenized ids of the text.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code>, <code>torch.Tensor</code>, <code>tf.Tensor</code> or <code>np.ndarray</code></p>
`}}),yt=new z({props:{name:"push_to_hub",anchor:"transformers.PreTrainedTokenizerFast.push_to_hub",parameters:[{name:"repo_id",val:": str"},{name:"use_temp_dir",val:": typing.Optional[bool] = None"},{name:"commit_message",val:": typing.Optional[str] = None"},{name:"private",val:": typing.Optional[bool] = None"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"max_shard_size",val:": typing.Union[str, int, NoneType] = '5GB'"},{name:"create_pr",val:": bool = False"},{name:"safe_serialization",val:": bool = True"},{name:"revision",val:": typing.Optional[str] = None"},{name:"commit_description",val:": typing.Optional[str] = None"},{name:"tags",val:": typing.Optional[list[str]] = None"},{name:"**deprecated_kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.repo_id",description:`<strong>repo_id</strong> (<code>str</code>) &#x2014;
The name of the repository you want to push your tokenizer to. It should contain your organization name
when pushing to a given organization.`,name:"repo_id"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.use_temp_dir",description:`<strong>use_temp_dir</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
Will default to <code>True</code> if there is no directory named like <code>repo_id</code>, <code>False</code> otherwise.`,name:"use_temp_dir"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.commit_message",description:`<strong>commit_message</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Message to commit while pushing. Will default to <code>&quot;Upload tokenizer&quot;</code>.`,name:"commit_message"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.private",description:`<strong>private</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to make the repo private. If <code>None</code> (default), the repo will be public unless the organization&#x2019;s default is private. This value is ignored if the repo already exists.`,name:"private"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.token",description:`<strong>token</strong> (<code>bool</code> or <code>str</code>, <em>optional</em>) &#x2014;
The token to use as HTTP bearer authorization for remote files. If <code>True</code>, will use the token generated
when running <code>hf auth login</code> (stored in <code>~/.huggingface</code>). Will default to <code>True</code> if <code>repo_url</code>
is not specified.`,name:"token"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.max_shard_size",description:`<strong>max_shard_size</strong> (<code>int</code> or <code>str</code>, <em>optional</em>, defaults to <code>&quot;5GB&quot;</code>) &#x2014;
Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
will then be each of size lower than this size. If expressed as a string, needs to be digits followed
by a unit (like <code>&quot;5MB&quot;</code>). We default it to <code>&quot;5GB&quot;</code> so that users can easily load models on free-tier
Google Colab instances without any CPU OOM issues.`,name:"max_shard_size"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.create_pr",description:`<strong>create_pr</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to create a PR with the uploaded files or directly commit.`,name:"create_pr"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.safe_serialization",description:`<strong>safe_serialization</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to convert the model weights in safetensors format for safer serialization.`,name:"safe_serialization"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.revision",description:`<strong>revision</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Branch to push the uploaded files to.`,name:"revision"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.commit_description",description:`<strong>commit_description</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The description of the commit that will be created`,name:"commit_description"},{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.tags",description:`<strong>tags</strong> (<code>list[str]</code>, <em>optional</em>) &#x2014;
List of tags to push on the Hub.`,name:"tags"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L847"}}),Ce=new nr({props:{anchor:"transformers.PreTrainedTokenizerFast.push_to_hub.example",$$slots:{default:[Hc]},$$scope:{ctx:J}}}),wt=new z({props:{name:"convert_ids_to_tokens",anchor:"transformers.PreTrainedTokenizerFast.convert_ids_to_tokens",parameters:[{name:"ids",val:": typing.Union[int, list[int]]"},{name:"skip_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.convert_ids_to_tokens.ids",description:`<strong>ids</strong> (<code>int</code> or <code>list[int]</code>) &#x2014;
The token id (or token ids) to convert to tokens.`,name:"ids"},{anchor:"transformers.PreTrainedTokenizerFast.convert_ids_to_tokens.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_fast.py#L406",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded token(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code> or <code>list[str]</code></p>
`}}),zt=new z({props:{name:"convert_tokens_to_ids",anchor:"transformers.PreTrainedTokenizerFast.convert_tokens_to_ids",parameters:[{name:"tokens",val:": typing.Union[str, collections.abc.Iterable[str]]"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.convert_tokens_to_ids.tokens",description:"<strong>tokens</strong> (<code>str</code> or <code>Iterable[str]</code>) &#x2014; One or several token(s) to convert to token id(s).",name:"tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_fast.py#L354",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token id or list of token ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code> or <code>list[int]</code></p>
`}}),$t=new z({props:{name:"get_added_vocab",anchor:"transformers.PreTrainedTokenizerFast.get_added_vocab",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_fast.py#L272",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The added tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>dict[str, int]</code></p>
`}}),Pt=new z({props:{name:"num_special_tokens_to_add",anchor:"transformers.PreTrainedTokenizerFast.num_special_tokens_to_add",parameters:[{name:"pair",val:": bool = False"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.num_special_tokens_to_add.pair",description:`<strong>pair</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the number of added tokens should be computed in the case of a sequence pair or a single
sequence.`,name:"pair"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_fast.py#L385",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of special tokens added to sequences.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),We=new Ic({props:{$$slots:{default:[Sc]},$$scope:{ctx:J}}}),Mt=new z({props:{name:"set_truncation_and_padding",anchor:"transformers.PreTrainedTokenizerFast.set_truncation_and_padding",parameters:[{name:"padding_strategy",val:": PaddingStrategy"},{name:"truncation_strategy",val:": TruncationStrategy"},{name:"max_length",val:": int"},{name:"stride",val:": int"},{name:"pad_to_multiple_of",val:": typing.Optional[int]"},{name:"padding_side",val:": typing.Optional[str]"}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.set_truncation_and_padding.padding_strategy",description:`<strong>padding_strategy</strong> (<a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>) &#x2014;
The kind of padding that will be applied to the input`,name:"padding_strategy"},{anchor:"transformers.PreTrainedTokenizerFast.set_truncation_and_padding.truncation_strategy",description:`<strong>truncation_strategy</strong> (<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>) &#x2014;
The kind of truncation that will be applied to the input`,name:"truncation_strategy"},{anchor:"transformers.PreTrainedTokenizerFast.set_truncation_and_padding.max_length",description:`<strong>max_length</strong> (<code>int</code>) &#x2014;
The maximum size of a sequence.`,name:"max_length"},{anchor:"transformers.PreTrainedTokenizerFast.set_truncation_and_padding.stride",description:`<strong>stride</strong> (<code>int</code>) &#x2014;
The stride to use when handling overflow.`,name:"stride"},{anchor:"transformers.PreTrainedTokenizerFast.set_truncation_and_padding.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability <code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.PreTrainedTokenizerFast.set_truncation_and_padding.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_fast.py#L437"}}),qt=new z({props:{name:"train_new_from_iterator",anchor:"transformers.PreTrainedTokenizerFast.train_new_from_iterator",parameters:[{name:"text_iterator",val:""},{name:"vocab_size",val:""},{name:"length",val:" = None"},{name:"new_special_tokens",val:" = None"},{name:"special_tokens_map",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PreTrainedTokenizerFast.train_new_from_iterator.text_iterator",description:`<strong>text_iterator</strong> (generator of <code>list[str]</code>) &#x2014;
The training corpus. Should be a generator of batches of texts, for instance a list of lists of texts
if you have everything in memory.`,name:"text_iterator"},{anchor:"transformers.PreTrainedTokenizerFast.train_new_from_iterator.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>) &#x2014;
The size of the vocabulary you want for your tokenizer.`,name:"vocab_size"},{anchor:"transformers.PreTrainedTokenizerFast.train_new_from_iterator.length",description:`<strong>length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The total number of sequences in the iterator. This is used to provide meaningful progress tracking`,name:"length"},{anchor:"transformers.PreTrainedTokenizerFast.train_new_from_iterator.new_special_tokens",description:`<strong>new_special_tokens</strong> (list of <code>str</code> or <code>AddedToken</code>, <em>optional</em>) &#x2014;
A list of new special tokens to add to the tokenizer you are training.`,name:"new_special_tokens"},{anchor:"transformers.PreTrainedTokenizerFast.train_new_from_iterator.special_tokens_map",description:`<strong>special_tokens_map</strong> (<code>dict[str, str]</code>, <em>optional</em>) &#x2014;
If you want to rename some of the special tokens this tokenizer uses, pass along a mapping old special
token name to new special token name in this argument.`,name:"special_tokens_map"},{anchor:"transformers.PreTrainedTokenizerFast.train_new_from_iterator.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Additional keyword arguments passed along to the trainer from the &#x1F917; Tokenizers library.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_fast.py#L744",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A new tokenizer of the same type as the original one, trained on
<code>text_iterator</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast"
>PreTrainedTokenizerFast</a></p>
`}}),Ct=new rr({props:{title:"BatchEncoding",local:"transformers.BatchEncoding",headingTag:"h2"}}),It=new z({props:{name:"class transformers.BatchEncoding",anchor:"transformers.BatchEncoding",parameters:[{name:"data",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"encoding",val:": typing.Union[tokenizers.Encoding, collections.abc.Sequence[tokenizers.Encoding], NoneType] = None"},{name:"tensor_type",val:": typing.Union[NoneType, str, transformers.utils.generic.TensorType] = None"},{name:"prepend_batch_axis",val:": bool = False"},{name:"n_sequences",val:": typing.Optional[int] = None"}],parametersDescription:[{anchor:"transformers.BatchEncoding.data",description:`<strong>data</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Dictionary of lists/arrays/tensors returned by the <code>__call__</code>/<code>encode_plus</code>/<code>batch_encode_plus</code> methods
(&#x2018;input_ids&#x2019;, &#x2018;attention_mask&#x2019;, etc.).`,name:"data"},{anchor:"transformers.BatchEncoding.encoding",description:`<strong>encoding</strong> (<code>tokenizers.Encoding</code> or <code>Sequence[tokenizers.Encoding]</code>, <em>optional</em>) &#x2014;
If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
space to token space the <code>tokenizers.Encoding</code> instance or list of instance (for batches) hold this
information.`,name:"encoding"},{anchor:"transformers.BatchEncoding.tensor_type",description:`<strong>tensor_type</strong> (<code>Union[None, str, TensorType]</code>, <em>optional</em>) &#x2014;
You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
initialization.`,name:"tensor_type"},{anchor:"transformers.BatchEncoding.prepend_batch_axis",description:`<strong>prepend_batch_axis</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add a batch axis when converting to tensors (see <code>tensor_type</code> above). Note that this
parameter has an effect if the parameter <code>tensor_type</code> is set, <em>otherwise has no effect</em>.`,name:"prepend_batch_axis"},{anchor:"transformers.BatchEncoding.n_sequences",description:`<strong>n_sequences</strong> (<code>Optional[int]</code>, <em>optional</em>) &#x2014;
You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
initialization.`,name:"n_sequences"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L192"}}),Ft=new z({props:{name:"char_to_token",anchor:"transformers.BatchEncoding.char_to_token",parameters:[{name:"batch_or_char_index",val:": int"},{name:"char_index",val:": typing.Optional[int] = None"},{name:"sequence_index",val:": int = 0"}],parametersDescription:[{anchor:"transformers.BatchEncoding.char_to_token.batch_or_char_index",description:`<strong>batch_or_char_index</strong> (<code>int</code>) &#x2014;
Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
the word in the sequence`,name:"batch_or_char_index"},{anchor:"transformers.BatchEncoding.char_to_token.char_index",description:`<strong>char_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If a batch index is provided in <em>batch_or_token_index</em>, this can be the index of the word in the
sequence.`,name:"char_index"},{anchor:"transformers.BatchEncoding.char_to_token.sequence_index",description:`<strong>sequence_index</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
or 1) the provided character index belongs to.`,name:"sequence_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L563",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Index of the token, or None if the char index refers to a whitespace only token and whitespace is
trimmed with <code>trim_offsets=True</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),jt=new z({props:{name:"char_to_word",anchor:"transformers.BatchEncoding.char_to_word",parameters:[{name:"batch_or_char_index",val:": int"},{name:"char_index",val:": typing.Optional[int] = None"},{name:"sequence_index",val:": int = 0"}],parametersDescription:[{anchor:"transformers.BatchEncoding.char_to_word.batch_or_char_index",description:`<strong>batch_or_char_index</strong> (<code>int</code>) &#x2014;
Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
the character in the original string.`,name:"batch_or_char_index"},{anchor:"transformers.BatchEncoding.char_to_word.char_index",description:`<strong>char_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If a batch index is provided in <em>batch_or_token_index</em>, this can be the index of the character in the
original string.`,name:"char_index"},{anchor:"transformers.BatchEncoding.char_to_word.sequence_index",description:`<strong>sequence_index</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
or 1) the provided character index belongs to.`,name:"sequence_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L650",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Index or indices of the associated encoded token(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code> or <code>list[int]</code></p>
`}}),Wt=new z({props:{name:"convert_to_tensors",anchor:"transformers.BatchEncoding.convert_to_tensors",parameters:[{name:"tensor_type",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"prepend_batch_axis",val:": bool = False"}],parametersDescription:[{anchor:"transformers.BatchEncoding.convert_to_tensors.tensor_type",description:`<strong>tensor_type</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
The type of tensors to use. If <code>str</code>, should be one of the values of the enum <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>. If
<code>None</code>, no modification is done.`,name:"tensor_type"},{anchor:"transformers.BatchEncoding.convert_to_tensors.prepend_batch_axis",description:`<strong>prepend_batch_axis</strong> (<code>int</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add the batch dimension during the conversion.`,name:"prepend_batch_axis"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L689"}}),Jt=new z({props:{name:"sequence_ids",anchor:"transformers.BatchEncoding.sequence_ids",parameters:[{name:"batch_index",val:": int = 0"}],parametersDescription:[{anchor:"transformers.BatchEncoding.sequence_ids.batch_index",description:"<strong>batch_index</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014; The index to access in the batch.",name:"batch_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L327",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list indicating the sequence id corresponding to each token. Special tokens added
by the tokenizer are mapped to <code>None</code> and other tokens are mapped to the index of their corresponding
sequence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Optional[int]]</code></p>
`}}),Ut=new z({props:{name:"to",anchor:"transformers.BatchEncoding.to",parameters:[{name:"device",val:": typing.Union[str, ForwardRef('torch.device')]"},{name:"non_blocking",val:": bool = False"}],parametersDescription:[{anchor:"transformers.BatchEncoding.to.device",description:"<strong>device</strong> (<code>str</code> or <code>torch.device</code>) &#x2014; The device to put the tensors on.",name:"device"},{anchor:"transformers.BatchEncoding.to.non_blocking",description:"<strong>non_blocking</strong> (<code>bool</code>) &#x2014; Whether to perform the copy asynchronously.",name:"non_blocking"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L792",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The same instance after modification.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),Nt=new z({props:{name:"token_to_chars",anchor:"transformers.BatchEncoding.token_to_chars",parameters:[{name:"batch_or_token_index",val:": int"},{name:"token_index",val:": typing.Optional[int] = None"}],parametersDescription:[{anchor:"transformers.BatchEncoding.token_to_chars.batch_or_token_index",description:`<strong>batch_or_token_index</strong> (<code>int</code>) &#x2014;
Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
the token in the sequence.`,name:"batch_or_token_index"},{anchor:"transformers.BatchEncoding.token_to_chars.token_index",description:`<strong>token_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If a batch index is provided in <em>batch_or_token_index</em>, this can be the index of the token or tokens in
the sequence.`,name:"token_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L524",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Span of characters in the original string, or None, if the token
(e.g. <s>, </s>) doesn’t correspond to any chars in the origin string.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.CharSpan"
>CharSpan</a></p>
`}}),Bt=new z({props:{name:"token_to_sequence",anchor:"transformers.BatchEncoding.token_to_sequence",parameters:[{name:"batch_or_token_index",val:": int"},{name:"token_index",val:": typing.Optional[int] = None"}],parametersDescription:[{anchor:"transformers.BatchEncoding.token_to_sequence.batch_or_token_index",description:`<strong>batch_or_token_index</strong> (<code>int</code>) &#x2014;
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the token in the sequence.`,name:"batch_or_token_index"},{anchor:"transformers.BatchEncoding.token_to_sequence.token_index",description:`<strong>token_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If a batch index is provided in <em>batch_or_token_index</em>, this can be the index of the token in the
sequence.`,name:"token_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L394",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Index of the word in the input sequence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),Vt=new z({props:{name:"token_to_word",anchor:"transformers.BatchEncoding.token_to_word",parameters:[{name:"batch_or_token_index",val:": int"},{name:"token_index",val:": typing.Optional[int] = None"}],parametersDescription:[{anchor:"transformers.BatchEncoding.token_to_word.batch_or_token_index",description:`<strong>batch_or_token_index</strong> (<code>int</code>) &#x2014;
Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
the token in the sequence.`,name:"batch_or_token_index"},{anchor:"transformers.BatchEncoding.token_to_word.token_index",description:`<strong>token_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If a batch index is provided in <em>batch_or_token_index</em>, this can be the index of the token in the
sequence.`,name:"token_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L433",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Index of the word in the input sequence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),Lt=new z({props:{name:"tokens",anchor:"transformers.BatchEncoding.tokens",parameters:[{name:"batch_index",val:": int = 0"}],parametersDescription:[{anchor:"transformers.BatchEncoding.tokens.batch_index",description:"<strong>batch_index</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014; The index to access in the batch.",name:"batch_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L309",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of tokens at that index.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[str]</code></p>
`}}),Et=new z({props:{name:"word_ids",anchor:"transformers.BatchEncoding.word_ids",parameters:[{name:"batch_index",val:": int = 0"}],parametersDescription:[{anchor:"transformers.BatchEncoding.word_ids.batch_index",description:"<strong>batch_index</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014; The index to access in the batch.",name:"batch_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L375",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list indicating the word corresponding to each token. Special tokens added by the
tokenizer are mapped to <code>None</code> and other tokens are mapped to the index of their corresponding word
(several tokens will be mapped to the same word index if they are parts of that word).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Optional[int]]</code></p>
`}}),Zt=new z({props:{name:"word_to_chars",anchor:"transformers.BatchEncoding.word_to_chars",parameters:[{name:"batch_or_word_index",val:": int"},{name:"word_index",val:": typing.Optional[int] = None"},{name:"sequence_index",val:": int = 0"}],parametersDescription:[{anchor:"transformers.BatchEncoding.word_to_chars.batch_or_word_index",description:`<strong>batch_or_word_index</strong> (<code>int</code>) &#x2014;
Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
the word in the sequence`,name:"batch_or_word_index"},{anchor:"transformers.BatchEncoding.word_to_chars.word_index",description:`<strong>word_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If a batch index is provided in <em>batch_or_token_index</em>, this can be the index of the word in the
sequence.`,name:"word_index"},{anchor:"transformers.BatchEncoding.word_to_chars.sequence_index",description:`<strong>sequence_index</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
or 1) the provided word index belongs to.`,name:"sequence_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L605",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Span(s) of the associated character or characters in the string. CharSpan
are NamedTuple with:</p>
<ul>
<li>start: index of the first character associated to the token in the original string</li>
<li>end: index of the character following the last character associated to the token in the original
string</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>CharSpan</code> or <code>list[CharSpan]</code></p>
`}}),Dt=new z({props:{name:"word_to_tokens",anchor:"transformers.BatchEncoding.word_to_tokens",parameters:[{name:"batch_or_word_index",val:": int"},{name:"word_index",val:": typing.Optional[int] = None"},{name:"sequence_index",val:": int = 0"}],parametersDescription:[{anchor:"transformers.BatchEncoding.word_to_tokens.batch_or_word_index",description:`<strong>batch_or_word_index</strong> (<code>int</code>) &#x2014;
Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
the word in the sequence.`,name:"batch_or_word_index"},{anchor:"transformers.BatchEncoding.word_to_tokens.word_index",description:`<strong>word_index</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If a batch index is provided in <em>batch_or_token_index</em>, this can be the index of the word in the
sequence.`,name:"word_index"},{anchor:"transformers.BatchEncoding.word_to_tokens.sequence_index",description:`<strong>sequence_index</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
or 1) the provided word index belongs to.`,name:"sequence_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L471",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Span of tokens in the encoded sequence. Returns
<code>None</code> if no tokens correspond to the word. This can happen especially when the token is a special token
that has been used to format the tokenization. For example when we add a class token at the very beginning
of the tokenization.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>(<a
  href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.TokenSpan"
>TokenSpan</a>, <em>optional</em>)</p>
`}}),Ht=new z({props:{name:"words",anchor:"transformers.BatchEncoding.words",parameters:[{name:"batch_index",val:": int = 0"}],parametersDescription:[{anchor:"transformers.BatchEncoding.words.batch_index",description:"<strong>batch_index</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014; The index to access in the batch.",name:"batch_index"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L351",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list indicating the word corresponding to each token. Special tokens added by the
tokenizer are mapped to <code>None</code> and other tokens are mapped to the index of their corresponding word
(several tokens will be mapped to the same word index if they are parts of that word).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[Optional[int]]</code></p>
`}}),St=new Nc({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/tokenizer.md"}}),{c(){c=r("meta"),I=n(),b=r("p"),_=n(),l(P.$$.fragment),d=n(),C=r("p"),C.innerHTML=Mi,sr=n(),Ze=r("ol"),Ze.innerHTML=qi,ar=n(),De=r("p"),De.innerHTML=Ci,ir=n(),He=r("p"),He.innerHTML=Ii,dr=n(),Se=r("ul"),Se.innerHTML=Fi,cr=n(),Ae=r("p"),Ae.innerHTML=ji,lr=n(),l(Ge.$$.fragment),pr=n(),Re=r("p"),Re.innerHTML=Wi,mr=n(),Xe=r("p"),Xe.innerHTML=Ji,hr=n(),l(Ye.$$.fragment),ur=n(),l(Oe.$$.fragment),fr=n(),T=r("div"),l(Qe.$$.fragment),Vr=n(),Yt=r("p"),Yt.textContent=Ui,Lr=n(),Ot=r("p"),Ot.innerHTML=Ni,Er=n(),Qt=r("p"),Qt.textContent=Bi,Zr=n(),Kt=r("p"),Kt.textContent=Vi,Dr=n(),en=r("p"),en.textContent=Li,Hr=n(),tn=r("ul"),tn.innerHTML=Ei,Sr=n(),fe=r("div"),l(Ke.$$.fragment),Ar=n(),nn=r("p"),nn.textContent=Zi,Gr=n(),L=r("div"),l(et.$$.fragment),Rr=n(),on=r("p"),on.textContent=Di,Xr=n(),rn=r("p"),rn.textContent=Hi,Yr=n(),sn=r("p"),sn.innerHTML=Si,Or=n(),l(ge.$$.fragment),Qr=n(),F=r("div"),l(tt.$$.fragment),Kr=n(),an=r("p"),an.textContent=Ai,es=n(),dn=r("p"),dn.textContent=Gi,ts=n(),cn=r("p"),cn.innerHTML=Ri,ns=n(),ln=r("p"),ln.innerHTML=Xi,os=n(),pn=r("ul"),pn.innerHTML=Yi,rs=n(),mn=r("p"),mn.innerHTML=Oi,ss=n(),l(_e.$$.fragment),as=n(),ke=r("div"),l(nt.$$.fragment),is=n(),hn=r("p"),hn.innerHTML=Qi,ds=n(),be=r("div"),l(ot.$$.fragment),cs=n(),un=r("p"),un.textContent=Ki,ls=n(),Y=r("div"),l(rt.$$.fragment),ps=n(),fn=r("p"),fn.textContent=ed,ms=n(),gn=r("p"),gn.innerHTML=td,hs=n(),O=r("div"),l(st.$$.fragment),us=n(),_n=r("p"),_n.textContent=nd,fs=n(),kn=r("p"),kn.innerHTML=od,gs=n(),Q=r("div"),l(at.$$.fragment),_s=n(),bn=r("p"),bn.textContent=rd,ks=n(),l(Te.$$.fragment),bs=n(),ve=r("div"),l(it.$$.fragment),Ts=n(),Tn=r("p"),Tn.textContent=sd,vs=n(),xe=r("div"),l(dt.$$.fragment),xs=n(),vn=r("p"),vn.textContent=ad,ys=n(),ye=r("div"),l(ct.$$.fragment),ws=n(),xn=r("p"),xn.textContent=id,zs=n(),K=r("div"),l(lt.$$.fragment),$s=n(),yn=r("p"),yn.textContent=dd,Ps=n(),l(we.$$.fragment),Ms=n(),ee=r("div"),l(pt.$$.fragment),qs=n(),wn=r("p"),wn.textContent=cd,Cs=n(),zn=r("p"),zn.innerHTML=ld,Is=n(),te=r("div"),l(mt.$$.fragment),Fs=n(),$n=r("p"),$n.textContent=pd,js=n(),Pn=r("p"),Pn.textContent=md,gr=n(),l(ht.$$.fragment),_r=n(),ut=r("p"),ut.innerHTML=hd,kr=n(),v=r("div"),l(ft.$$.fragment),Ws=n(),Mn=r("p"),Mn.textContent=ud,Js=n(),qn=r("p"),qn.innerHTML=fd,Us=n(),Cn=r("p"),Cn.textContent=gd,Ns=n(),In=r("p"),In.textContent=_d,Bs=n(),Fn=r("p"),Fn.textContent=kd,Vs=n(),jn=r("ul"),jn.innerHTML=bd,Ls=n(),ze=r("div"),l(gt.$$.fragment),Es=n(),Wn=r("p"),Wn.textContent=Td,Zs=n(),E=r("div"),l(_t.$$.fragment),Ds=n(),Jn=r("p"),Jn.textContent=vd,Hs=n(),Un=r("p"),Un.textContent=xd,Ss=n(),Nn=r("p"),Nn.innerHTML=yd,As=n(),l($e.$$.fragment),Gs=n(),j=r("div"),l(kt.$$.fragment),Rs=n(),Bn=r("p"),Bn.textContent=wd,Xs=n(),Vn=r("p"),Vn.textContent=zd,Ys=n(),Ln=r("p"),Ln.innerHTML=$d,Os=n(),En=r("p"),En.innerHTML=Pd,Qs=n(),Zn=r("ul"),Zn.innerHTML=Md,Ks=n(),Dn=r("p"),Dn.innerHTML=qd,ea=n(),l(Pe.$$.fragment),ta=n(),Me=r("div"),l(bt.$$.fragment),na=n(),Hn=r("p"),Hn.innerHTML=Cd,oa=n(),qe=r("div"),l(Tt.$$.fragment),ra=n(),Sn=r("p"),Sn.textContent=Id,sa=n(),ne=r("div"),l(vt.$$.fragment),aa=n(),An=r("p"),An.textContent=Fd,ia=n(),Gn=r("p"),Gn.innerHTML=jd,da=n(),oe=r("div"),l(xt.$$.fragment),ca=n(),Rn=r("p"),Rn.textContent=Wd,la=n(),Xn=r("p"),Xn.innerHTML=Jd,pa=n(),re=r("div"),l(yt.$$.fragment),ma=n(),Yn=r("p"),Yn.textContent=Ud,ha=n(),l(Ce.$$.fragment),ua=n(),Ie=r("div"),l(wt.$$.fragment),fa=n(),On=r("p"),On.textContent=Nd,ga=n(),Fe=r("div"),l(zt.$$.fragment),_a=n(),Qn=r("p"),Qn.textContent=Bd,ka=n(),je=r("div"),l($t.$$.fragment),ba=n(),Kn=r("p"),Kn.textContent=Vd,Ta=n(),se=r("div"),l(Pt.$$.fragment),va=n(),eo=r("p"),eo.textContent=Ld,xa=n(),l(We.$$.fragment),ya=n(),ae=r("div"),l(Mt.$$.fragment),wa=n(),to=r("p"),to.textContent=Ed,za=n(),no=r("p"),no.textContent=Zd,$a=n(),Je=r("div"),l(qt.$$.fragment),Pa=n(),oo=r("p"),oo.textContent=Dd,br=n(),l(Ct.$$.fragment),Tr=n(),M=r("div"),l(It.$$.fragment),Ma=n(),ro=r("p"),ro.innerHTML=Hd,qa=n(),so=r("p"),so.textContent=Sd,Ca=n(),Z=r("div"),l(Ft.$$.fragment),Ia=n(),ao=r("p"),ao.textContent=Ad,Fa=n(),io=r("p"),io.textContent=Gd,ja=n(),co=r("ul"),co.innerHTML=Rd,Wa=n(),lo=r("p"),lo.textContent=Xd,Ja=n(),D=r("div"),l(jt.$$.fragment),Ua=n(),po=r("p"),po.textContent=Yd,Na=n(),mo=r("p"),mo.textContent=Od,Ba=n(),ho=r("ul"),ho.innerHTML=Qd,Va=n(),uo=r("p"),uo.textContent=Kd,La=n(),Ue=r("div"),l(Wt.$$.fragment),Ea=n(),fo=r("p"),fo.textContent=ec,Za=n(),ie=r("div"),l(Jt.$$.fragment),Da=n(),go=r("p"),go.textContent=tc,Ha=n(),_o=r("ul"),_o.innerHTML=nc,Sa=n(),Ne=r("div"),l(Ut.$$.fragment),Aa=n(),ko=r("p"),ko.innerHTML=oc,Ga=n(),N=r("div"),l(Nt.$$.fragment),Ra=n(),bo=r("p"),bo.textContent=rc,Xa=n(),To=r("p"),To.innerHTML=sc,Ya=n(),vo=r("ul"),vo.innerHTML=ac,Oa=n(),xo=r("p"),xo.textContent=ic,Qa=n(),yo=r("ul"),yo.innerHTML=dc,Ka=n(),H=r("div"),l(Bt.$$.fragment),ei=n(),wo=r("p"),wo.innerHTML=cc,ti=n(),zo=r("p"),zo.textContent=lc,ni=n(),$o=r("ul"),$o.innerHTML=pc,oi=n(),Po=r("p"),Po.textContent=mc,ri=n(),S=r("div"),l(Vt.$$.fragment),si=n(),Mo=r("p"),Mo.textContent=hc,ai=n(),qo=r("p"),qo.textContent=uc,ii=n(),Co=r("ul"),Co.innerHTML=fc,di=n(),Io=r("p"),Io.textContent=gc,ci=n(),Be=r("div"),l(Lt.$$.fragment),li=n(),Fo=r("p"),Fo.textContent=_c,pi=n(),Ve=r("div"),l(Et.$$.fragment),mi=n(),jo=r("p"),jo.textContent=kc,hi=n(),B=r("div"),l(Zt.$$.fragment),ui=n(),Wo=r("p"),Wo.textContent=bc,fi=n(),Jo=r("p"),Jo.textContent=Tc,gi=n(),Uo=r("ul"),Uo.innerHTML=vc,_i=n(),No=r("p"),No.textContent=xc,ki=n(),Bo=r("ul"),Bo.innerHTML=yc,bi=n(),U=r("div"),l(Dt.$$.fragment),Ti=n(),Vo=r("p"),Vo.textContent=wc,vi=n(),Lo=r("p"),Lo.innerHTML=zc,xi=n(),Eo=r("ul"),Eo.innerHTML=$c,yi=n(),Zo=r("p"),Zo.textContent=Pc,wi=n(),Do=r("ul"),Do.innerHTML=Mc,zi=n(),Ho=r("p"),Ho.textContent=qc,$i=n(),Le=r("div"),l(Ht.$$.fragment),Pi=n(),So=r("p"),So.textContent=Cc,vr=n(),l(St.$$.fragment),xr=n(),or=r("p"),this.h()},l(t){const g=Uc("svelte-u9bgzb",document.head);c=s(g,"META",{name:!0,content:!0}),g.forEach(i),I=o(t),b=s(t,"P",{}),x(b).forEach(i),_=o(t),p(P.$$.fragment,t),d=o(t),C=s(t,"P",{"data-svelte-h":!0}),a(C)!=="svelte-17srdwe"&&(C.innerHTML=Mi),sr=o(t),Ze=s(t,"OL",{"data-svelte-h":!0}),a(Ze)!=="svelte-1t8u2n0"&&(Ze.innerHTML=qi),ar=o(t),De=s(t,"P",{"data-svelte-h":!0}),a(De)!=="svelte-us8vdh"&&(De.innerHTML=Ci),ir=o(t),He=s(t,"P",{"data-svelte-h":!0}),a(He)!=="svelte-a8vfgj"&&(He.innerHTML=Ii),dr=o(t),Se=s(t,"UL",{"data-svelte-h":!0}),a(Se)!=="svelte-377vss"&&(Se.innerHTML=Fi),cr=o(t),Ae=s(t,"P",{"data-svelte-h":!0}),a(Ae)!=="svelte-17p6s2q"&&(Ae.innerHTML=ji),lr=o(t),p(Ge.$$.fragment,t),pr=o(t),Re=s(t,"P",{"data-svelte-h":!0}),a(Re)!=="svelte-if7mcp"&&(Re.innerHTML=Wi),mr=o(t),Xe=s(t,"P",{"data-svelte-h":!0}),a(Xe)!=="svelte-8xfeba"&&(Xe.innerHTML=Ji),hr=o(t),p(Ye.$$.fragment,t),ur=o(t),p(Oe.$$.fragment,t),fr=o(t),T=s(t,"DIV",{class:!0});var w=x(T);p(Qe.$$.fragment,w),Vr=o(w),Yt=s(w,"P",{"data-svelte-h":!0}),a(Yt)!=="svelte-1vieurq"&&(Yt.textContent=Ui),Lr=o(w),Ot=s(w,"P",{"data-svelte-h":!0}),a(Ot)!=="svelte-45pp9h"&&(Ot.innerHTML=Ni),Er=o(w),Qt=s(w,"P",{"data-svelte-h":!0}),a(Qt)!=="svelte-y6enlf"&&(Qt.textContent=Bi),Zr=o(w),Kt=s(w,"P",{"data-svelte-h":!0}),a(Kt)!=="svelte-abl5qq"&&(Kt.textContent=Vi),Dr=o(w),en=s(w,"P",{"data-svelte-h":!0}),a(en)!=="svelte-1ixo79u"&&(en.textContent=Li),Hr=o(w),tn=s(w,"UL",{"data-svelte-h":!0}),a(tn)!=="svelte-1gddudt"&&(tn.innerHTML=Ei),Sr=o(w),fe=s(w,"DIV",{class:!0});var At=x(fe);p(Ke.$$.fragment,At),Ar=o(At),nn=s(At,"P",{"data-svelte-h":!0}),a(nn)!=="svelte-kpxj0c"&&(nn.textContent=Zi),At.forEach(i),Gr=o(w),L=s(w,"DIV",{class:!0});var G=x(L);p(et.$$.fragment,G),Rr=o(G),on=s(G,"P",{"data-svelte-h":!0}),a(on)!=="svelte-c27xjk"&&(on.textContent=Di),Xr=o(G),rn=s(G,"P",{"data-svelte-h":!0}),a(rn)!=="svelte-j0w5r1"&&(rn.textContent=Hi),Yr=o(G),sn=s(G,"P",{"data-svelte-h":!0}),a(sn)!=="svelte-mkudpf"&&(sn.innerHTML=Si),Or=o(G),p(ge.$$.fragment,G),G.forEach(i),Qr=o(w),F=s(w,"DIV",{class:!0});var W=x(F);p(tt.$$.fragment,W),Kr=o(W),an=s(W,"P",{"data-svelte-h":!0}),a(an)!=="svelte-1j8s0i5"&&(an.textContent=Ai),es=o(W),dn=s(W,"P",{"data-svelte-h":!0}),a(dn)!=="svelte-1w3ayx9"&&(dn.textContent=Gi),ts=o(W),cn=s(W,"P",{"data-svelte-h":!0}),a(cn)!=="svelte-mkudpf"&&(cn.innerHTML=Ri),ns=o(W),ln=s(W,"P",{"data-svelte-h":!0}),a(ln)!=="svelte-5hxtpc"&&(ln.innerHTML=Xi),os=o(W),pn=s(W,"UL",{"data-svelte-h":!0}),a(pn)!=="svelte-1pes0uj"&&(pn.innerHTML=Yi),rs=o(W),mn=s(W,"P",{"data-svelte-h":!0}),a(mn)!=="svelte-hs52sw"&&(mn.innerHTML=Oi),ss=o(W),p(_e.$$.fragment,W),W.forEach(i),as=o(w),ke=s(w,"DIV",{class:!0});var Gt=x(ke);p(nt.$$.fragment,Gt),is=o(Gt),hn=s(Gt,"P",{"data-svelte-h":!0}),a(hn)!=="svelte-j87b6t"&&(hn.innerHTML=Qi),Gt.forEach(i),ds=o(w),be=s(w,"DIV",{class:!0});var Rt=x(be);p(ot.$$.fragment,Rt),cs=o(Rt),un=s(Rt,"P",{"data-svelte-h":!0}),a(un)!=="svelte-1deng2j"&&(un.textContent=Ki),Rt.forEach(i),ls=o(w),Y=s(w,"DIV",{class:!0});var he=x(Y);p(rt.$$.fragment,he),ps=o(he),fn=s(he,"P",{"data-svelte-h":!0}),a(fn)!=="svelte-vbfkpu"&&(fn.textContent=ed),ms=o(he),gn=s(he,"P",{"data-svelte-h":!0}),a(gn)!=="svelte-125uxon"&&(gn.innerHTML=td),he.forEach(i),hs=o(w),O=s(w,"DIV",{class:!0});var ue=x(O);p(st.$$.fragment,ue),us=o(ue),_n=s(ue,"P",{"data-svelte-h":!0}),a(_n)!=="svelte-12b8hzo"&&(_n.textContent=nd),fs=o(ue),kn=s(ue,"P",{"data-svelte-h":!0}),a(kn)!=="svelte-1kyhveh"&&(kn.innerHTML=od),ue.forEach(i),gs=o(w),Q=s(w,"DIV",{class:!0});var Ao=x(Q);p(at.$$.fragment,Ao),_s=o(Ao),bn=s(Ao,"P",{"data-svelte-h":!0}),a(bn)!=="svelte-tpmkl3"&&(bn.textContent=rd),ks=o(Ao),p(Te.$$.fragment,Ao),Ao.forEach(i),bs=o(w),ve=s(w,"DIV",{class:!0});var wr=x(ve);p(it.$$.fragment,wr),Ts=o(wr),Tn=s(wr,"P",{"data-svelte-h":!0}),a(Tn)!=="svelte-cx157h"&&(Tn.textContent=sd),wr.forEach(i),vs=o(w),xe=s(w,"DIV",{class:!0});var zr=x(xe);p(dt.$$.fragment,zr),xs=o(zr),vn=s(zr,"P",{"data-svelte-h":!0}),a(vn)!=="svelte-1urz5jj"&&(vn.textContent=ad),zr.forEach(i),ys=o(w),ye=s(w,"DIV",{class:!0});var $r=x(ye);p(ct.$$.fragment,$r),ws=o($r),xn=s($r,"P",{"data-svelte-h":!0}),a(xn)!=="svelte-2yfcci"&&(xn.textContent=id),$r.forEach(i),zs=o(w),K=s(w,"DIV",{class:!0});var Go=x(K);p(lt.$$.fragment,Go),$s=o(Go),yn=s(Go,"P",{"data-svelte-h":!0}),a(yn)!=="svelte-1oqr1g4"&&(yn.textContent=dd),Ps=o(Go),p(we.$$.fragment,Go),Go.forEach(i),Ms=o(w),ee=s(w,"DIV",{class:!0});var Ro=x(ee);p(pt.$$.fragment,Ro),qs=o(Ro),wn=s(Ro,"P",{"data-svelte-h":!0}),a(wn)!=="svelte-suiszn"&&(wn.textContent=cd),Cs=o(Ro),zn=s(Ro,"P",{"data-svelte-h":!0}),a(zn)!=="svelte-1bw0rb5"&&(zn.innerHTML=ld),Ro.forEach(i),Is=o(w),te=s(w,"DIV",{class:!0});var Xo=x(te);p(mt.$$.fragment,Xo),Fs=o(Xo),$n=s(Xo,"P",{"data-svelte-h":!0}),a($n)!=="svelte-sso1qb"&&($n.textContent=pd),js=o(Xo),Pn=s(Xo,"P",{"data-svelte-h":!0}),a(Pn)!=="svelte-1i4xsf5"&&(Pn.textContent=md),Xo.forEach(i),w.forEach(i),gr=o(t),p(ht.$$.fragment,t),_r=o(t),ut=s(t,"P",{"data-svelte-h":!0}),a(ut)!=="svelte-wnznt5"&&(ut.innerHTML=hd),kr=o(t),v=s(t,"DIV",{class:!0});var $=x(v);p(ft.$$.fragment,$),Ws=o($),Mn=s($,"P",{"data-svelte-h":!0}),a(Mn)!=="svelte-1e1i5yj"&&(Mn.textContent=ud),Js=o($),qn=s($,"P",{"data-svelte-h":!0}),a(qn)!=="svelte-45pp9h"&&(qn.innerHTML=fd),Us=o($),Cn=s($,"P",{"data-svelte-h":!0}),a(Cn)!=="svelte-99yswb"&&(Cn.textContent=gd),Ns=o($),In=s($,"P",{"data-svelte-h":!0}),a(In)!=="svelte-1y9tnev"&&(In.textContent=_d),Bs=o($),Fn=s($,"P",{"data-svelte-h":!0}),a(Fn)!=="svelte-1ixo79u"&&(Fn.textContent=kd),Vs=o($),jn=s($,"UL",{"data-svelte-h":!0}),a(jn)!=="svelte-1gddudt"&&(jn.innerHTML=bd),Ls=o($),ze=s($,"DIV",{class:!0});var Pr=x(ze);p(gt.$$.fragment,Pr),Es=o(Pr),Wn=s(Pr,"P",{"data-svelte-h":!0}),a(Wn)!=="svelte-kpxj0c"&&(Wn.textContent=Td),Pr.forEach(i),Zs=o($),E=s($,"DIV",{class:!0});var de=x(E);p(_t.$$.fragment,de),Ds=o(de),Jn=s(de,"P",{"data-svelte-h":!0}),a(Jn)!=="svelte-c27xjk"&&(Jn.textContent=vd),Hs=o(de),Un=s(de,"P",{"data-svelte-h":!0}),a(Un)!=="svelte-j0w5r1"&&(Un.textContent=xd),Ss=o(de),Nn=s(de,"P",{"data-svelte-h":!0}),a(Nn)!=="svelte-mkudpf"&&(Nn.innerHTML=yd),As=o(de),p($e.$$.fragment,de),de.forEach(i),Gs=o($),j=s($,"DIV",{class:!0});var V=x(j);p(kt.$$.fragment,V),Rs=o(V),Bn=s(V,"P",{"data-svelte-h":!0}),a(Bn)!=="svelte-1j8s0i5"&&(Bn.textContent=wd),Xs=o(V),Vn=s(V,"P",{"data-svelte-h":!0}),a(Vn)!=="svelte-1w3ayx9"&&(Vn.textContent=zd),Ys=o(V),Ln=s(V,"P",{"data-svelte-h":!0}),a(Ln)!=="svelte-mkudpf"&&(Ln.innerHTML=$d),Os=o(V),En=s(V,"P",{"data-svelte-h":!0}),a(En)!=="svelte-5hxtpc"&&(En.innerHTML=Pd),Qs=o(V),Zn=s(V,"UL",{"data-svelte-h":!0}),a(Zn)!=="svelte-1pes0uj"&&(Zn.innerHTML=Md),Ks=o(V),Dn=s(V,"P",{"data-svelte-h":!0}),a(Dn)!=="svelte-hs52sw"&&(Dn.innerHTML=qd),ea=o(V),p(Pe.$$.fragment,V),V.forEach(i),ta=o($),Me=s($,"DIV",{class:!0});var Mr=x(Me);p(bt.$$.fragment,Mr),na=o(Mr),Hn=s(Mr,"P",{"data-svelte-h":!0}),a(Hn)!=="svelte-j87b6t"&&(Hn.innerHTML=Cd),Mr.forEach(i),oa=o($),qe=s($,"DIV",{class:!0});var qr=x(qe);p(Tt.$$.fragment,qr),ra=o(qr),Sn=s(qr,"P",{"data-svelte-h":!0}),a(Sn)!=="svelte-1deng2j"&&(Sn.textContent=Id),qr.forEach(i),sa=o($),ne=s($,"DIV",{class:!0});var Yo=x(ne);p(vt.$$.fragment,Yo),aa=o(Yo),An=s(Yo,"P",{"data-svelte-h":!0}),a(An)!=="svelte-vbfkpu"&&(An.textContent=Fd),ia=o(Yo),Gn=s(Yo,"P",{"data-svelte-h":!0}),a(Gn)!=="svelte-125uxon"&&(Gn.innerHTML=jd),Yo.forEach(i),da=o($),oe=s($,"DIV",{class:!0});var Oo=x(oe);p(xt.$$.fragment,Oo),ca=o(Oo),Rn=s(Oo,"P",{"data-svelte-h":!0}),a(Rn)!=="svelte-12b8hzo"&&(Rn.textContent=Wd),la=o(Oo),Xn=s(Oo,"P",{"data-svelte-h":!0}),a(Xn)!=="svelte-1kyhveh"&&(Xn.innerHTML=Jd),Oo.forEach(i),pa=o($),re=s($,"DIV",{class:!0});var Qo=x(re);p(yt.$$.fragment,Qo),ma=o(Qo),Yn=s(Qo,"P",{"data-svelte-h":!0}),a(Yn)!=="svelte-tpmkl3"&&(Yn.textContent=Ud),ha=o(Qo),p(Ce.$$.fragment,Qo),Qo.forEach(i),ua=o($),Ie=s($,"DIV",{class:!0});var Cr=x(Ie);p(wt.$$.fragment,Cr),fa=o(Cr),On=s(Cr,"P",{"data-svelte-h":!0}),a(On)!=="svelte-cx157h"&&(On.textContent=Nd),Cr.forEach(i),ga=o($),Fe=s($,"DIV",{class:!0});var Ir=x(Fe);p(zt.$$.fragment,Ir),_a=o(Ir),Qn=s(Ir,"P",{"data-svelte-h":!0}),a(Qn)!=="svelte-1t879z4"&&(Qn.textContent=Bd),Ir.forEach(i),ka=o($),je=s($,"DIV",{class:!0});var Fr=x(je);p($t.$$.fragment,Fr),ba=o(Fr),Kn=s(Fr,"P",{"data-svelte-h":!0}),a(Kn)!=="svelte-17grjhy"&&(Kn.textContent=Vd),Fr.forEach(i),Ta=o($),se=s($,"DIV",{class:!0});var Ko=x(se);p(Pt.$$.fragment,Ko),va=o(Ko),eo=s(Ko,"P",{"data-svelte-h":!0}),a(eo)!=="svelte-1oqr1g4"&&(eo.textContent=Ld),xa=o(Ko),p(We.$$.fragment,Ko),Ko.forEach(i),ya=o($),ae=s($,"DIV",{class:!0});var er=x(ae);p(Mt.$$.fragment,er),wa=o(er),to=s(er,"P",{"data-svelte-h":!0}),a(to)!=="svelte-wj61ov"&&(to.textContent=Ed),za=o(er),no=s(er,"P",{"data-svelte-h":!0}),a(no)!=="svelte-1423b5j"&&(no.textContent=Zd),er.forEach(i),$a=o($),Je=s($,"DIV",{class:!0});var jr=x(Je);p(qt.$$.fragment,jr),Pa=o(jr),oo=s(jr,"P",{"data-svelte-h":!0}),a(oo)!=="svelte-dk6kyv"&&(oo.textContent=Dd),jr.forEach(i),$.forEach(i),br=o(t),p(Ct.$$.fragment,t),Tr=o(t),M=s(t,"DIV",{class:!0});var q=x(M);p(It.$$.fragment,q),Ma=o(q),ro=s(q,"P",{"data-svelte-h":!0}),a(ro)!=="svelte-1cw2qez"&&(ro.innerHTML=Hd),qa=o(q),so=s(q,"P",{"data-svelte-h":!0}),a(so)!=="svelte-1df8ukr"&&(so.textContent=Sd),Ca=o(q),Z=s(q,"DIV",{class:!0});var ce=x(Z);p(Ft.$$.fragment,ce),Ia=o(ce),ao=s(ce,"P",{"data-svelte-h":!0}),a(ao)!=="svelte-1gpnz9t"&&(ao.textContent=Ad),Fa=o(ce),io=s(ce,"P",{"data-svelte-h":!0}),a(io)!=="svelte-hoi93q"&&(io.textContent=Gd),ja=o(ce),co=s(ce,"UL",{"data-svelte-h":!0}),a(co)!=="svelte-i7p8st"&&(co.innerHTML=Rd),Wa=o(ce),lo=s(ce,"P",{"data-svelte-h":!0}),a(lo)!=="svelte-udqgev"&&(lo.textContent=Xd),ce.forEach(i),Ja=o(q),D=s(q,"DIV",{class:!0});var le=x(D);p(jt.$$.fragment,le),Ua=o(le),po=s(le,"P",{"data-svelte-h":!0}),a(po)!=="svelte-mtqdue"&&(po.textContent=Yd),Na=o(le),mo=s(le,"P",{"data-svelte-h":!0}),a(mo)!=="svelte-hoi93q"&&(mo.textContent=Od),Ba=o(le),ho=s(le,"UL",{"data-svelte-h":!0}),a(ho)!=="svelte-2vo0y6"&&(ho.innerHTML=Qd),Va=o(le),uo=s(le,"P",{"data-svelte-h":!0}),a(uo)!=="svelte-udqgev"&&(uo.textContent=Kd),le.forEach(i),La=o(q),Ue=s(q,"DIV",{class:!0});var Wr=x(Ue);p(Wt.$$.fragment,Wr),Ea=o(Wr),fo=s(Wr,"P",{"data-svelte-h":!0}),a(fo)!=="svelte-pxfh9u"&&(fo.textContent=ec),Wr.forEach(i),Za=o(q),ie=s(q,"DIV",{class:!0});var tr=x(ie);p(Jt.$$.fragment,tr),Da=o(tr),go=s(tr,"P",{"data-svelte-h":!0}),a(go)!=="svelte-1839ko1"&&(go.textContent=tc),Ha=o(tr),_o=s(tr,"UL",{"data-svelte-h":!0}),a(_o)!=="svelte-1ap7xk7"&&(_o.innerHTML=nc),tr.forEach(i),Sa=o(q),Ne=s(q,"DIV",{class:!0});var Jr=x(Ne);p(Ut.$$.fragment,Jr),Aa=o(Jr),ko=s(Jr,"P",{"data-svelte-h":!0}),a(ko)!=="svelte-1uzoac3"&&(ko.innerHTML=oc),Jr.forEach(i),Ga=o(q),N=s(q,"DIV",{class:!0});var R=x(N);p(Nt.$$.fragment,R),Ra=o(R),bo=s(R,"P",{"data-svelte-h":!0}),a(bo)!=="svelte-xav75j"&&(bo.textContent=rc),Xa=o(R),To=s(R,"P",{"data-svelte-h":!0}),a(To)!=="svelte-5yd7mz"&&(To.innerHTML=sc),Ya=o(R),vo=s(R,"UL",{"data-svelte-h":!0}),a(vo)!=="svelte-4fpj7o"&&(vo.innerHTML=ac),Oa=o(R),xo=s(R,"P",{"data-svelte-h":!0}),a(xo)!=="svelte-hoi93q"&&(xo.textContent=ic),Qa=o(R),yo=s(R,"UL",{"data-svelte-h":!0}),a(yo)!=="svelte-10t7hv1"&&(yo.innerHTML=dc),R.forEach(i),Ka=o(q),H=s(q,"DIV",{class:!0});var pe=x(H);p(Bt.$$.fragment,pe),ei=o(pe),wo=s(pe,"P",{"data-svelte-h":!0}),a(wo)!=="svelte-gouuon"&&(wo.innerHTML=cc),ti=o(pe),zo=s(pe,"P",{"data-svelte-h":!0}),a(zo)!=="svelte-hoi93q"&&(zo.textContent=lc),ni=o(pe),$o=s(pe,"UL",{"data-svelte-h":!0}),a($o)!=="svelte-q52scs"&&($o.innerHTML=pc),oi=o(pe),Po=s(pe,"P",{"data-svelte-h":!0}),a(Po)!=="svelte-1fj0zmd"&&(Po.textContent=mc),pe.forEach(i),ri=o(q),S=s(q,"DIV",{class:!0});var me=x(S);p(Vt.$$.fragment,me),si=o(me),Mo=s(me,"P",{"data-svelte-h":!0}),a(Mo)!=="svelte-a0a2o0"&&(Mo.textContent=hc),ai=o(me),qo=s(me,"P",{"data-svelte-h":!0}),a(qo)!=="svelte-hoi93q"&&(qo.textContent=uc),ii=o(me),Co=s(me,"UL",{"data-svelte-h":!0}),a(Co)!=="svelte-vbk04m"&&(Co.innerHTML=fc),di=o(me),Io=s(me,"P",{"data-svelte-h":!0}),a(Io)!=="svelte-1fj0zmd"&&(Io.textContent=gc),me.forEach(i),ci=o(q),Be=s(q,"DIV",{class:!0});var Ur=x(Be);p(Lt.$$.fragment,Ur),li=o(Ur),Fo=s(Ur,"P",{"data-svelte-h":!0}),a(Fo)!=="svelte-or6vgj"&&(Fo.textContent=_c),Ur.forEach(i),pi=o(q),Ve=s(q,"DIV",{class:!0});var Nr=x(Ve);p(Et.$$.fragment,Nr),mi=o(Nr),jo=s(Nr,"P",{"data-svelte-h":!0}),a(jo)!=="svelte-1xfwjqh"&&(jo.textContent=kc),Nr.forEach(i),hi=o(q),B=s(q,"DIV",{class:!0});var X=x(B);p(Zt.$$.fragment,X),ui=o(X),Wo=s(X,"P",{"data-svelte-h":!0}),a(Wo)!=="svelte-1p4j7cm"&&(Wo.textContent=bc),fi=o(X),Jo=s(X,"P",{"data-svelte-h":!0}),a(Jo)!=="svelte-1m5vplt"&&(Jo.textContent=Tc),gi=o(X),Uo=s(X,"UL",{"data-svelte-h":!0}),a(Uo)!=="svelte-tivk04"&&(Uo.innerHTML=vc),_i=o(X),No=s(X,"P",{"data-svelte-h":!0}),a(No)!=="svelte-hoi93q"&&(No.textContent=xc),ki=o(X),Bo=s(X,"UL",{"data-svelte-h":!0}),a(Bo)!=="svelte-rgo9bl"&&(Bo.innerHTML=yc),X.forEach(i),bi=o(q),U=s(q,"DIV",{class:!0});var A=x(U);p(Dt.$$.fragment,A),Ti=o(A),Vo=s(A,"P",{"data-svelte-h":!0}),a(Vo)!=="svelte-p6ca34"&&(Vo.textContent=wc),vi=o(A),Lo=s(A,"P",{"data-svelte-h":!0}),a(Lo)!=="svelte-13kv6y9"&&(Lo.innerHTML=zc),xi=o(A),Eo=s(A,"UL",{"data-svelte-h":!0}),a(Eo)!=="svelte-adw1ie"&&(Eo.innerHTML=$c),yi=o(A),Zo=s(A,"P",{"data-svelte-h":!0}),a(Zo)!=="svelte-hoi93q"&&(Zo.textContent=Pc),wi=o(A),Do=s(A,"UL",{"data-svelte-h":!0}),a(Do)!=="svelte-11iunx9"&&(Do.innerHTML=Mc),zi=o(A),Ho=s(A,"P",{"data-svelte-h":!0}),a(Ho)!=="svelte-udqgev"&&(Ho.textContent=qc),A.forEach(i),$i=o(q),Le=s(q,"DIV",{class:!0});var Br=x(Le);p(Ht.$$.fragment,Br),Pi=o(Br),So=s(Br,"P",{"data-svelte-h":!0}),a(So)!=="svelte-1xfwjqh"&&(So.textContent=Cc),Br.forEach(i),q.forEach(i),vr=o(t),p(St.$$.fragment,t),xr=o(t),or=s(t,"P",{}),x(or).forEach(i),this.h()},h(){y(c,"name","hf:doc:metadata"),y(c,"content",Gc),y(fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(xe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(Le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),y(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(t,g){e(document.head,c),k(t,I,g),k(t,b,g),k(t,_,g),m(P,t,g),k(t,d,g),k(t,C,g),k(t,sr,g),k(t,Ze,g),k(t,ar,g),k(t,De,g),k(t,ir,g),k(t,He,g),k(t,dr,g),k(t,Se,g),k(t,cr,g),k(t,Ae,g),k(t,lr,g),m(Ge,t,g),k(t,pr,g),k(t,Re,g),k(t,mr,g),k(t,Xe,g),k(t,hr,g),m(Ye,t,g),k(t,ur,g),m(Oe,t,g),k(t,fr,g),k(t,T,g),m(Qe,T,null),e(T,Vr),e(T,Yt),e(T,Lr),e(T,Ot),e(T,Er),e(T,Qt),e(T,Zr),e(T,Kt),e(T,Dr),e(T,en),e(T,Hr),e(T,tn),e(T,Sr),e(T,fe),m(Ke,fe,null),e(fe,Ar),e(fe,nn),e(T,Gr),e(T,L),m(et,L,null),e(L,Rr),e(L,on),e(L,Xr),e(L,rn),e(L,Yr),e(L,sn),e(L,Or),m(ge,L,null),e(T,Qr),e(T,F),m(tt,F,null),e(F,Kr),e(F,an),e(F,es),e(F,dn),e(F,ts),e(F,cn),e(F,ns),e(F,ln),e(F,os),e(F,pn),e(F,rs),e(F,mn),e(F,ss),m(_e,F,null),e(T,as),e(T,ke),m(nt,ke,null),e(ke,is),e(ke,hn),e(T,ds),e(T,be),m(ot,be,null),e(be,cs),e(be,un),e(T,ls),e(T,Y),m(rt,Y,null),e(Y,ps),e(Y,fn),e(Y,ms),e(Y,gn),e(T,hs),e(T,O),m(st,O,null),e(O,us),e(O,_n),e(O,fs),e(O,kn),e(T,gs),e(T,Q),m(at,Q,null),e(Q,_s),e(Q,bn),e(Q,ks),m(Te,Q,null),e(T,bs),e(T,ve),m(it,ve,null),e(ve,Ts),e(ve,Tn),e(T,vs),e(T,xe),m(dt,xe,null),e(xe,xs),e(xe,vn),e(T,ys),e(T,ye),m(ct,ye,null),e(ye,ws),e(ye,xn),e(T,zs),e(T,K),m(lt,K,null),e(K,$s),e(K,yn),e(K,Ps),m(we,K,null),e(T,Ms),e(T,ee),m(pt,ee,null),e(ee,qs),e(ee,wn),e(ee,Cs),e(ee,zn),e(T,Is),e(T,te),m(mt,te,null),e(te,Fs),e(te,$n),e(te,js),e(te,Pn),k(t,gr,g),m(ht,t,g),k(t,_r,g),k(t,ut,g),k(t,kr,g),k(t,v,g),m(ft,v,null),e(v,Ws),e(v,Mn),e(v,Js),e(v,qn),e(v,Us),e(v,Cn),e(v,Ns),e(v,In),e(v,Bs),e(v,Fn),e(v,Vs),e(v,jn),e(v,Ls),e(v,ze),m(gt,ze,null),e(ze,Es),e(ze,Wn),e(v,Zs),e(v,E),m(_t,E,null),e(E,Ds),e(E,Jn),e(E,Hs),e(E,Un),e(E,Ss),e(E,Nn),e(E,As),m($e,E,null),e(v,Gs),e(v,j),m(kt,j,null),e(j,Rs),e(j,Bn),e(j,Xs),e(j,Vn),e(j,Ys),e(j,Ln),e(j,Os),e(j,En),e(j,Qs),e(j,Zn),e(j,Ks),e(j,Dn),e(j,ea),m(Pe,j,null),e(v,ta),e(v,Me),m(bt,Me,null),e(Me,na),e(Me,Hn),e(v,oa),e(v,qe),m(Tt,qe,null),e(qe,ra),e(qe,Sn),e(v,sa),e(v,ne),m(vt,ne,null),e(ne,aa),e(ne,An),e(ne,ia),e(ne,Gn),e(v,da),e(v,oe),m(xt,oe,null),e(oe,ca),e(oe,Rn),e(oe,la),e(oe,Xn),e(v,pa),e(v,re),m(yt,re,null),e(re,ma),e(re,Yn),e(re,ha),m(Ce,re,null),e(v,ua),e(v,Ie),m(wt,Ie,null),e(Ie,fa),e(Ie,On),e(v,ga),e(v,Fe),m(zt,Fe,null),e(Fe,_a),e(Fe,Qn),e(v,ka),e(v,je),m($t,je,null),e(je,ba),e(je,Kn),e(v,Ta),e(v,se),m(Pt,se,null),e(se,va),e(se,eo),e(se,xa),m(We,se,null),e(v,ya),e(v,ae),m(Mt,ae,null),e(ae,wa),e(ae,to),e(ae,za),e(ae,no),e(v,$a),e(v,Je),m(qt,Je,null),e(Je,Pa),e(Je,oo),k(t,br,g),m(Ct,t,g),k(t,Tr,g),k(t,M,g),m(It,M,null),e(M,Ma),e(M,ro),e(M,qa),e(M,so),e(M,Ca),e(M,Z),m(Ft,Z,null),e(Z,Ia),e(Z,ao),e(Z,Fa),e(Z,io),e(Z,ja),e(Z,co),e(Z,Wa),e(Z,lo),e(M,Ja),e(M,D),m(jt,D,null),e(D,Ua),e(D,po),e(D,Na),e(D,mo),e(D,Ba),e(D,ho),e(D,Va),e(D,uo),e(M,La),e(M,Ue),m(Wt,Ue,null),e(Ue,Ea),e(Ue,fo),e(M,Za),e(M,ie),m(Jt,ie,null),e(ie,Da),e(ie,go),e(ie,Ha),e(ie,_o),e(M,Sa),e(M,Ne),m(Ut,Ne,null),e(Ne,Aa),e(Ne,ko),e(M,Ga),e(M,N),m(Nt,N,null),e(N,Ra),e(N,bo),e(N,Xa),e(N,To),e(N,Ya),e(N,vo),e(N,Oa),e(N,xo),e(N,Qa),e(N,yo),e(M,Ka),e(M,H),m(Bt,H,null),e(H,ei),e(H,wo),e(H,ti),e(H,zo),e(H,ni),e(H,$o),e(H,oi),e(H,Po),e(M,ri),e(M,S),m(Vt,S,null),e(S,si),e(S,Mo),e(S,ai),e(S,qo),e(S,ii),e(S,Co),e(S,di),e(S,Io),e(M,ci),e(M,Be),m(Lt,Be,null),e(Be,li),e(Be,Fo),e(M,pi),e(M,Ve),m(Et,Ve,null),e(Ve,mi),e(Ve,jo),e(M,hi),e(M,B),m(Zt,B,null),e(B,ui),e(B,Wo),e(B,fi),e(B,Jo),e(B,gi),e(B,Uo),e(B,_i),e(B,No),e(B,ki),e(B,Bo),e(M,bi),e(M,U),m(Dt,U,null),e(U,Ti),e(U,Vo),e(U,vi),e(U,Lo),e(U,xi),e(U,Eo),e(U,yi),e(U,Zo),e(U,wi),e(U,Do),e(U,zi),e(U,Ho),e(M,$i),e(M,Le),m(Ht,Le,null),e(Le,Pi),e(Le,So),k(t,vr,g),m(St,t,g),k(t,xr,g),k(t,or,g),yr=!0},p(t,[g]){const w={};g&2&&(w.$$scope={dirty:g,ctx:t}),ge.$set(w);const At={};g&2&&(At.$$scope={dirty:g,ctx:t}),_e.$set(At);const G={};g&2&&(G.$$scope={dirty:g,ctx:t}),Te.$set(G);const W={};g&2&&(W.$$scope={dirty:g,ctx:t}),we.$set(W);const Gt={};g&2&&(Gt.$$scope={dirty:g,ctx:t}),$e.$set(Gt);const Rt={};g&2&&(Rt.$$scope={dirty:g,ctx:t}),Pe.$set(Rt);const he={};g&2&&(he.$$scope={dirty:g,ctx:t}),Ce.$set(he);const ue={};g&2&&(ue.$$scope={dirty:g,ctx:t}),We.$set(ue)},i(t){yr||(h(P.$$.fragment,t),h(Ge.$$.fragment,t),h(Ye.$$.fragment,t),h(Oe.$$.fragment,t),h(Qe.$$.fragment,t),h(Ke.$$.fragment,t),h(et.$$.fragment,t),h(ge.$$.fragment,t),h(tt.$$.fragment,t),h(_e.$$.fragment,t),h(nt.$$.fragment,t),h(ot.$$.fragment,t),h(rt.$$.fragment,t),h(st.$$.fragment,t),h(at.$$.fragment,t),h(Te.$$.fragment,t),h(it.$$.fragment,t),h(dt.$$.fragment,t),h(ct.$$.fragment,t),h(lt.$$.fragment,t),h(we.$$.fragment,t),h(pt.$$.fragment,t),h(mt.$$.fragment,t),h(ht.$$.fragment,t),h(ft.$$.fragment,t),h(gt.$$.fragment,t),h(_t.$$.fragment,t),h($e.$$.fragment,t),h(kt.$$.fragment,t),h(Pe.$$.fragment,t),h(bt.$$.fragment,t),h(Tt.$$.fragment,t),h(vt.$$.fragment,t),h(xt.$$.fragment,t),h(yt.$$.fragment,t),h(Ce.$$.fragment,t),h(wt.$$.fragment,t),h(zt.$$.fragment,t),h($t.$$.fragment,t),h(Pt.$$.fragment,t),h(We.$$.fragment,t),h(Mt.$$.fragment,t),h(qt.$$.fragment,t),h(Ct.$$.fragment,t),h(It.$$.fragment,t),h(Ft.$$.fragment,t),h(jt.$$.fragment,t),h(Wt.$$.fragment,t),h(Jt.$$.fragment,t),h(Ut.$$.fragment,t),h(Nt.$$.fragment,t),h(Bt.$$.fragment,t),h(Vt.$$.fragment,t),h(Lt.$$.fragment,t),h(Et.$$.fragment,t),h(Zt.$$.fragment,t),h(Dt.$$.fragment,t),h(Ht.$$.fragment,t),h(St.$$.fragment,t),yr=!0)},o(t){u(P.$$.fragment,t),u(Ge.$$.fragment,t),u(Ye.$$.fragment,t),u(Oe.$$.fragment,t),u(Qe.$$.fragment,t),u(Ke.$$.fragment,t),u(et.$$.fragment,t),u(ge.$$.fragment,t),u(tt.$$.fragment,t),u(_e.$$.fragment,t),u(nt.$$.fragment,t),u(ot.$$.fragment,t),u(rt.$$.fragment,t),u(st.$$.fragment,t),u(at.$$.fragment,t),u(Te.$$.fragment,t),u(it.$$.fragment,t),u(dt.$$.fragment,t),u(ct.$$.fragment,t),u(lt.$$.fragment,t),u(we.$$.fragment,t),u(pt.$$.fragment,t),u(mt.$$.fragment,t),u(ht.$$.fragment,t),u(ft.$$.fragment,t),u(gt.$$.fragment,t),u(_t.$$.fragment,t),u($e.$$.fragment,t),u(kt.$$.fragment,t),u(Pe.$$.fragment,t),u(bt.$$.fragment,t),u(Tt.$$.fragment,t),u(vt.$$.fragment,t),u(xt.$$.fragment,t),u(yt.$$.fragment,t),u(Ce.$$.fragment,t),u(wt.$$.fragment,t),u(zt.$$.fragment,t),u($t.$$.fragment,t),u(Pt.$$.fragment,t),u(We.$$.fragment,t),u(Mt.$$.fragment,t),u(qt.$$.fragment,t),u(Ct.$$.fragment,t),u(It.$$.fragment,t),u(Ft.$$.fragment,t),u(jt.$$.fragment,t),u(Wt.$$.fragment,t),u(Jt.$$.fragment,t),u(Ut.$$.fragment,t),u(Nt.$$.fragment,t),u(Bt.$$.fragment,t),u(Vt.$$.fragment,t),u(Lt.$$.fragment,t),u(Et.$$.fragment,t),u(Zt.$$.fragment,t),u(Dt.$$.fragment,t),u(Ht.$$.fragment,t),u(St.$$.fragment,t),yr=!1},d(t){t&&(i(I),i(b),i(_),i(d),i(C),i(sr),i(Ze),i(ar),i(De),i(ir),i(He),i(dr),i(Se),i(cr),i(Ae),i(lr),i(pr),i(Re),i(mr),i(Xe),i(hr),i(ur),i(fr),i(T),i(gr),i(_r),i(ut),i(kr),i(v),i(br),i(Tr),i(M),i(vr),i(xr),i(or)),i(c),f(P,t),f(Ge,t),f(Ye,t),f(Oe,t),f(Qe),f(Ke),f(et),f(ge),f(tt),f(_e),f(nt),f(ot),f(rt),f(st),f(at),f(Te),f(it),f(dt),f(ct),f(lt),f(we),f(pt),f(mt),f(ht,t),f(ft),f(gt),f(_t),f($e),f(kt),f(Pe),f(bt),f(Tt),f(vt),f(xt),f(yt),f(Ce),f(wt),f(zt),f($t),f(Pt),f(We),f(Mt),f(qt),f(Ct,t),f(It),f(Ft),f(jt),f(Wt),f(Jt),f(Ut),f(Nt),f(Bt),f(Vt),f(Lt),f(Et),f(Zt),f(Dt),f(Ht),f(St,t)}}}const Gc='{"title":"Tokenizer","local":"tokenizer","sections":[],"depth":1}';function Rc(J){return jc(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class nl extends Wc{constructor(c){super(),Jc(this,c,Rc,Ac,Fc,{})}}export{nl as component};
