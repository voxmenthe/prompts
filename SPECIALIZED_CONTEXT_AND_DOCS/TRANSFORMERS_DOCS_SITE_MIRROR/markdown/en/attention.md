# Attention mechanisms

Most transformer models use full attention in the sense that the attention matrix is square. It can be a big
computational bottleneck when you have long texts. Longformer and reformer are models that try to be more efficient and
use a sparse version of the attention matrix to speed up training.

## LSH attention

[Reformer](model_doc/reformer) uses LSH attention. In the softmax(QK^t), only the biggest elements (in the softmax
dimension) of the matrix QK^t are going to give useful contributions. So for each query q in Q, we can consider only
the keys k in K that are close to q. A hash function is used to determine if q and k are close. The attention mask is
modified to mask the current token (except at the first position), because it will give a query and a key equal (so
very similar to each other). Since the hash can be a bit random, several hash functions are used in practice
(determined by a n\_rounds parameter) and then are averaged together.

## Local attention

[Longformer](model_doc/longformer) uses local attention: often, the local context (e.g., what are the two tokens to the
left and right?) is enough to take action for a given token. Also, by stacking attention layers that have a small
window, the last layer will have a receptive field of more than just the tokens in the window, allowing them to build a
representation of the whole sentence.

Some preselected input tokens are also given global attention: for those few tokens, the attention matrix can access
all tokens and this process is symmetric: all other tokens have access to those specific tokens (on top of the ones in
their local window). This is shown in Figure 2d of the paper, see below for a sample attention mask:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local_attention_mask.png)

Using those attention matrices with less parameters then allows the model to have inputs having a bigger sequence
length.

## Other tricks

### Axial positional encodings

[Reformer](model_doc/reformer) uses axial positional encodings: in traditional transformer models, the positional encoding
E is a matrix of sizelll byddd,lll being the sequence length andddd the dimension of the
hidden state. If you have very long texts, this matrix can be huge and take way too much space on the GPU. To alleviate
that, axial positional encodings consist of factorizing that big matrix E in two smaller matrices E1 and E2, with
dimensionsl1×d1l\_{1} \times d\_{1}l1​×d1​ andl2×d2l\_{2} \times d\_{2}l2​×d2​, such thatl1×l2=ll\_{1} \times l\_{2} = ll1​×l2​=l andd1+d2=dd\_{1} + d\_{2} = dd1​+d2​=d (with the product for the lengths, this ends up being way smaller). The embedding for time
stepjjj in E is obtained by concatenating the embeddings for timestepj%l1j \% l1j%l1 in E1 andj//l1j // l1j//l1
in E2.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/attention.md)
