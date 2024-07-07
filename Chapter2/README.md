# Chapter2 Large Language Model Fundamentals

## 2.1 Transformer Model

* `Attention Layer`：Integrate contextual semantics using multi-head attention mechanisms.
* `Position-wise FFN`：Perform more complex transformations on the representation of each word in the input text sequence through fully connected layers.
* `Residual Connection`：Corresponding to the `Add` module in the middle. This make the information flow more efficient and benefits model optimization.
* `Layer normalization`：Corresponding to the `Norm` module in the middle.  Perform layer normalizaiton on the representation sequence, which also stabilizes the optimization.

<img src="./assets/transformer_architecture.png" alt="image-20240705194340263" style="zoom:50%;" />

### 2.1.1 Enbedding Representation Layer

For each input text sequence, first convert each word into its corresponding vector representation through the input embedding layer. Before sending it to encoder for modeling its contextual sematics, a very significant operation is to add positional encoding features to the word embeddings and then send to subsequent modules for futher processing. Transformer uses sine and cosine fuctions of different frequencies to set the positional encoding.
$$
\text{PE}(\text{pos}, 2i) = \sin(\frac{\text{pos}}{10000^{2i/d}}) \\
\text{PE}(\text{pos}, 2i + 1) = \cos(\frac{\text{pos}}{10000^{2i/d}})
$$
where $pos$ represents the position of the word, $2i$ and $2i+1$ represent the corresponding dimensions in the positional encoding vector, and $d$ corresponds to the total dimensions of the positional encoding.

>THINK： WHY IS POSITION ENCODING DESIGNED LIKE THIS. 
>
>* The range of sine and cosine fuction is $[-1,+1]$, so adding the positional encoding to the original word embeddings will not cause the results to deviate too far, thereby preserving the original semantic information of the words.
>* According to the basic properties of trigonometric functions, it can be known that the coding of the $pos + k$ position is a linear combination of the coding of the pos position, which means that the position coding contains the distance information between words. ??

Implementation of positional encoding using `Pytorch`

```python
# ./code/PositionalEncoder.py
import torch 
import torch.nn as nn
import math
import numpy as np

class MyPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # Create a matrix of shape (max_seq_len, d_model) with positional encodings
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Convert to PyTorch tensor
        pe = torch.tensor(pe, dtype=torch.float32).unsqueeze(0)
        # Register buffer to avoid updating during training
        self.register_buffer('pe', pe)
	
    def forward(self, x):   # x.shape == (batch_size, seq_size, d_hid)
        return x + self.pe[:, :x.size(1)].clone().detach()
```

### 2.1.2 Attention Layer

To further model dependencies on contextual semantics, it is essential to introduce three elements involved in self-attention mechanisms: Query, Key and Value. In the process of encoding the representation of each word in the input sequence, these three elements are used to calculate the weight score corresponding to the context word. Specifically, through three linear transformations $W^Q \in R^{d\times d_q},W^K \in R^{d\times d_k},W^V \in R^{d\times d_v}$, each word in the input sequence is represented by $x_i$, which is converted into its corresponding $q_i \in R^{d_k}, k_i \in R^{d_k}, v_i \in R^{d_v}$

In order to obtain the context information that needs to be paid attention to when encoding the word $x_i$, the matching scores $q_i \cdot k_1, q_i \cdot k_2,...,q_i \cdot k_t$. In order to prevent the gradient explosion and poor convergence efficiency caused by excessive matching scores in the subsquence `Softmax` calulation process. These scores are divided by the scaling factor $\sqrt{d}$ to stablize the optimization.
$$
Z = \text{Attention}(Q, K, V) = \text{Softmax}(\frac{QK^T}{\sqrt{d}})V
$$


