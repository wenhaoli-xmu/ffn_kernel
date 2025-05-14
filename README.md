# Installation

We use `torch==2.1.0+cu121` and `triton==3.2.0` as our experimental environment. 

```bash
git clone https://github.com/wenhaoli-xmu/ffn_kernel
cd ffn_kernel
pip install .
```

# (Optional) Search Kernel Configs

Our default kernel configs are searched based on A100 80G, using 1024 as the embed dimension for `MaskedLinear` and 1024 & 2048 for FFN's input and intermediate dimensions.

⚠️_Re-evaluating the optimal configs is required if the embed dimension or gpu device changed._

1. Run latency benchmarks to get the optimal configs:

    ```bash
    # this requirement is for latency recording
    git clone https://wenhaoli-xmu/lm-profiler.git
    cd lm-profiler && pip install -e .
    cd ..

    # --d refers to embed dim
    # --h refers to intermediate dim
    python find_spec/bf16_linear_fwd.py --d 1024
    python find_spec/bf16_ffn_fwd.py --d 1024 --h 2048
    ```
    
2. Apply the optimal configs:

    After running, the optimal configs are stored in json files, and you can run the following commands to copy them to the root directory for usage.
    ```bash
    mv bf16_linear_fwd.json ffn_kernel/
    mv bf16_linear_bwd.json ffn_kernel/
    ```

# Quick Start

## Linear

Both forward & backward propagations are supported by `MaskedLinear`, but the efficiency of backward is not as high as PyTorch, so we do not recommand using our kernel in training scenarios.

```python
from ffn_kernel import MaskedLinear


class TorchLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, visual_mask):
        visual_mask = visual_mask[:, :, None]
        return self.linear_v(x) * visual_mask + self.linear_t(x) * (~visual_mask)


class TritonLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, visual_mask):
        batch_size = x.shape[0]
        MaskedLinear.apply(
            x.flatten(0,1),
            self.linear_v.weight.data.T,
            self.linear_t.weight.data.T,
            visual_mask.flatten(),
        ).unflatten(0, (batch_size, -1))
```

## FFN

Only the forward propagation is supported.

```python
from ffn_kernel import MaskedFFN

class Torch(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.u1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.u2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.u3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    @torch.no_grad()
    def forward(self, x, mask):
        mask = mask[:, :, None]
        proj1 = torch.nn.functional.silu(self.w1(x)) * self.w3(x)
        proj2 = torch.nn.functional.silu(self.u1(x)) * self.u3(x)
        return self.w2(proj1) * mask + self.u2(proj2) * ~mask


class Triton(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.u1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.u2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.u3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    @torch.no_grad()
    def forward(self, x, mask):
        batch_size = x.shape[0]
        return MaskedFFN.apply(
            x.flatten(0,1),
            self.w1.weight.data.T,
            self.w2.weight.data.T,
            self.w3.weight.data.T,
            self.u1.weight.data.T,
            self.u2.weight.data.T,
            self.u3.weight.data.T,
            mask.flatten(0,1)
        ).unflatten(0, (batch_size, -1))
```

# Profile


Prepare required packages:
```bash
# download profiling tools
git clone https://github.com/wenhaoli-xmu/lm-profiler
cd lm-profiler
pip isntall -e .
pip isntall IPython
```

Then, you can profile the kernels by runing:
```bash
python test_ffn.py --bsz 1
python test_linear.py --bsz 1
```

# Precision Check

```bash
python test_ffn.py --check
python test_linear.py --check
```

# Implementing Details

Forward prop formula of FFN:

$$
C=T_1\sigma(T_1)\odot T_2= (xW_1)\sigma(xW_1) \odot \left(xW_3\right)
$$

Backward prop formula of FFN:

$$
\begin{align}
\text{d}W_3&=x^\top{\color{blue}(G\odot T_1\sigma(T_1))}, \quad (k,m),(m,n)\\
\text{d}W_1&= x^\top {\color{blue}\left[G\odot T_2\odot (T_1+T_1\sigma(T_1)-T_1^2\sigma(T_1))\right]}, \quad (k,m),(m,n) \\
\text{d}x&=\left[\color{blue}G\odot T_2\odot (T_1+T_1\sigma(T_1)-T_1^2\sigma(T_1))\right]W_1^T+{\color{blue}(G\odot T_1\sigma(T_1))}W_3^\top,\quad (m,n),(n,k)
\end{align} 
$$

Strategy:

1. Calculate blue part using elem-wise kernel. 
2. Launch fused kernel to calculate final results.

