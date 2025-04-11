# Installation

```bash
git clone https://github.com/wenhaoli-xmu/not-important
cd not-important
pip install .
```

# Quick Start

## Search Optimal Kernel Configs (Optional)

The default kernel config is searched under A100 80G
If you want to find the configs that runs the fastest on your machine, follow these steps:

1. Run latency benchmarks to get the optimal configs:

    ```bash
    python find_spec/find_fwd.py
    python find_spec/find_bwd_da.py
    python find_spec/find_bwd_db.py
    ```
    
2. Apply the optimal configs:
    Assume the optimal config for the forward kernel is `(16,16,128,1)`
    First, locate to `ffn_kernel->linear.py->_bf16_linear_forward`, then change the corresponding constant:
    ```python
    BLOCK_SIZE_M=16
    BLOCK_SIZE_N=16
    BLOCK_SIZE_K=128
    GROUP_SIZE_M=1
    ```

## Linear

```python
from ffn_kernel import linear_bf16, linear_fp32


class TorchLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    @torch.no_grad()
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
        return (linear_bf16 if x.dtype == torch.bfloat16 else linear_fp32)(
            x.flatten(0,1),
            self.linear_v.weight.data.T,
            self.linear_t.weight.data.T,
            visual_mask.flatten(),
        ).unflatten(0, (batch_size, -1))
```

## FFN

```python
from ffn_kernel import ffn_fp16, ffn_bf16, ffn_fp32


class TorchFFN(nn.Module):
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


class TritonFFN(nn.Module):
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
        """
        Parameters
        ----------
        :x: (bsz, seq_len, embed_dim)
        :mask: (bsz, seq_len)
        """
        batch_size = x.shape[0]

        assert mask.dtype == torch.bool
        if x.dtype == torch.float16:
            f = ffn_fp16
        elif x.dtype == torch.bfloat16:
            f = ffn
        elif x.dtype == torch.float32:
            f = ffn_fp32

        return f(
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


# Precision Check

```bash
# download profiling tools
git clone https://github.com/wenhaoli-xmu/lm-profiler
cd lm-profiler
pip isntall -e .
pip isntall IPython
```

```bash
# check float32 kernel
python test_ffn.py --check --fp32 --bsz 1

# check bfloat16 kernel
python test_ffn.py --check --bsz 1
```
