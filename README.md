# Installation

```bash
git clone https://github.com/wenhaoli-xmu/lm-profiler
cd lm-profiler
pip isntall -e .

# clone this repo
cd not-important
pip install .
```

# Quick Start

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
```

```bash
# check float32 kernel
python test_ffn.py --check --fp32 --bsz 1

# check bfloat16 kernel
python test_ffn.py --check --bsz 1
```
