import torch
import torch.nn as nn
from profiler import WallTime
from random import randint
import random
from ffn_kernel_bf16 import last_matmul
random.seed(0)
torch.random.manual_seed(0)


def generate_mask(batch_size, seq_len, image_size):
    visual_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device='cuda')
    mid = randint(a=image_size // 2 + 1, b=visual_mask.numel() - image_size // 2 - 1)
    low = mid - image_size // 2
    high = mid + image_size // 2
    visual_mask[:, low:high] = True
    return visual_mask


def profile_module(name, module, inputs, num_trials=10):
    profile = WallTime(f"{name}", cuda=0)
    outs = []
    for _ in range(num_trials):
        with profile:
            out = module(*inputs)
            outs.append(out)
    profile.result(detail=True)
    return outs


class Torch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    @torch.no_grad()
    def forward(self, x, visual_mask):
        visual_mask = visual_mask[:, :, None]
        return self.linear_v(x) * visual_mask + self.linear_t(x) * (~visual_mask)


class Triton(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_v = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_t = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, visual_mask):
        batch_size = x.shape[0]
        return last_matmul(
            x.flatten(0,1),
            self.linear_v.weight.data.T,
            self.linear_t.weight.data.T,
            visual_mask.flatten(),
        ).unflatten(0, (batch_size, -1))


torch.cuda.set_device(0)

batch_size = 16
embed_dim = 1024
image_size = 1024

# create modules
pytorch_module = Torch(embed_dim, embed_dim).cuda().bfloat16()
triton_module = Triton(embed_dim, embed_dim).cuda().bfloat16()

# clone weights
triton_module.linear_v.weight.data = pytorch_module.linear_v.weight.data.clone()
triton_module.linear_t.weight.data = pytorch_module.linear_t.weight.data.clone()

for seq_len in [2048 * i for i in range(1, 101)]:
    x = torch.rand((batch_size, seq_len, embed_dim), dtype=torch.bfloat16, device='cuda')
    visual_mask = generate_mask(batch_size, seq_len, image_size)

    inputs = (x, visual_mask)
    outs1 = profile_module(f"pytorch-{seq_len}", pytorch_module, inputs)
    outs2 = profile_module(f"triton-{seq_len}", triton_module, inputs)
    print("="*10)
