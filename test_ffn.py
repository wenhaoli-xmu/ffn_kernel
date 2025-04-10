import torch
import torch.nn as nn
from profiler import WallTime
from random import randint
import random
from ffn_kernel import ffn_bf16, ffn_fp32
import IPython
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
    def __init__(self, hidden_size, intermediate_size, fp32_kernel=False):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.u1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.u2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.u3 = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.fp32_kernel = fp32_kernel

    @torch.no_grad()
    def forward(self, x, mask):
        batch_size = x.shape[0]
        return (ffn_fp32 if self.fp32_kernel else ffn_bf16)(
            x.flatten(0,1),
            self.w1.weight.data.T,
            self.w2.weight.data.T,
            self.w3.weight.data.T,
            self.u1.weight.data.T,
            self.u2.weight.data.T,
            self.u3.weight.data.T,
            mask.flatten(0,1)
        ).unflatten(0, (batch_size, -1))


if __name__ == "__main__":
    import argparse
    from pygments.console import colorize

    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--check", action='store_true')
    parser.add_argument("--autotune", action='store_true')
    parser.add_argument("--fp32", action='store_true')
    args = parser.parse_args()

    assert ~(args.check & args.autotune)

    torch.cuda.set_device(0)

    batch_size = args.bsz
    embed_dim = 1024
    intermediate_dim = 2048
    image_size = 1024

    # create modules
    pytorch_module = Torch(embed_dim, intermediate_dim).cuda()
    triton_module = Triton(embed_dim, intermediate_dim, fp32_kernel=args.fp32).cuda()
    if not args.fp32:
        pytorch_module = pytorch_module.to(torch.bfloat16)
        triton_module = triton_module.to(torch.bfloat16)

    # clone weights
    triton_module.w1.weight.data = pytorch_module.w1.weight.data.clone()
    triton_module.w2.weight.data = pytorch_module.w2.weight.data.clone()
    triton_module.w3.weight.data = pytorch_module.w3.weight.data.clone()
    triton_module.u1.weight.data = pytorch_module.u1.weight.data.clone()
    triton_module.u2.weight.data = pytorch_module.u2.weight.data.clone()
    triton_module.u3.weight.data = pytorch_module.u3.weight.data.clone()

    if args.autotune:
        import os
        os.environ['TRITON_PRINT_AUTOTUNING'] = "1"

    for seq_len in [2048 * i for i in range(1, 32)]:
        x = torch.rand((batch_size, seq_len, embed_dim), dtype=torch.bfloat16 if not args.fp32 else torch.float32, device='cuda')
        mask = generate_mask(batch_size, seq_len, image_size)
        inputs = (x, mask)

        outs1 = profile_module(f"pytorch-{seq_len}", pytorch_module, inputs)
        outs2 = profile_module(f"triton-{seq_len}", triton_module, inputs)

        if args.check:
            print(colorize('green', 'pytorch:'), outs1[0])
            print(colorize('green', 'triton:'), outs2[0])
            print(colorize('green', 'l2-norm distance: ') + f"{torch.dist(outs1[0], outs2[0])}")

        if args.autotune or args.check:
            IPython.embed()

        print("="*10)
